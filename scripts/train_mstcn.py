#!/usr/bin/env python3
"""
Train an MS-TCN temporal head on frozen LemonFM cached features.

This is the parametric temporal baseline requested by the supervisor:
the same frozen backbone, same splits, same per-video training regime as
train_stag.py, but with MS-TCN (Farha & Gall 2019 / TeCNO 2020) as the head
instead of STAG.  Comparing the two directly isolates the value of STAG's
non-parametric affinity-graph temporal module vs. a standard learned TCN.

Usage:
    python scripts/train_mstcn.py --root . --task cholec80_phase --epochs 50
    python scripts/train_mstcn.py --root . --task cholec80_tool  --epochs 50
    python scripts/train_mstcn.py --root . --task grasp_tool     --epochs 50
    python scripts/train_mstcn.py --root . --task cholect50_verb --fold 1 --epochs 50

Run layout (parallel to train_stag.py):
    runs/<task_slug>/mstcn/<timestamp>/
        config.json, log.txt, metrics.jsonl, best.json,
        checkpoint_best.pt, predictions_test.npz, run_summary.txt
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (   # noqa: E402
    load_labels, load_splits, get_split_partitions,
    preload_features, to_torch_labels,
)
from metrics import phase_metrics, multilabel_metrics   # noqa: E402
from mstcn_head import build_mstcn, MSTCN_RUN_NAME   # noqa: E402


# Task configs - must match train_stag.py's TASK_CONFIGS
TASK_CONFIGS = {
    "cholec80_phase": {"label_file": "preprocessed/labels/cholec80_phase.json",
                       "feature_group": "cholec80", "primary": "accuracy"},
    "cholec80_tool":  {"label_file": "preprocessed/labels/cholec80_tool.json",
                       "feature_group": "cholec80", "primary": "mAP"},
    "cholect50_verb": {"label_file": "preprocessed/labels/cholect50_verb.json",
                       "feature_group": "cholect50", "primary": "mAP"},
    "grasp_tool":     {"label_file": "preprocessed/labels/grasp_tool.json",
                       "feature_group": "grasp",    "primary": "mAP"},
}


# ------------------------------------------------------------------- helpers ---

class Tee:
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except: pass

    def flush(self):
        for st in self.streams:
            try: st.flush()
            except: pass


def _slug(task, fold):
    return task if fold is None else f"{task}_fold{fold}"


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _fp32(arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.float32)


# ----------------------------------------------------------------- loss / eval ---

def compute_loss_mstcn(all_outputs, labels, label_type):
    """
    Multi-stage loss: sum of per-stage loss + smoothing loss on final stage.
    all_outputs: list of [T, N_c] tensors (one per stage).
    """
    total = torch.tensor(0.0, device=labels.device)
    for logits in all_outputs:
        if label_type == "multi-class":
            mask = labels >= 0
            if mask.sum() == 0: continue
            total = total + F.cross_entropy(logits[mask], labels[mask])
        else:
            mask = labels[:, 0] >= 0
            if mask.sum() == 0: continue
            total = total + F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float())

    # Smoothing loss on the final stage (standard MS-TCN training detail)
    final = all_outputs[-1]
    if final.shape[0] > 1:
        smooth = torch.mean(
            torch.clamp(
                (F.log_softmax(final[1:], dim=-1) -
                 F.log_softmax(final[:-1], dim=-1)) ** 2,
                min=0, max=16.0,
            )
        )
        total = total + 0.15 * smooth

    return total


def run_epoch(model, feats, labels, vids, optimizer, label_type, device,
              train: bool, collect_predictions: bool = False):
    model.train() if train else model.eval()
    order = list(vids)
    if train:
        np.random.shuffle(order)

    losses = []
    y_pred_chunks, y_true_chunks, video_offsets, video_ids_used = [], [], [0], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for vid in order:
            if vid not in feats or vid not in labels:
                continue
            pooled = _fp32(feats[vid]["pooled"], device)   # [T, 1536]
            y = labels[vid].to(device)

            all_outputs = model.all_stage_outputs(pooled)  # list of [T, N_c]

            if train:
                loss = compute_loss_mstcn(all_outputs, y, label_type)
                if loss is not None and loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
            else:
                logits = all_outputs[-1]  # final stage
                if label_type == "multi-class":
                    yp = logits.argmax(dim=-1).cpu().numpy()
                else:
                    yp = torch.sigmoid(logits).cpu().numpy()
                yt = y.cpu().numpy()
                y_pred_chunks.append(yp)
                y_true_chunks.append(yt)
                video_offsets.append(video_offsets[-1] + yp.shape[0])
                video_ids_used.append(vid)

    if train:
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    if not y_pred_chunks:
        return None

    y_pred = np.concatenate(y_pred_chunks, axis=0)
    y_true = np.concatenate(y_true_chunks, axis=0)
    metrics = (phase_metrics(y_pred, y_true) if label_type == "multi-class"
               else multilabel_metrics(y_pred, y_true))

    if collect_predictions:
        return metrics, y_pred, y_true, video_offsets, video_ids_used
    return metrics


# ---------------------------------------------------------------------- main ---

def main(args):
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(os.path.join(root, args.out_dir))
    cfg = TASK_CONFIGS[args.task]

    # Run directory (parallel to train_stag: runs/baselines/<head>/<slug>/<causality>/<ts>/)
    # Run directory: runs/baselines/<head>/<slug>/<causal|acausal>/<ts>/
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slug(args.task, args.fold)
    head_name = MSTCN_RUN_NAME.get(args.task, "mstcn")
    causality = "causal" if args.causal else "acausal"
    run_dir = os.path.join(out_dir, "baselines", head_name, slug, causality, ts)
    os.makedirs(run_dir, exist_ok=True)

    log_fh = open(os.path.join(run_dir, "log.txt"), "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[time] {datetime.now().isoformat(timespec='seconds')}")
    print(f"[dev ] {device}  n_gpu={torch.cuda.device_count()}")
    print(f"[run ] {run_dir}")
    print(f"[task] {args.task}  fold={args.fold}")

    label_data = load_labels(os.path.join(root, cfg["label_file"]))
    num_classes = label_data["num_classes"]
    label_type  = label_data["label_type"]
    splits = load_splits(os.path.join(root, "preprocessed", "splits.json"))
    train_vids, val_vids, test_vids = get_split_partitions(splits, args.task, args.fold)
    print(f"[split] train={len(train_vids)}  val={len(val_vids)}  test={len(test_vids)}")

    all_vids = sorted(set(train_vids) | set(val_vids) | set(test_vids))
    feats = preload_features(
        os.path.join(root, "preprocessed", "features"),
        cfg["feature_group"], all_vids,
        load_spatial=False,   # MS-TCN uses only pooled features
        verbose=False,
    )
    labels = {}
    for vid in all_vids:
        vdata = label_data["videos"].get(vid)
        if vdata is None: continue
        labels[vid] = to_torch_labels(vdata["labels"], label_type)

    model = build_mstcn(args.task, num_classes, causal=args.causal).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {head_name} ({causality})  trainable params = {n_params/1e6:.3f} M")
    save_json(os.path.join(run_dir, "config.json"), {
        "cli_args": vars(args), "num_classes": num_classes,
        "label_type": label_type, "n_params": n_params,
        "head": head_name, "causality": causality,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    primary   = cfg["primary"]
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_ckpt    = os.path.join(run_dir, "checkpoint_best.pt")
    best_pred    = os.path.join(run_dir, "predictions_test.npz")

    print(f"[train] epochs={args.epochs}  lr={args.lr}  primary={primary}")
    best_val, best_ep, best_test = -1.0, -1, None

    for ep in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, feats, labels, train_vids, optimizer,
                       label_type, device, train=True)
        va = run_epoch(model, feats, labels, val_vids,   optimizer,
                       label_type, device, train=False)
        te = run_epoch(model, feats, labels, test_vids,  optimizer,
                       label_type, device, train=False, collect_predictions=True)
        if isinstance(te, tuple):
            te_m, yp, yt, offsets, ids = te
        else:
            te_m, yp, yt, offsets, ids = te, None, None, None, None

        scheduler.step()
        val_s  = va[primary]  if va  else -1.0
        test_s = te_m[primary] if te_m else -1.0
        print(f"[ep {ep:03d}] loss={tr['loss']:.4f}  "
              f"val_{primary}={val_s*100:6.2f}  "
              f"test_{primary}={test_s*100:6.2f}  "
              f"({time.time()-t0:.1f}s)")
        append_jsonl(metrics_path, {
            "epoch": ep, "train_loss": tr["loss"],
            "val": va, "test": te_m, "lr": optimizer.param_groups[0]["lr"],
        })
        if val_s > best_val:
            best_val = val_s; best_ep = ep; best_test = te_m
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_score": val_s, "test_metrics": te_m}, best_ckpt)
            if yp is not None:
                np.savez_compressed(best_pred, y_pred=yp, y_true=yt,
                                    video_offsets=np.array(offsets),
                                    video_ids=np.array(ids))

    print("=" * 80)
    print(f"[best] epoch={best_ep}  val_{primary}={best_val*100:.2f}")
    if best_test:
        for k, v in best_test.items():
            print(f"  test.{k:20s} = {v*100:6.2f}")
    save_json(os.path.join(run_dir, "best.json"), {
        "best_epoch": best_ep, "best_val": best_val,
        "primary_metric": primary, "test_metrics": best_test,
        "task": args.task, "fold": args.fold,
        "class_names": label_data["class_names"],
    })
    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write(f"Head           : {head_name} ({causality})\n")
        f.write(f"Task           : {args.task}\nFold           : {args.fold}\n")
        f.write(f"Best epoch     : {best_ep}\nBest val       : {best_val*100:.2f}%\n")
        f.write(f"Trainable params: {n_params/1e6:.3f} M\n")
        if best_test:
            for k, v in best_test.items():
                f.write(f"  test.{k:20s} = {v*100:.2f}%\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root",    required=True)
    p.add_argument("--task",    required=True, choices=list(TASK_CONFIGS))
    p.add_argument("--fold",    type=int, default=None)
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--causal", action="store_true",
                   help="Use causal (one-directional) convolutions. "
                        "Default is acausal (bidirectional), matching STAG's "
                        "symmetric affinity window for a fair comparison.")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    main(args)
