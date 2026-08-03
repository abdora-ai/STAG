#!/usr/bin/env python3
"""
Train the STAG head on cached LemonFM features. Saves a full self-contained run
directory with config, log, per-epoch metrics, best-model checkpoint, and
final test predictions.

Run layout:
    runs/<task_slug>/<timestamp>/
        config.json           # CLI args + task hyperparameters
        log.txt               # mirror of everything printed to stdout / stderr
        metrics.jsonl         # per-epoch JSON metrics (append-only)
        best.json             # best-val summary + associated test metrics
        checkpoint_best.pt    # model + optimizer state at best-val epoch
        predictions_test.npz  # y_pred, y_true, video_offsets for the test split
        run_summary.txt       # human-readable final summary

Usage:
    python scripts/train_stag.py --root . --task cholec80_phase --epochs 50
    python scripts/train_stag.py --root . --task cholec80_tool  --epochs 50
    python scripts/train_stag.py --root . --task grasp_tool     --epochs 50
    python scripts/train_stag.py --root . --task cholect50_verb --fold 1 --epochs 50
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
from stag_head import STAG   # noqa: E402


TASK_CONFIGS = {
    "cholec80_phase": {
        "feature_group": "cholec80",
        "label_file":    "preprocessed/labels/cholec80_phase.json",
        "num_slots":     1,
        "window":        256,
        "depth":         8,
        "primary":       "accuracy",
    },
    "cholec80_tool": {
        "feature_group": "cholec80",
        "label_file":    "preprocessed/labels/cholec80_tool.json",
        "num_slots":     7,
        "window":        64,
        "depth":         4,
        "primary":       "mAP",
    },
    "cholect50_verb": {
        "feature_group": "cholect50",
        "label_file":    "preprocessed/labels/cholect50_verb.json",
        "num_slots":     10,
        "window":        8,
        "depth":         1,
        "primary":       "mAP",
    },
    "grasp_tool": {
        "feature_group": "grasp",
        "label_file":    "preprocessed/labels/grasp_tool.json",
        "num_slots":     7,
        "window":        8,
        "depth":         2,
        "primary":       "mAP",
    },
}


# --------------------------------------------------------- I/O + logging ---

class Tee:
    """Duplicate stream writes to multiple file-like objects."""
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _slug(task, fold):
    return task if fold is None else f"{task}_fold{fold}"


def prepare_run_dir(out_dir, task, fold, mode, run_name=None):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = run_name if run_name else ts
    # runs/<task_slug>/<mode>/<ts>/
    run_dir = os.path.join(out_dir, _slug(task, fold), mode, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def start_logging(run_dir):
    log_path = os.path.join(run_dir, "log.txt")
    fh = open(log_path, "w", buffering=1)   # line-buffered
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    return fh


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


# ----------------------------------------------------- Data + evaluation ---

def _to_fp32_gpu(arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.float32)


def compute_loss(logits, labels, label_type):
    if label_type == "multi-class":
        mask = labels >= 0
        if mask.sum() == 0:
            return None
        return F.cross_entropy(logits[mask], labels[mask])
    mask = labels[:, 0] >= 0
    if mask.sum() == 0:
        return None
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def run_epoch(model, feats, labels, vids, optimizer, label_type, device,
              train: bool, collect_predictions: bool = False):
    if train:
        model.train()
    else:
        model.eval()

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
            pooled = _to_fp32_gpu(feats[vid]["pooled"], device)
            spatial = _to_fp32_gpu(feats[vid]["spatial"], device)
            y = labels[vid].to(device)

            logits = model(pooled, spatial)  # [T, N_c]

            if train:
                loss = compute_loss(logits, y, label_type)
                if loss is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
            else:
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


# ---------------------------------------------------------------- main ---

def main(args):
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(os.path.join(root, args.out_dir))
    cfg = TASK_CONFIGS[args.task]

    # Set up the run directory and logging first, so everything below is captured.
    run_dir = prepare_run_dir(out_dir, args.task, args.fold, args.mode,
                              run_name=args.run_name)
    start_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[time] start = {datetime.now().isoformat(timespec='seconds')}")
    print(f"[host] {os.uname().nodename}")
    print(f"[dev ] {device}  n_gpu={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"       gpu[{i}] = {torch.cuda.get_device_name(i)}")
    print(f"[run ] {run_dir}")
    print(f"[task] {args.task}  fold={args.fold}  mode={args.mode}")
    print()

    # Save config early so a killed run still leaves a trace.
    save_json(os.path.join(run_dir, "config.json"), {
        "cli_args": vars(args),
        "task_config": cfg,
        "run_dir": run_dir,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    label_file = os.path.join(root, cfg["label_file"])
    label_data = load_labels(label_file)
    num_classes = label_data["num_classes"]
    label_type = label_data["label_type"]
    class_names = label_data["class_names"]
    print(f"[label] {num_classes}-class {label_type}")
    print(f"[label] class names: {class_names}")

    splits = load_splits(os.path.join(root, "preprocessed", "splits.json"))
    train_vids, val_vids, test_vids = get_split_partitions(splits, args.task, args.fold)
    print(f"[split] train={len(train_vids)}  val={len(val_vids)}  test={len(test_vids)}")
    if args.verbose:
        print(f"[split] train ids: {train_vids}")
        print(f"[split] val   ids: {val_vids}")
        print(f"[split] test  ids: {test_vids}")

    features_root = os.path.join(root, "preprocessed", "features")
    group = cfg["feature_group"]
    all_vids = sorted(set(train_vids) | set(val_vids) | set(test_vids))
    print(f"[data ] preloading features for {len(all_vids)} videos from "
          f"{features_root}/{group}")
    feats = preload_features(features_root, group, all_vids,
                             load_spatial=True, verbose=args.verbose)

    labels = {}
    for vid in all_vids:
        vdata = label_data["videos"].get(vid)
        if vdata is None:
            print(f"  [warn] no label entry for {vid}")
            continue
        labels[vid] = to_torch_labels(vdata["labels"], label_type)

    # Build model
    model = STAG(
        in_dim=1536, num_classes=num_classes, task_type=label_type,
        num_slots=cfg["num_slots"], slot_dim=256,
        mode=args.mode,
        window=cfg["window"], depth=cfg["depth"],
        temperature=20.0, sigma_init=1.5, alpha_init=0.5,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] mode={args.mode}  num_slots={cfg['num_slots']}  "
          f"window={cfg['window']}  depth={cfg['depth']}")
    print(f"[model] use_slot={model.use_slot}  use_prototype={model.use_prototype}  "
          f"use_diffusion={model.use_diffusion}")
    print(f"[model] trainable params = {n_params/1e6:.3f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    primary = cfg["primary"]
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
    best_pred_path = os.path.join(run_dir, "predictions_test.npz")

    print(f"[train] epochs={args.epochs}  lr={args.lr}  primary={primary}")
    print(f"[log  ] {os.path.join(run_dir, 'log.txt')}")
    print(f"[log  ] {metrics_path}")
    print("=" * 100)

    best_val, best_ep, best_test = -1.0, -1, None

    for ep in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, feats, labels, train_vids, optimizer,
                                label_type, device, train=True)
        val_stats = run_epoch(model, feats, labels, val_vids, optimizer,
                              label_type, device, train=False)

        # Evaluate on test and (if best-val) collect its predictions to save.
        collect = False  # decide after val
        test_stats = run_epoch(model, feats, labels, test_vids, optimizer,
                               label_type, device, train=False,
                               collect_predictions=True)
        if isinstance(test_stats, tuple):
            test_metrics_dict, y_pred_test, y_true_test, offsets_test, test_ids = test_stats
        else:
            test_metrics_dict, y_pred_test, y_true_test, offsets_test, test_ids = (
                test_stats, None, None, None, None
            )

        scheduler.step()
        elapsed = time.time() - t0

        val_score = val_stats[primary] if val_stats else -1.0
        test_score = test_metrics_dict[primary] if test_metrics_dict else -1.0
        diag = model.diffusion_params()

        line = (
            f"[ep {ep:03d}] "
            f"loss={train_stats['loss']:.4f}  "
            f"val_{primary}={val_score*100:6.2f}  "
            f"test_{primary}={test_score*100:6.2f}  "
            f"α={diag['alpha']:.3f}  σ={diag['sigma']:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.1f}s)"
        )
        print(line)

        row = {
            "epoch": ep,
            "train_loss": train_stats["loss"],
            "val": val_stats,
            "test": test_metrics_dict,
            "alpha": diag["alpha"],
            "sigma": diag["sigma"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        append_jsonl(metrics_path, row)

        if val_score > best_val:
            best_val = val_score
            best_ep = ep
            best_test = test_metrics_dict

            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_score": val_score,
                "test_metrics": test_metrics_dict,
                "task": args.task,
                "fold": args.fold,
                "cfg": cfg,
            }, best_ckpt_path)

            if y_pred_test is not None:
                np.savez_compressed(
                    best_pred_path,
                    y_pred=y_pred_test,
                    y_true=y_true_test,
                    video_offsets=np.array(offsets_test),
                    video_ids=np.array(test_ids),
                )

    print("=" * 100)
    print(f"[best]  epoch={best_ep}  val_{primary}={best_val*100:.2f}")
    if best_test:
        for k, v in best_test.items():
            print(f"        test.{k:20s} = {v*100:6.2f}")

    # Final summary
    save_json(os.path.join(run_dir, "best.json"), {
        "best_epoch": best_ep,
        "best_val": best_val,
        "primary_metric": primary,
        "test_metrics": best_test,
        "task": args.task,
        "fold": args.fold,
        "num_classes": num_classes,
        "class_names": class_names,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    })

    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write(f"Task           : {args.task}\n")
        f.write(f"Fold           : {args.fold}\n")
        f.write(f"Mode           : {args.mode}\n")
        f.write(f"Primary metric : {primary}\n")
        f.write(f"Best epoch     : {best_ep}\n")
        f.write(f"Best val       : {best_val*100:.2f}%\n")
        f.write(f"Test metrics @ best-val epoch:\n")
        if best_test:
            for k, v in best_test.items():
                f.write(f"  {k:20s} = {v*100:6.2f}%\n")
        f.write(f"\nRun directory  : {run_dir}\n")
        f.write(f"Finished at    : {datetime.now().isoformat(timespec='seconds')}\n")

    print(f"\n[artifacts]  {run_dir}")
    print(f"             ├── config.json")
    print(f"             ├── log.txt")
    print(f"             ├── metrics.jsonl")
    print(f"             ├── best.json")
    print(f"             ├── run_summary.txt")
    print(f"             ├── checkpoint_best.pt")
    print(f"             └── predictions_test.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Repo/data root")
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--fold", type=int, default=None,
                   help="Fold 1..5 for cholect50_verb; ignored otherwise")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs",
                   help="Root directory for run artifacts (default: runs)")
    p.add_argument("--run-name", type=str, default=None,
                   help="Override the timestamped run name")
    p.add_argument("--mode", type=str, default="full",
                   choices=["linear", "prototype", "slot_proto", "full"],
                   help="Ablation mode. 'full' = complete STAG head.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    main(args)
