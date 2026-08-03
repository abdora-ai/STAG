#!/usr/bin/env python3
"""
Reproduce the paper's LemonFM linear-probing baseline (Table 2) using our
cached frozen features + shuffled mini-batches of frames.

This mirrors the paper's downstream training scripts:

    Cholec80 phase   : batch 512, LR 1e-3, AdamW, wd 1e-5, CE,  100 epochs
                       balanced-class sampler
    Cholec80 tool    : batch 256, LR 1e-4, Adam,  wd 1e-4, BCE,  50 epochs
    GraSP tool       : batch 256, LR 1e-4, Adam,  wd 1e-4, BCE,  50 epochs
    CholecT50 verb   : batch 256, LR 1e-4, Adam,  wd 1e-5, BCE,  50 epochs

The head is a plain `nn.Linear(1536, num_classes)` on top of the LemonFM
pooled feature - same architecture as the paper's linear-probing setup, only
the backbone forward is precomputed instead of run at every step.

Run directory follows the same convention as train_stag.py:
    runs/<task_slug>/paper_linear/<timestamp>/{config, log, metrics, best, ...}

Usage:
    python scripts/reproduce_baseline.py --root . --task cholec80_phase
    python scripts/reproduce_baseline.py --root . --task cholec80_tool
    python scripts/reproduce_baseline.py --root . --task grasp_tool
    python scripts/reproduce_baseline.py --root . --task cholect50_verb --fold 1
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_labels, load_splits, get_split_partitions   # noqa: E402
from metrics import phase_metrics, multilabel_metrics                    # noqa: E402


BASELINE_CONFIGS = {
    "cholec80_phase": {
        "feature_group":   "cholec80",
        "label_file":      "preprocessed/labels/cholec80_phase.json",
        "batch_size":      512,
        "lr":              1e-3,
        "epochs":          100,
        "opt":             "adamw",
        "weight_decay":    1e-5,
        "loss":            "ce",           # multi-class softmax
        "primary":         "accuracy",
        "balance_classes": True,
    },
    "cholec80_tool": {
        "feature_group":   "cholec80",
        "label_file":      "preprocessed/labels/cholec80_tool.json",
        "batch_size":      256,
        "lr":              1e-4,
        "epochs":          50,
        "opt":             "adam",
        "weight_decay":    1e-4,
        "loss":            "bce",
        "primary":         "mAP",
        "balance_classes": False,
    },
    "grasp_tool": {
        "feature_group":   "grasp",
        "label_file":      "preprocessed/labels/grasp_tool.json",
        "batch_size":      256,
        "lr":              1e-4,
        "epochs":          50,
        "opt":             "adam",
        "weight_decay":    1e-4,
        "loss":            "bce",
        "primary":         "mAP",
        "balance_classes": False,
    },
    "cholect50_verb": {
        "feature_group":   "cholect50",
        "label_file":      "preprocessed/labels/cholect50_verb.json",
        "batch_size":      256,
        "lr":              1e-4,
        "epochs":          50,
        "opt":             "adam",
        "weight_decay":    1e-5,
        "loss":            "bce",
        "primary":         "mAP",
        "balance_classes": False,
    },
}


# ---------- Logging (reuse the Tee pattern) ----------

class _Tee:
    def __init__(self, *streams):
        self.streams = list(streams)
    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


def _slug(task, fold):
    return task if fold is None else f"{task}_fold{fold}"


def prepare_run_dir(out_dir, task, fold):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = os.path.join(out_dir, _slug(task, fold), "paper_linear", ts)
    os.makedirs(d, exist_ok=True)
    return d


def start_logging(run_dir):
    fh = open(os.path.join(run_dir, "log.txt"), "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh


def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=str)


def append_jsonl(path, obj):
    with open(path, "a") as f: f.write(json.dumps(obj, default=str) + "\n")


# ---------- Data flattening ----------

class FrameFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        if labels.ndim == 1:
            self.labels = torch.from_numpy(labels.astype(np.int64))
        else:
            self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def flatten_features_and_labels(features_root, group, label_data, video_ids, label_type):
    feats, labs, kept = [], [], []
    for vid in video_ids:
        vdata = label_data["videos"].get(vid)
        if vdata is None:
            continue
        pooled_path = os.path.join(features_root, group, "pooled", f"{vid}.npy")
        if not os.path.isfile(pooled_path):
            print(f"  [warn] missing features for {vid}: {pooled_path}")
            continue
        p = np.load(pooled_path).astype(np.float32)             # (T, 1536)
        if label_type == "multi-class":
            y = np.asarray(vdata["labels"], dtype=np.int64)      # (T,)
        else:
            y = np.asarray(vdata["labels"], dtype=np.float32)    # (T, C)
        if p.shape[0] != y.shape[0]:
            print(f"  [warn] T mismatch on {vid}: pooled={p.shape[0]} labels={y.shape[0]}")
            n = min(p.shape[0], y.shape[0])
            p = p[:n]; y = y[:n]
        feats.append(p); labs.append(y); kept.append(vid)
    feats = np.concatenate(feats, axis=0)
    labs = np.concatenate(labs, axis=0)
    return feats, labs, kept


def build_sampler(labels, label_type, balance):
    if not balance or label_type != "multi-class":
        return None
    counts = Counter(labels.tolist())
    class_weights = {c: 1.0 / max(1, counts[c]) for c in counts}
    sample_weights = np.array([class_weights[int(y)] for y in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels), replacement=True,
    )
    return sampler


# ---------- Training loop ----------

def build_optimizer(model, cfg):
    if cfg["opt"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def loss_fn_from_cfg(cfg):
    return nn.CrossEntropyLoss() if cfg["loss"] == "ce" else nn.BCEWithLogitsLoss()


@torch.no_grad()
def evaluate(model, features, labels, label_type, device, batch_size=1024):
    model.eval()
    T = features.shape[0]
    y_pred_chunks, y_true_chunks = [], []
    for i in range(0, T, batch_size):
        x = torch.from_numpy(features[i:i + batch_size]).float().to(device)
        logits = model(x)
        if label_type == "multi-class":
            y_pred_chunks.append(logits.argmax(dim=-1).cpu().numpy())
        else:
            y_pred_chunks.append(torch.sigmoid(logits).cpu().numpy())
        y_true_chunks.append(labels[i:i + batch_size])
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    y_true = np.concatenate(y_true_chunks, axis=0)
    if label_type == "multi-class":
        return phase_metrics(y_pred, y_true), y_pred, y_true
    return multilabel_metrics(y_pred, y_true), y_pred, y_true


def main(args):
    root = os.path.abspath(args.root)
    cfg = BASELINE_CONFIGS[args.task]
    run_dir = prepare_run_dir(os.path.join(root, args.out_dir), args.task, args.fold)
    start_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[time] start = {datetime.now().isoformat(timespec='seconds')}")
    print(f"[host] {os.uname().nodename}")
    print(f"[dev ] {device}  n_gpu={torch.cuda.device_count()}")
    print(f"[run ] {run_dir}")
    print(f"[task] {args.task}  fold={args.fold}")
    print(f"[cfg ] {cfg}")

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

    splits = load_splits(os.path.join(root, "preprocessed", "splits.json"))
    train_vids, val_vids, test_vids = get_split_partitions(splits, args.task, args.fold)
    print(f"[split] train={len(train_vids)}  val={len(val_vids)}  test={len(test_vids)}")

    features_root = os.path.join(root, "preprocessed", "features")
    group = cfg["feature_group"]

    print("[data ] flattening features")
    tr_feats, tr_labs, tr_kept = flatten_features_and_labels(
        features_root, group, label_data, train_vids, label_type)
    va_feats, va_labs, va_kept = flatten_features_and_labels(
        features_root, group, label_data, val_vids, label_type)
    te_feats, te_labs, te_kept = flatten_features_and_labels(
        features_root, group, label_data, test_vids, label_type)
    print(f"[data ] frames train={tr_feats.shape}  val={va_feats.shape}  test={te_feats.shape}")

    # Loaders
    train_ds = FrameFeatureDataset(tr_feats, tr_labs)
    sampler = build_sampler(tr_labs, label_type, cfg["balance_classes"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          sampler=sampler, shuffle=(sampler is None),
                          num_workers=4, pin_memory=True, drop_last=False)

    # Model
    model = nn.Linear(1536, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Linear(1536 -> {num_classes})  params = {n_params/1e3:.1f} K")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    loss_fn = loss_fn_from_cfg(cfg)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
    best_pred_path = os.path.join(run_dir, "predictions_test.npz")

    primary = cfg["primary"]
    print(f"[train] epochs={cfg['epochs']}  bs={cfg['batch_size']}  lr={cfg['lr']}  loss={cfg['loss']}")
    print("=" * 100)

    best_val, best_ep, best_test = -1.0, -1, None

    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_metrics, _, _ = evaluate(model, va_feats, va_labs, label_type, device)
        test_metrics, te_pred, te_true = evaluate(model, te_feats, te_labs, label_type, device)

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_score = val_metrics[primary]
        test_score = test_metrics[primary]
        elapsed = time.time() - t0

        print(f"[ep {ep:03d}]  loss={train_loss:.4f}  "
              f"val_{primary}={val_score*100:6.2f}  "
              f"test_{primary}={test_score*100:6.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)")

        append_jsonl(metrics_path, {
            "epoch": ep, "train_loss": train_loss,
            "val": val_metrics, "test": test_metrics,
            "lr": optimizer.param_groups[0]["lr"], "elapsed_s": elapsed,
        })

        if val_score > best_val:
            best_val = val_score; best_ep = ep; best_test = test_metrics
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_score": val_score,
                "test_metrics": test_metrics,
                "task": args.task,
                "fold": args.fold,
                "cfg": cfg,
            }, best_ckpt_path)
            np.savez_compressed(best_pred_path, y_pred=te_pred, y_true=te_true)

    print("=" * 100)
    print(f"[best]  epoch={best_ep}  val_{primary}={best_val*100:.2f}")
    if best_test:
        for k, v in best_test.items():
            print(f"        test.{k:20s} = {v*100:6.2f}")

    save_json(os.path.join(run_dir, "best.json"), {
        "best_epoch": best_ep,
        "best_val": best_val,
        "primary_metric": primary,
        "test_metrics": best_test,
        "task": args.task,
        "fold": args.fold,
        "num_classes": num_classes,
        "class_names": label_data["class_names"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    })

    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write(f"Task           : {args.task}\n")
        f.write(f"Fold           : {args.fold}\n")
        f.write(f"Mode           : paper_linear\n")
        f.write(f"Primary metric : {primary}\n")
        f.write(f"Best epoch     : {best_ep}\n")
        f.write(f"Best val       : {best_val*100:.2f}%\n")
        f.write("Test metrics @ best-val epoch:\n")
        if best_test:
            for k, v in best_test.items():
                f.write(f"  {k:20s} = {v*100:6.2f}%\n")
        f.write(f"\nRun directory  : {run_dir}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--task", required=True, choices=list(BASELINE_CONFIGS.keys()))
    p.add_argument("--fold", type=int, default=None, help="Fold 1..5 for cholect50_verb")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=30)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    main(args)
