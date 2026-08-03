#!/usr/bin/env python3
"""
Confusion matrix for Cholec80 phase (multi-class) from predictions_test.npz.

For the tool/verb multi-label tasks we instead produce per-class Precision /
Recall / F1 bar charts.

Usage:
    python viz/plot_confusion.py --runs runs --out-dir viz_out/confusion
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_paper_style, TASK_PRETTY  # noqa: E402


KNOWN_MODES = ("linear", "prototype", "slot_proto", "full")


def _iter_run_dirs(runs_root):
    for task_slug in sorted(os.listdir(runs_root)):
        td = os.path.join(runs_root, task_slug)
        if not os.path.isdir(td):
            continue
        for entry in sorted(os.listdir(td)):
            sub = os.path.join(td, entry)
            if not os.path.isdir(sub):
                continue
            if entry in KNOWN_MODES:
                for ts in sorted(os.listdir(sub)):
                    ts_dir = os.path.join(sub, ts)
                    if os.path.isfile(os.path.join(ts_dir, "best.json")) and \
                       os.path.isfile(os.path.join(ts_dir, "predictions_test.npz")):
                        yield task_slug, entry, ts_dir
            else:
                if os.path.isfile(os.path.join(sub, "best.json")) and \
                   os.path.isfile(os.path.join(sub, "predictions_test.npz")):
                    yield task_slug, "full", sub


def _load_pred(run_dir):
    npz = np.load(os.path.join(run_dir, "predictions_test.npz"), allow_pickle=True)
    best = json.load(open(os.path.join(run_dir, "best.json"), "r"))
    class_names = best.get("class_names") or []
    return npz["y_pred"], npz["y_true"], class_names, best


def plot_confusion(y_pred, y_true, class_names, title, out_stub):
    apply_paper_style()
    C = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("Ground-truth phase")
    ax.set_title(title)

    thr = 0.5
    for i in range(C):
        for j in range(C):
            val = cm_norm[i, j]
            if val <= 0.005:
                continue
            ax.text(j, i, f"{val*100:.0f}",
                    ha="center", va="center",
                    fontsize=8, color="white" if val > thr else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction")
    for ext in ("pdf", "png"):
        p = f"{out_stub}.{ext}"
        fig.savefig(p)
        print(f"[ok] wrote {p}")
    plt.close(fig)


def plot_per_class_prf(y_prob, y_true, class_names, title, out_stub, thr=0.5):
    apply_paper_style()
    y_pred = (y_prob >= thr).astype(np.int32)
    C = len(class_names)

    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    ap = np.array([average_precision_score(y_true[:, c], y_prob[:, c]) if y_true[:, c].sum() else np.nan
                   for c in range(C)])

    x = np.arange(C)
    w = 0.2
    fig, ax = plt.subplots(figsize=(1.3 * C + 4.5, 4.6))
    ax.bar(x - 1.5*w, prec, w, color="#0072B2", label="Precision", edgecolor="black", linewidth=0.3)
    ax.bar(x - 0.5*w, rec,  w, color="#E69F00", label="Recall",    edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.5*w, f1,   w, color="#009E73", label="F1",        edgecolor="black", linewidth=0.3)
    ax.bar(x + 1.5*w, ap,   w, color="#CC79A7", label="AP",        edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--")
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
    for ext in ("pdf", "png"):
        p = f"{out_stub}.{ext}"
        fig.savefig(p)
        print(f"[ok] wrote {p}")
    plt.close(fig)


def _slug_to_base(task_slug):
    import re
    m = re.match(r"^(.*)_fold(\d+)$", task_slug)
    return m.group(1) if m else task_slug


def main(args):
    # Prefer the "full" (best) mode; take one run per (base_task).
    # For cross-val tasks with multiple folds, concatenate y_pred/y_true across folds.
    grouped = defaultdict(list)  # base_task -> list of (task_slug, mode, run_dir)
    for task_slug, mode, run_dir in _iter_run_dirs(args.runs):
        grouped[_slug_to_base(task_slug)].append((task_slug, mode, run_dir))

    for base_task, entries in grouped.items():
        # Prefer full mode
        entries_full = [(t, m, d) for (t, m, d) in entries if m == "full"]
        if not entries_full:
            continue

        # Take latest per task_slug
        latest = {}
        for t, m, d in entries_full:
            if t not in latest or d > latest[t][2]:
                latest[t] = (t, m, d)
        entries_full = list(latest.values())

        y_pred_all, y_true_all, class_names = [], [], None
        for t, m, d in entries_full:
            y_pred, y_true, cn, best = _load_pred(d)
            if class_names is None:
                class_names = cn
            y_pred_all.append(y_pred)
            y_true_all.append(y_true)

        if not y_pred_all:
            continue
        y_pred = np.concatenate(y_pred_all, axis=0)
        y_true = np.concatenate(y_true_all, axis=0)

        os.makedirs(args.out_dir, exist_ok=True)

        if base_task == "cholec80_phase":
            title = f"{TASK_PRETTY[base_task]} - STAG (full)"
            plot_confusion(y_pred, y_true, class_names, title,
                           os.path.join(args.out_dir, f"confusion_{base_task}"))
        else:
            # multi-label: y_pred is probabilities
            title = f"{TASK_PRETTY[base_task]} - per-class metrics (STAG full)"
            plot_per_class_prf(y_pred, y_true, class_names, title,
                               os.path.join(args.out_dir, f"per_class_{base_task}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", default="viz_out/confusion")
    main(ap.parse_args())
