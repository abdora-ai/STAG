#!/usr/bin/env python3
"""
Per-run training curves from metrics.jsonl.

Produces one 2x2 grid per task: {train loss, val primary, test primary, α/σ}.

Usage:
    python viz/plot_curves.py --runs runs --out-dir viz_out/curves
    python viz/plot_curves.py --runs runs --out-dir viz_out/curves --tasks cholec80_phase
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_paper_style, PALETTE, MODE_PRETTY, TASK_PRETTY  # noqa: E402


ORDER_MODES = ["linear", "prototype", "slot_proto", "full"]


def _iter_run_dirs(runs_root):
    for task_slug in sorted(os.listdir(runs_root)):
        td = os.path.join(runs_root, task_slug)
        if not os.path.isdir(td):
            continue
        for entry in sorted(os.listdir(td)):
            sub = os.path.join(td, entry)
            if not os.path.isdir(sub):
                continue
            if entry in ORDER_MODES:
                for ts in sorted(os.listdir(sub)):
                    ts_dir = os.path.join(sub, ts)
                    if os.path.isfile(os.path.join(ts_dir, "metrics.jsonl")):
                        yield task_slug, entry, ts_dir
            else:
                # legacy ts at task level → treat as "full"
                if os.path.isfile(os.path.join(sub, "metrics.jsonl")):
                    yield task_slug, "full", sub


def load_metrics(run_dir):
    rows = []
    with open(os.path.join(run_dir, "metrics.jsonl"), "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _split_fold(task_slug):
    import re
    m = re.match(r"^(.*)_fold(\d+)$", task_slug)
    return (m.group(1), int(m.group(2))) if m else (task_slug, None)


def plot_task(task, all_runs, out_dir):
    """all_runs: list of dicts {mode, fold, rows}."""
    apply_paper_style()

    primary_key = None
    for run in all_runs:
        for r in run["rows"]:
            for k in ("mAP", "accuracy"):
                if r.get("test") and k in r["test"]:
                    primary_key = k
                    break
            if primary_key:
                break
        if primary_key:
            break
    if primary_key is None:
        primary_key = "mAP"

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    ax_loss, ax_val, ax_test, ax_diff = axes.flat

    # Bucket by mode; average over folds for CholecT50 verb
    by_mode = defaultdict(list)
    for run in all_runs:
        by_mode[run["mode"]].append(run["rows"])

    for mode in ORDER_MODES:
        runs = by_mode.get(mode, [])
        if not runs:
            continue
        colors = PALETTE[mode]
        for label_run, rows in enumerate(runs):
            eps = np.array([r["epoch"] for r in rows])
            losses = np.array([r.get("train_loss", np.nan) for r in rows])
            val = np.array([r.get("val", {}).get(primary_key, np.nan) if r.get("val") else np.nan
                            for r in rows]) * 100
            tst = np.array([r.get("test", {}).get(primary_key, np.nan) if r.get("test") else np.nan
                            for r in rows]) * 100
            alpha = np.array([r.get("alpha", np.nan) for r in rows])
            sigma = np.array([r.get("sigma", np.nan) for r in rows])
            lbl = MODE_PRETTY[mode] if label_run == 0 else None
            ax_loss.plot(eps, losses, color=colors, alpha=0.8, linewidth=1.5, label=lbl)
            ax_val.plot(eps, val,    color=colors, alpha=0.8, linewidth=1.5, label=lbl)
            ax_test.plot(eps, tst,   color=colors, alpha=0.8, linewidth=1.5, label=lbl)
            if mode == "full":
                ax_diff.plot(eps, alpha, color="#009E73", linewidth=1.5, label="α" if label_run == 0 else None)
                ax_diff.plot(eps, sigma, color="#0072B2", linewidth=1.5, linestyle="--",
                             label="σ" if label_run == 0 else None)

    for ax, title, ylabel in [
        (ax_loss, "Train loss",           "loss"),
        (ax_val,  f"Val {primary_key}",   f"{primary_key} (%)"),
        (ax_test, f"Test {primary_key}",  f"{primary_key} (%)"),
        (ax_diff, "Diffusion params (full)", "value"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Training curves - {TASK_PRETTY.get(task, task)}", y=1.02)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"curves_{task}.{ext}")
        fig.savefig(p)
        print(f"[ok] wrote {p}")
    plt.close(fig)


def main(args):
    task_to_runs = defaultdict(list)
    for task_slug, mode, run_dir in _iter_run_dirs(args.runs):
        base_task, fold = _split_fold(task_slug)
        if args.tasks and base_task not in args.tasks:
            continue
        rows = load_metrics(run_dir)
        if not rows:
            continue
        task_to_runs[base_task].append({"mode": mode, "fold": fold, "rows": rows,
                                        "run_dir": run_dir})

    for task, runs in task_to_runs.items():
        plot_task(task, runs, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", default="viz_out/curves")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Base task names to include (default: all)")
    main(ap.parse_args())
