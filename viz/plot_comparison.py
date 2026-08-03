#!/usr/bin/env python3
"""
Grouped bar chart: STAG (full) vs paper LemonFM linear-probing (Table 2) across
the four tasks.

Reads STAG numbers from viz_out/runs_summary.csv (mode="full" only).
Paper numbers are hard-coded from Table 2.

Outputs:
    viz_out/fig_vs_paper.pdf
    viz_out/fig_vs_paper.png
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_paper_style, PALETTE, TASK_PRETTY  # noqa: E402


# LemonFM linear-probing row from paper Table 2 (primary metric).
PAPER_LEMONFM = {
    "cholec80_phase": 75.8,   # Cholec80 accuracy
    "cholec80_tool":  76.8,   # mAP
    "grasp_tool":     76.4,   # mAP
    "cholect50_verb": 50.4,   # mAP
}

ORDER_TASKS = ["cholec80_phase", "cholec80_tool", "grasp_tool", "cholect50_verb"]


def load_ours(summary_csv):
    grouped = defaultdict(list)
    primary = {}
    with open(summary_csv, "r") as f:
        for r in csv.DictReader(f):
            if r.get("mode") != "full":
                continue
            task = r.get("task", "")
            pm = r.get("primary_metric", "")
            primary[task] = pm
            v = r.get(f"test.{pm}", "")
            try:
                grouped[task].append(float(v) * 100.0)
            except (TypeError, ValueError):
                pass
    return grouped, primary


def main(args):
    apply_paper_style()
    ours, _ = load_ours(args.summary)

    n = len(ORDER_TASKS)
    x = np.arange(n)
    width = 0.36

    paper_vals = [PAPER_LEMONFM[t] for t in ORDER_TASKS]
    ours_means, ours_stds = [], []
    for t in ORDER_TASKS:
        vs = ours.get(t, [])
        ours_means.append(np.mean(vs) if vs else 0.0)
        ours_stds.append(np.std(vs) if len(vs) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars_p = ax.bar(x - width / 2, paper_vals, width, color=PALETTE["paper"],
                    edgecolor="black", linewidth=0.4,
                    label="LemonFM linear (paper Table 2)")
    bars_o = ax.bar(x + width / 2, ours_means, width, yerr=ours_stds, capsize=3,
                    color=PALETTE["ours"], edgecolor="black", linewidth=0.4,
                    error_kw={"elinewidth": 0.8, "ecolor": "0.25"},
                    label="STAG (ours)")

    for b, v in zip(bars_p, paper_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="0.15")
    for b, v in zip(bars_o, ours_means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="0.15",
                fontweight="bold")

    # Δ arrows / labels
    for i, (p, o) in enumerate(zip(paper_vals, ours_means)):
        if o == 0:
            continue
        delta = o - p
        color = "#2a7d3f" if delta > 0 else "#a01a1a"
        ax.text(i, max(p, o) + 6.0, f"Δ = {delta:+.1f}",
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_PRETTY[t] for t in ORDER_TASKS])
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=10)
    ax.set_title("STAG vs LemonFM linear probing (paper Table 2)")

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(args.out_dir, f"fig_vs_paper.{ext}")
        fig.savefig(p)
        print(f"[ok] wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="viz_out/runs_summary.csv")
    ap.add_argument("--out-dir", default="viz_out")
    main(ap.parse_args())
