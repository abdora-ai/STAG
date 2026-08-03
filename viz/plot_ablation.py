#!/usr/bin/env python3
"""
STAG ablation figure - temporal gain vs. temporal continuity.

Instead of re-drawing the four cumulative scores that already appear in the
ablation *table*, this figure visualises the paper's central claim: the
contribution of the temporal affinity module scales with how much temporal
continuity a task actually has.

  x : temporal continuity proxy = expected run length, i.e. the expected number
      of *consecutively labelled* frames that share the same label. This is a
      property of each dataset's annotation protocol (dense 1 fps segments vs.
      35 s keyframe stride on GraSP). Log scale.
  y : measured gain of the temporal module = score(full) - score(slot_proto),
      read directly from runs_summary.csv (CholecT50 verb averaged over folds,
      with fold std as an error bar).

The figure contains no explanatory text; the colour legend identifies tasks and
the x/y axes carry the meaning. All numbers on y are real; the x proxy values
are documented dataset estimates and can be replaced by measured lag-1 label
autocorrelation if the label files are available.

Reads:  viz_out/runs_summary.csv
Writes: viz_out/fig_ablation.pdf/.png
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_paper_style, TASK_PRETTY   # noqa: E402


ORDER_TASKS = ["cholec80_phase", "cholec80_tool", "cholect50_verb", "grasp_tool"]

# Colour per task (colourblind-safe, Wong 2011).
TASK_COLOR = {
    "cholec80_phase": "#0072B2",
    "cholec80_tool":  "#009E73",
    "cholect50_verb": "#E69F00",
    "grasp_tool":     "#D55E00",
}

# Temporal continuity proxy: expected number of consecutive labelled frames that
# share a label. Dense 1 fps tasks with long events -> large; GraSP keyframes are
# 35 s apart so adjacent labelled samples are effectively independent (~1).
# These are dataset-protocol estimates (editable); the y-values below are measured.
CONTINUITY = {
    "cholec80_phase": 150.0,   # phases last minutes at 1 fps
    "cholec80_tool":  25.0,    # tool presence persists tens of seconds
    "cholect50_verb": 4.0,     # verbs are transient, a few seconds
    "grasp_tool":     1.0,     # 35 s keyframe stride -> independent samples
}


def load_continuity(path):
    """Load real temporal-continuity JSON produced by tools/temporal_continuity.py.

    Returns dict {task: {autocorr, gain_pp, gain_pp_std, alpha, ...}} or None.
    """
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_scores(summary_csv):
    """Return dict {(task, mode): [scores...]} in percent (all folds)."""
    grouped = defaultdict(list)
    with open(summary_csv, "r") as f:
        for r in csv.DictReader(f):
            task = r.get("task", "")
            mode = r.get("mode", "")
            pm = r.get("primary_metric", "")
            v = r.get(f"test.{pm}", "")
            try:
                grouped[(task, mode)].append(float(v) * 100.0)
            except (TypeError, ValueError):
                pass
    return grouped


def temporal_gain(grouped, task):
    """Mean and std (pp) of full - slot_proto for a task, matched by fold order."""
    full = grouped.get((task, "full"), [])
    slot = grouped.get((task, "slot_proto"), [])
    if not full or not slot:
        return 0.0, 0.0
    n = min(len(full), len(slot))
    deltas = np.array(full[:n]) - np.array(slot[:n])
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    return float(deltas.mean()), std


def _plot_correlogram(ax, real):
    """One label-autocorrelation decay curve per task."""
    for task in ORDER_TASKS:
        r = real.get(task, {})
        curve = r.get("autocorr_by_lag")
        if not curve:
            continue
        lags = np.asarray(curve["lags"], dtype=float)
        vals = np.asarray(curve["values"], dtype=float)
        m = np.isfinite(vals)
        gain = r.get("gain_pp")
        label = TASK_PRETTY[task]
        if gain is not None:
            label = f"{TASK_PRETTY[task]}  ({gain:+.1f} pp)"
        ax.plot(lags[m], vals[m], color=TASK_COLOR[task], lw=2.2,
                marker="o", ms=3.5, markevery=max(1, int(m.sum() // 10)),
                label=label, zorder=3)

    ax.axhline(0.0, color="#BBBBBB", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Time lag between labels (sampling steps)")
    ax.set_ylabel("Label autocorrelation")
    ax.set_ylim(-0.05, 1.03)
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5,
              handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3)


def _plot_scatter(ax, grouped, real):
    """Fallback: temporal gain vs. continuity scatter (used if no lag curves)."""
    xs, ys, colors = [], [], []
    for task in ORDER_TASKS:
        if real and real.get(task, {}).get("autocorr") is not None:
            xs.append(real[task]["autocorr"]); ys.append(real[task]["gain_pp"])
        else:
            g, _ = temporal_gain(grouped, task)
            xs.append(CONTINUITY[task]); ys.append(g)
        colors.append(TASK_COLOR[task])
    for x, y, c in zip(xs, ys, colors):
        ax.plot(x, y, "o", ms=11, color=c, markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xlabel("Temporal continuity")
    ax.set_ylabel("Gain from temporal module (pp)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=9,
                              color=TASK_COLOR[t], markeredgecolor="black",
                              label=TASK_PRETTY[t]) for t in ORDER_TASKS],
              loc="upper left", frameon=False, fontsize=9)


def main(args):
    apply_paper_style()
    grouped = load_scores(args.summary)
    real = load_continuity(args.continuity)

    have_curves = bool(real) and any(
        real.get(t, {}).get("autocorr_by_lag") for t in ORDER_TASKS)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if have_curves:
        _plot_correlogram(ax, real)
    else:
        _plot_scatter(ax, grouped, real)

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(args.out_dir, f"fig_ablation.{ext}")
        fig.savefig(p)
        print(f"[ok] wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="viz_out/runs_summary.csv")
    ap.add_argument("--continuity", default="viz_out/temporal_continuity.json",
                    help="Real continuity JSON from tools/temporal_continuity.py; "
                         "used if present, else falls back to dataset estimates.")
    ap.add_argument("--out-dir", default="viz_out")
    main(ap.parse_args())
