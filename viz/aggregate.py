#!/usr/bin/env python3
"""
Walk `runs/` and aggregate every best.json into a single tidy CSV + Markdown
table. Handles both the mode-nested layout `runs/<task>/<mode>/<ts>/` and the
legacy pre-ablation layout `runs/<task>/<ts>/` (which is treated as mode="full").

Also produces:
    runs_summary.csv         one row per (task, fold, mode) with the best-val
                             epoch's test metrics.
    runs_summary.md          the same table rendered as a markdown table.
    runs_summary_pivot.csv   pivot with rows = task, columns = mode, cells =
                             primary metric (mean over folds where applicable).

Usage:
    python viz/aggregate.py --runs runs --out-dir viz_out
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
KNOWN_MODES = ("paper_linear", "linear", "prototype", "slot_proto", "full")


def _iter_run_dirs(runs_root):
    """
    Yield (task_slug, mode, run_dir) for every runnable directory:
      - runs/<task_slug>/<mode>/<ts>/best.json
      - runs/<task_slug>/<ts>/best.json   (legacy = mode "full")
    """
    for task_slug in sorted(os.listdir(runs_root)):
        task_dir = os.path.join(runs_root, task_slug)
        if not os.path.isdir(task_dir):
            continue
        for entry in sorted(os.listdir(task_dir)):
            sub = os.path.join(task_dir, entry)
            if not os.path.isdir(sub):
                continue
            if entry in KNOWN_MODES:
                # mode dir; iterate its timestamped children
                for ts in sorted(os.listdir(sub)):
                    ts_dir = os.path.join(sub, ts)
                    if os.path.isdir(ts_dir) and os.path.isfile(os.path.join(ts_dir, "best.json")):
                        yield task_slug, entry, ts_dir
            elif TS_RE.match(entry):
                # legacy layout: task/ts → mode "full"
                if os.path.isfile(os.path.join(sub, "best.json")):
                    yield task_slug, "full", sub


def _load_best(run_dir):
    with open(os.path.join(run_dir, "best.json"), "r") as f:
        return json.load(f)


def _split_fold(task_slug):
    """cholect50_verb_fold3 -> ('cholect50_verb', 3);  cholec80_phase -> ('cholec80_phase', None)."""
    m = re.match(r"^(.*)_fold(\d+)$", task_slug)
    if m:
        return m.group(1), int(m.group(2))
    return task_slug, None


def collect(runs_root):
    """Latest run per (task_slug, mode); returns a list of dicts."""
    latest = {}
    for task_slug, mode, run_dir in _iter_run_dirs(runs_root):
        key = (task_slug, mode)
        # Prefer the most recent timestamp (directory names are ISO-like so lexicographic works)
        if key not in latest or run_dir > latest[key]:
            latest[key] = run_dir

    rows = []
    for (task_slug, mode), run_dir in sorted(latest.items()):
        try:
            best = _load_best(run_dir)
        except Exception as e:
            print(f"[warn] cannot read {run_dir}/best.json: {e}")
            continue
        base_task, fold = _split_fold(task_slug)
        row = {
            "task": base_task,
            "task_slug": task_slug,
            "fold": fold,
            "mode": mode,
            "run_dir": os.path.relpath(run_dir),
            "primary_metric": best.get("primary_metric"),
            "best_epoch": best.get("best_epoch"),
            "best_val": best.get("best_val"),
        }
        for k, v in (best.get("test_metrics") or {}).items():
            row[f"test.{k}"] = v
        rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        print(f"[warn] no rows; not writing {path}")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    # Put the important columns first
    preferred = ["task", "task_slug", "fold", "mode", "primary_metric",
                 "best_epoch", "best_val", "test.mAP", "test.accuracy",
                 "test.macro_f1", "test.macro_precision", "test.macro_recall",
                 "run_dir"]
    ordered = [c for c in preferred if c in fieldnames] + [
        c for c in fieldnames if c not in preferred
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ordered})
    print(f"[ok] wrote {path}  ({len(rows)} rows)")


def write_markdown_table(rows, path):
    if not rows:
        return
    hdrs = ["task", "fold", "mode", "best_epoch", "best_val",
            "test.mAP", "test.accuracy", "test.macro_f1"]
    lines = ["| " + " | ".join(hdrs) + " |",
             "| " + " | ".join(["---"] * len(hdrs)) + " |"]
    for r in rows:
        cells = []
        for h in hdrs:
            v = r.get(h)
            if v is None or v == "":
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v*100:.2f}" if 0.0 <= v <= 1.0 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {path}")


def write_pivot(rows, path):
    """Pivot: rows = task (with mean over folds for cross-val), cols = mode."""
    if not rows:
        return
    # gather primary metric per (task, mode, fold)
    grouped = defaultdict(lambda: defaultdict(list))  # {task: {mode: [values]}}
    primary_per_task = {}
    for r in rows:
        task = r["task"]
        mode = r["mode"]
        pm = r.get("primary_metric") or ""
        primary_per_task[task] = pm
        val = r.get(f"test.{pm}")
        if val is None:
            continue
        grouped[task][mode].append(float(val))

    modes_order = [m for m in ("paper_linear", "linear", "prototype", "slot_proto", "full")]
    tasks_order = sorted(grouped.keys())

    lines = []
    hdr = ["task", "metric"] + modes_order
    lines.append(",".join(hdr))
    for task in tasks_order:
        row = [task, primary_per_task.get(task, "")]
        for m in modes_order:
            vs = grouped[task].get(m, [])
            if not vs:
                row.append("")
            elif len(vs) == 1:
                row.append(f"{vs[0]*100:.2f}")
            else:
                import statistics as st
                mean = st.mean(vs) * 100
                std = st.pstdev(vs) * 100
                row.append(f"{mean:.2f}±{std:.2f}")
        lines.append(",".join(row))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", default="viz_out")
    args = ap.parse_args()

    if not os.path.isdir(args.runs):
        raise SystemExit(f"[error] runs directory not found: {args.runs}")

    rows = collect(args.runs)
    if not rows:
        raise SystemExit("[error] no best.json files found under runs/")

    write_csv(rows, os.path.join(args.out_dir, "runs_summary.csv"))
    write_markdown_table(rows, os.path.join(args.out_dir, "runs_summary.md"))
    write_pivot(rows, os.path.join(args.out_dir, "runs_summary_pivot.csv"))

    # Also print the pivot to stdout for quick eyeballing
    print()
    with open(os.path.join(args.out_dir, "runs_summary_pivot.csv"), "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
