#!/usr/bin/env bash
# One-shot: build the paper-ready aggregate table + all quantitative plots +
# real-image slot-attention overlays, using whatever is in runs/.
#
# Assumes the venv (.venv-lemon) is already active.
#
# Usage:
#   bash viz/run_all.sh
#   OUT_DIR=paper_figs bash viz/run_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-viz_out}"
mkdir -p "$OUT_DIR"

echo "[1/5] aggregating runs/ -> $OUT_DIR/runs_summary.*"
python viz/aggregate.py --runs runs --out-dir "$OUT_DIR"

echo
echo "[2/5] ablation bar chart"
python viz/plot_ablation.py --summary "$OUT_DIR/runs_summary.csv" --out-dir "$OUT_DIR"

echo
echo "[3/5] STAG-vs-paper bar chart"
python viz/plot_comparison.py --summary "$OUT_DIR/runs_summary.csv" --out-dir "$OUT_DIR"

echo
echo "[4/5] training curves per task"
python viz/plot_curves.py --runs runs --out-dir "$OUT_DIR/curves"

echo
echo "[5/5] confusion matrix + per-class metrics"
python viz/plot_confusion.py --runs runs --out-dir "$OUT_DIR/confusion"

echo
echo "[extra] unified slot attention overlay (needs lemonfm.pth in repo root)"
if [ -f lemonfm.pth ]; then
    python viz/plot_slot_attention.py \
        --root . --checkpoint lemonfm.pth --runs runs \
        --out-dir "$OUT_DIR/slot_attention" || \
        echo "  [warn] unified slot attention failed"
else
    echo "  [skip] lemonfm.pth not found in repo root"
fi

echo
echo "==== Outputs ===="
find "$OUT_DIR" -type f | sort
