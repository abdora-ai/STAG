#!/usr/bin/env bash
# =============================================================================
# run_temporal_baselines.sh
#
# Trains MS-TCN / TeCNO temporal baselines on frozen LemonFM features to
# directly address the supervisor's comparison request.
#
# Each task is trained in BOTH causal and acausal mode:
#   • Causal   = standard MS-TCN/TeCNO (can only see past frames).
#   • Acausal  = bidirectional, same as STAG's symmetric affinity window.
#                This is the fair apples-to-apples comparison with STAG.
#
# Run directory layout:
#   runs/baselines/
#   ├── tecno/                       ← phase (4-stage config = TeCNO)
#   │   ├── cholec80_phase/
#   │   │   ├── causal/<ts>/         → config.json, best.json, log.txt …
#   │   │   └── acausal/<ts>/
#   ├── mstcn/                       ← tool + verb (2-stage config = MS-TCN)
#   │   ├── cholec80_tool/
#   │   │   ├── causal/<ts>/
#   │   │   └── acausal/<ts>/
#   │   ├── grasp_tool/ …
#   │   └── cholect50_verb_fold1..5/ …
#
# DATASET PLACEMENT (before running):
#   ROOT/
#   ├── lemonfm.pth
#   ├── Cholec80/
#   │   ├── videos/            video01.mp4 … video80.mp4
#   │   ├── phase_annotations/ video01-phase.txt …
#   │   └── tool_annotations/  video01-tool.txt …
#   ├── CholecT50/
#   │   ├── videos/            VID01/ … (*.png frames)
#   │   └── labels/            VID01.json …
#   └── GraSP/
#       └── GraSP_1fps/
#           ├── annotations/   grasp_short-term_train.json
#           │                  grasp_short-term_test.json
#           └── frames/        CASE001/ …
#
# WHERE TO GET THE DATA:
#   Cholec80  : https://docs.google.com/forms/d/1GwZFM1S0E_sCVEGC0p6OMBSbEGMBMPkV0SWIMQVK58
#   CholecT50 : https://forms.gle/9qMnYdN5RDWUMRXG6
#   GraSP     : https://github.com/BCV-Uniandes/GraSP
#   LemonFM   : contact che.chengan@kcl.ac.uk  (arXiv:2503.19740)
#
# USAGE:
#   bash scripts/run_temporal_baselines.sh [ROOT] [BACKBONE] [OPTIONS]
#
#   ROOT      repo root containing dataset dirs   (default: parent of scripts/)
#   BACKBONE  path to lemonfm.pth                 (default: ROOT/lemonfm.pth)
#
#   --skip-cache    skip feature caching (if already cached on this machine)
#   --skip-labels   skip label building  (if preprocessed/ already exists)
#   --epochs N      training epochs, default 50
#   --dry-run       print all commands without executing them
#
# EXAMPLE:
#   bash scripts/run_temporal_baselines.sh . lemonfm.pth
#   bash scripts/run_temporal_baselines.sh . lemonfm.pth --skip-cache --skip-labels
# =============================================================================

set -euo pipefail

# ── argument parsing ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
BACKBONE="${2:-$ROOT/lemonfm.pth}"

SKIP_CACHE=0; SKIP_LABELS=0; EPOCHS=50; DRY_RUN=0
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-cache)  SKIP_CACHE=1 ;;
        --skip-labels) SKIP_LABELS=1 ;;
        --epochs)      EPOCHS="$2"; shift ;;
        --dry-run)     DRY_RUN=1 ;;
        *) echo "[warn] unknown flag: $1" ;;
    esac
    shift
done

TMUX_SESSION="lemon-baselines"

echo "========================================================================"
echo "  MS-TCN / TeCNO temporal baselines  (causal + acausal)"
echo "  ROOT     : $ROOT"
echo "  BACKBONE : $BACKBONE"
echo "  EPOCHS   : $EPOCHS"
echo "  DRY_RUN  : $DRY_RUN"
echo "========================================================================"

# ── helpers ───────────────────────────────────────────────────────────────────
_run() {
    if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY] $*"; else echo "[RUN] $*"; "$@"; fi
}

# Launch a command in a named tmux window and return immediately.
_tmux() {
    local name="$1"; shift
    local cmd="$*"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-TMUX] window='$name'  cmd=$cmd"
        return
    fi
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 50 -n "$name"
        tmux send-keys -t "${TMUX_SESSION}:${name}" "$cmd" Enter
    else
        tmux new-window -t "$TMUX_SESSION" -n "$name"
        tmux send-keys -t "${TMUX_SESSION}:${name}" "$cmd" Enter
    fi
    echo "  [tmux] started window '$name'"
}

# ── step 1: check required inputs ────────────────────────────────────────────
echo ""
echo "[1/6] Checking required inputs..."
MISSING=0

_need_dir() {
    if [ ! -d "$1" ]; then
        printf "  ✗ MISSING: %s\n    Expected at: %s\n    Download:    %s\n\n" "$2" "$1" "$3"
        MISSING=1
    else
        echo "  ✓ $2"
    fi
}

_need_dir "$ROOT/Cholec80"          "Cholec80"    "https://docs.google.com/forms/d/1GwZFM1S0E_sCVEGC0p6OMBSbEGMBMPkV0SWIMQVK58"
_need_dir "$ROOT/CholecT50"         "CholecT50"   "https://forms.gle/9qMnYdN5RDWUMRXG6"
_need_dir "$ROOT/GraSP/GraSP_1fps"  "GraSP 1fps"  "https://github.com/BCV-Uniandes/GraSP"

if [ ! -f "$BACKBONE" ]; then
    printf "  ✗ MISSING: LemonFM backbone\n    Expected at: %s\n    Contact: che.chengan@kcl.ac.uk (arXiv:2503.19740)\n\n" "$BACKBONE"
    MISSING=1
else
    echo "  ✓ LemonFM backbone"
fi

if [[ $MISSING -eq 1 ]]; then
    echo "[ERROR] Missing inputs - see above. Place them then re-run."; exit 1
fi

# ── step 2: build Python environment ─────────────────────────────────────────
echo ""
echo "[2/6] Building Python environment..."
_run bash "$ROOT/env/setup_env.sh"

VENV="$ROOT/.venv-lemon"
if [[ $DRY_RUN -eq 0 ]] && [ ! -f "$VENV/bin/python" ]; then
    echo "[ERROR] venv not found at $VENV after setup."; exit 1
fi
PYTHON="$VENV/bin/python"
ACTIVATE="source $VENV/bin/activate"
echo "  ✓ Python: $PYTHON"

# ── step 3: extract GraSP tarballs ───────────────────────────────────────────
echo ""
echo "[3/6] Extracting GraSP tarballs..."
_run bash "$ROOT/tools/extract_grasp.sh" "$ROOT/GraSP"

# ── step 4: build labels ──────────────────────────────────────────────────────
if [[ $SKIP_LABELS -eq 0 ]]; then
    echo ""
    echo "[4/6] Building label JSONs and split file..."
    _run "$PYTHON" "$ROOT/scripts/build_labels.py" --root "$ROOT"
else
    echo "[4/6] Skipping label build (--skip-labels)."
fi

LABELS_OK=1
for task in cholec80_phase cholec80_tool cholect50_verb grasp_tool; do
    lf="$ROOT/preprocessed/labels/${task}.json"
    if [[ $DRY_RUN -eq 0 ]] && [ ! -f "$lf" ]; then
        echo "  [ERROR] Missing: $lf"; LABELS_OK=0
    fi
done
[[ $LABELS_OK -eq 0 ]] && { echo "[ERROR] Re-run without --skip-labels."; exit 1; }
echo "  ✓ All four label files present."

# ── step 5: feature caching ───────────────────────────────────────────────────
# Runs the frozen LemonFM backbone over every raw frame ONCE and saves
# pooled [T×1536] and spatial [T×49×1536] .npy files to disk.
# This is the only step that touches raw frames or the backbone.
# All training (MS-TCN, STAG, etc.) reads only the cached .npy files.
# The three groups are independent so we launch them in parallel tmux windows.

if [[ $SKIP_CACHE -eq 0 ]]; then
    echo ""
    echo "[5/6] Launching feature caching (3 parallel tmux windows)..."
    for group in cholec80 cholect50 grasp; do
        _tmux "cache-${group}" \
            "$ACTIVATE && cd $ROOT && \
            python scripts/cache_features.py \
                --root . --checkpoint $BACKBONE --only-group $group && \
            echo '[DONE] cache-${group}'"
    done

    if [[ $DRY_RUN -eq 0 ]]; then
        echo ""
        echo "  ⚠  Caching runs in background. Attach to watch:"
        echo "       tmux attach -t $TMUX_SESSION"
        echo ""
        echo "  Waiting for all three cache jobs to complete..."
        for group in cholec80 cholect50 grasp; do
            dir="$ROOT/preprocessed/features/${group}/pooled"
            until [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; do
                sleep 30
            done
            echo "  ✓ $group features ready"
        done
    fi
else
    echo "[5/6] Skipping feature caching (--skip-cache)."
fi

# ── step 6: training (causal + acausal, each in its own tmux window) ──────────
# Window naming: <head>-<task-short>-<causality>
# e.g.  tecno-phase-causal   mstcn-cholec-tool-acausal   mstcn-verb-f3-acausal
#
# Run directory: runs/baselines/<head>/<task>/<causal|acausal>/<timestamp>/

echo ""
echo "[6/6] Launching training jobs (causal + acausal for each task)..."
echo ""

BASE_CMD="$ACTIVATE && cd $ROOT && python scripts/train_mstcn.py --root . --epochs $EPOCHS --seed 42"

# ── Cholec80 phase  (TeCNO config)
for causality in causal acausal; do
    flag=""; [[ $causality == causal ]] && flag="--causal"
    _tmux "tecno-phase-${causality}" \
        "${BASE_CMD} --task cholec80_phase ${flag} && echo '[DONE] tecno-phase-${causality}'"
done

# ── Cholec80 tool  (MS-TCN config)
for causality in causal acausal; do
    flag=""; [[ $causality == causal ]] && flag="--causal"
    _tmux "mstcn-cholec-tool-${causality}" \
        "${BASE_CMD} --task cholec80_tool ${flag} && echo '[DONE] mstcn-cholec-tool-${causality}'"
done

# ── GraSP tool  (MS-TCN config)
for causality in causal acausal; do
    flag=""; [[ $causality == causal ]] && flag="--causal"
    _tmux "mstcn-grasp-tool-${causality}" \
        "${BASE_CMD} --task grasp_tool ${flag} && echo '[DONE] mstcn-grasp-tool-${causality}'"
done

# ── CholecT50 verb  (MS-TCN config, 5 folds × 2 causality = 10 windows)
for fold in 1 2 3 4 5; do
    for causality in causal acausal; do
        flag=""; [[ $causality == causal ]] && flag="--causal"
        _tmux "mstcn-verb-f${fold}-${causality}" \
            "${BASE_CMD} --task cholect50_verb --fold ${fold} ${flag} && \
            echo '[DONE] mstcn-verb-f${fold}-${causality}'"
    done
done

echo ""
echo "========================================================================"
echo "  All jobs launched in tmux session: $TMUX_SESSION"
echo ""
echo "  Attach:       tmux attach -t $TMUX_SESSION"
echo "  List windows: tmux list-windows -t $TMUX_SESSION"
echo ""
echo "  Window map:"
echo "    tecno-phase-{causal,acausal}          → Cholec80 phase  (TeCNO)"
echo "    mstcn-cholec-tool-{causal,acausal}    → Cholec80 tool   (MS-TCN)"
echo "    mstcn-grasp-tool-{causal,acausal}     → GraSP tool      (MS-TCN)"
echo "    mstcn-verb-f{1..5}-{causal,acausal}   → CholecT50 verb  (MS-TCN, 5 folds × 2)"
echo ""
echo "  Results land in:  runs/baselines/<head>/<task>/<causal|acausal>/<ts>/"
echo "    best.json      → primary metric (Acc/F1 for phase, mAP for tool/verb)"
echo "    run_summary.txt → human-readable summary"
echo "========================================================================"
