#!/usr/bin/env bash
# Reproduce the paper's LemonFM linear-probing baseline with the paper's
# training regime (shuffled mini-batches of frames), using cached features.
#
# GPU 0 : stag_pl_cholec80_phase
# GPU 1 : stag_pl_cholec80_tool
# GPU 2 : stag_pl_grasp_tool
# GPU 3 : stag_pl_cholect50_verb   (all 5 folds sequentially)
#
# Very fast: each run is a linear layer on cached features, batches shuffled.
# All four sessions should finish within ~15 min total.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv-lemon"
BOOT="export PATH=\"\$HOME/.local/bin:\$PATH\" && source $VENV_DIR/bin/activate"

launch() {
    local session="$1"; local gpu="$2"; local args="$3"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[skip] $session already exists"; return
    fi
    local cmd="$BOOT && CUDA_VISIBLE_DEVICES=$gpu python scripts/reproduce_baseline.py --root . $args; echo; echo '===== DONE ($session) ====='; exec bash"
    tmux new-session -d -s "$session" "bash -c \"$cmd\""
    echo "[start] $session  (GPU $gpu, args: $args)"
}

launch "stag_pl_cholec80_phase" 0 "--task cholec80_phase"
launch "stag_pl_cholec80_tool"  1 "--task cholec80_tool"
launch "stag_pl_grasp_tool"     2 "--task grasp_tool"

verb_session="stag_pl_cholect50_verb"
if ! tmux has-session -t "$verb_session" 2>/dev/null; then
    body=""
    for fold in 1 2 3 4 5; do
        body+="echo '=====' cholect50_verb fold=$fold '=====' && "
        body+="CUDA_VISIBLE_DEVICES=3 python scripts/reproduce_baseline.py --root . --task cholect50_verb --fold $fold && "
    done
    body+="echo '===== DONE ($verb_session) ====='; exec bash"
    tmux new-session -d -s "$verb_session" "bash -c \"$BOOT && $body\""
    echo "[start] $verb_session  (GPU 3, folds 1..5)"
fi

echo
tmux ls | grep '^stag_pl_' || true
