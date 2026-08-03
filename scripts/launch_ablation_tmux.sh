#!/usr/bin/env bash
# Launch the three ablation modes (linear, prototype, slot_proto) that are
# missing from the current runs/ tree. The "full" mode is already done.
#
# Each session runs sequentially through its three modes on a single GPU.
#
# GPU 0: stag_ab_cholec80_phase   -> 3 modes  (~ 60 min total)
# GPU 1: stag_ab_cholec80_tool    -> 3 modes  (~ 75 min total)
# GPU 2: stag_ab_grasp_tool       -> 3 modes  (~ 20 min total)
# GPU 3: stag_ab_cholect50_verb   -> 3 modes × 5 folds = 15 runs (~ 100 min total)
#
# Each run writes runs/<task_slug>/<mode>/<timestamp>/{config,log,metrics.jsonl,best,checkpoint,predictions,run_summary}.
#
# Usage:
#   bash scripts/launch_ablation_tmux.sh
#   EPOCHS=80 LR=1e-4 bash scripts/launch_ablation_tmux.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[error] tmux is not installed on this node." >&2
    exit 1
fi

VENV_DIR=".venv-lemon"
if [ ! -d "$VENV_DIR" ]; then
    echo "[error] $VENV_DIR does not exist. Run env/setup_env.sh first." >&2
    exit 1
fi

EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"

BOOT="export PATH=\"\$HOME/.local/bin:\$PATH\" && source $VENV_DIR/bin/activate"
BASE_ARGS="--root . --epochs $EPOCHS --lr $LR --seed $SEED"

# Modes to run this pass (skip "full" - already have those results)
MODES="linear prototype slot_proto"

# ---------- single-run tasks: one GPU each ----------

launch_ablation() {
    local session="$1"
    local gpu="$2"
    local task="$3"

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[skip] $session already exists"
        return
    fi

    local body=""
    for m in $MODES; do
        body+="echo '=====' $task mode=$m '=====' && "
        body+="CUDA_VISIBLE_DEVICES=$gpu python scripts/train_stag.py $BASE_ARGS --task $task --mode $m && "
    done
    body+="echo '===== DONE ($session) ====='; exec bash"

    tmux new-session -d -s "$session" "bash -c \"$BOOT && $body\""
    echo "[start] $session  (GPU $gpu, task=$task, modes=$MODES)"
}

launch_ablation "stag_ab_cholec80_phase" 0 cholec80_phase
launch_ablation "stag_ab_cholec80_tool"  1 cholec80_tool
launch_ablation "stag_ab_grasp_tool"     2 grasp_tool

# ---------- CholecT50: 3 modes × 5 folds on GPU 3 ----------

verb_session="stag_ab_cholect50_verb"
if tmux has-session -t "$verb_session" 2>/dev/null; then
    echo "[skip] $verb_session already exists"
else
    body=""
    for m in $MODES; do
        for fold in 1 2 3 4 5; do
            body+="echo '=====' cholect50_verb mode=$m fold=$fold '=====' && "
            body+="CUDA_VISIBLE_DEVICES=3 python scripts/train_stag.py $BASE_ARGS --task cholect50_verb --fold $fold --mode $m && "
        done
    done
    body+="echo '===== DONE ($verb_session) ====='; exec bash"

    tmux new-session -d -s "$verb_session" "bash -c \"$BOOT && $body\""
    echo "[start] $verb_session  (GPU 3, task=cholect50_verb, 3 modes × 5 folds = 15 runs)"
fi

echo
echo "Active ablation sessions:"
tmux ls | grep '^stag_ab_' || echo "  (none)"
cat <<EOF

Watch a session:
    tmux attach -t stag_ab_cholec80_phase
Detach: Ctrl-b then d
Tail a specific mode from outside tmux:
    tail -f runs/cholec80_phase/linear/*/log.txt
Kill all ablation sessions:
    for s in \$(tmux ls -F '#{session_name}' | grep '^stag_ab_'); do tmux kill-session -t "\$s"; done
EOF
