#!/usr/bin/env bash
# Launch STAG training tasks in separate tmux sessions, one GPU per session.
#
# GPU assignment (4 x H100):
#   GPU 0 : stag_cholec80_phase     - Cholec80 phase recognition
#   GPU 1 : stag_cholec80_tool      - Cholec80 tool presence detection
#   GPU 2 : stag_grasp_tool         - GraSP tool presence detection
#   GPU 3 : stag_cholect50_verb     - CholecT50 verb recognition (5 folds, sequential)
#
# Each session:
#   - activates the .venv-lemon virtualenv
#   - pins CUDA_VISIBLE_DEVICES
#   - runs `python scripts/train_stag.py ...`
#   - the trainer mirrors all stdout/stderr into runs/<task>/<ts>/log.txt
#     and writes config.json / metrics.jsonl / best.json / checkpoint_best.pt /
#     predictions_test.npz / run_summary.txt in the same directory.
#
# Usage:
#   bash scripts/launch_tmux.sh               # defaults: 50 epochs, lr 3e-4
#   EPOCHS=80 LR=1e-4 bash scripts/launch_tmux.sh
#
# Attaching to a session (view live output):
#   tmux ls                                   # list all sessions
#   tmux attach -t stag_cholec80_phase         # attach
#   # inside tmux: detach with Ctrl-b then d
#
# Killing all STAG sessions:
#   for s in $(tmux ls -F '#{session_name}' | grep '^stag_'); do
#       tmux kill-session -t "$s"
#   done
#
# Inspecting results after runs finish:
#   ls runs/
#   cat runs/cholec80_phase/<latest_ts>/run_summary.txt

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

# Boot commands prepended inside each session so the env is ready.
BOOT="export PATH=\"\$HOME/.local/bin:\$PATH\" && source $VENV_DIR/bin/activate"

# Common trainer args
BASE_ARGS="--root . --epochs $EPOCHS --lr $LR --seed $SEED"

# --------- Single-run tasks: one session, one GPU ---------

launch_single() {
    local session="$1"
    local gpu="$2"
    local task="$3"

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[skip] session $session already exists"
        return
    fi
    local cmd="$BOOT && CUDA_VISIBLE_DEVICES=$gpu python scripts/train_stag.py $BASE_ARGS --task $task; echo; echo '===== DONE ($session) ====='; exec bash"
    tmux new-session -d -s "$session" "bash -c \"$cmd\""
    echo "[start] $session  (GPU $gpu, task=$task)"
}

launch_single "stag_cholec80_phase" 0 cholec80_phase
launch_single "stag_cholec80_tool"  1 cholec80_tool
launch_single "stag_grasp_tool"     2 grasp_tool

# --------- CholecT50: 5 folds, sequential in one session on GPU 3 ---------

verb_session="stag_cholect50_verb"
if tmux has-session -t "$verb_session" 2>/dev/null; then
    echo "[skip] session $verb_session already exists"
else
    verb_loop=""
    for fold in 1 2 3 4 5; do
        verb_loop+="echo '===== CholecT50 verb fold $fold =====' && "
        verb_loop+="CUDA_VISIBLE_DEVICES=3 python scripts/train_stag.py $BASE_ARGS --task cholect50_verb --fold $fold && "
    done
    verb_loop+="echo '===== DONE ($verb_session) ====='; exec bash"

    tmux new-session -d -s "$verb_session" "bash -c \"$BOOT && $verb_loop\""
    echo "[start] $verb_session  (GPU 3, task=cholect50_verb folds 1..5 sequential)"
fi

# --------- Summary ---------

echo
echo "Active STAG sessions:"
tmux ls | grep '^stag_' || echo "  (none)"

cat <<EOF

Watch a session:
    tmux attach -t stag_cholec80_phase
Detach:  Ctrl-b then d
Kill all: for s in \$(tmux ls -F '#{session_name}' | grep '^stag_'); do tmux kill-session -t "\$s"; done

Each run writes:
    runs/<task>/<timestamp>/config.json
    runs/<task>/<timestamp>/log.txt
    runs/<task>/<timestamp>/metrics.jsonl
    runs/<task>/<timestamp>/best.json
    runs/<task>/<timestamp>/checkpoint_best.pt
    runs/<task>/<timestamp>/predictions_test.npz
    runs/<task>/<timestamp>/run_summary.txt
EOF
