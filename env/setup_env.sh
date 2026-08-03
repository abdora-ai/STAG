#!/usr/bin/env bash
# LEMON server environment setup - LRZ HGX H100, Linux x86_64, CUDA 12.4.
#
# Uses `uv` (single-binary Python/package manager) because the system Python on
# LRZ HGX nodes typically ships without `ensurepip`, breaking `python3 -m venv`.
# `uv` also installs its own portable Python if the system one is unusable.
#
# Usage:
#   bash env/setup_env.sh                # creates .venv-lemon in the repo root
#   bash env/setup_env.sh /path/to/venv  # custom location
#
# Idempotent: rerunning reuses existing venv and just tops up deps.

set -euo pipefail

VENV_DIR_ARG="${1:-.venv-lemon}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$VENV_DIR_ARG" in
    /*) VENV_DIR="$VENV_DIR_ARG" ;;
    *)  VENV_DIR="$REPO_ROOT/$VENV_DIR_ARG" ;;
esac
REQ_FILE="$REPO_ROOT/env/requirements-server.txt"

echo "[env] repo root    : $REPO_ROOT"
echo "[env] target venv  : $VENV_DIR"
echo "[env] requirements : $REQ_FILE"

# ---- Optional: load CUDA module (PyTorch wheels bundle their own CUDA, but
#      loading a matching CUDA can help downstream native builds).
if command -v module >/dev/null 2>&1; then
    echo "[env] module system detected; trying cuda/12.4"
    module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
fi

# ---- Install uv if missing (single static binary; no root needed)
if ! command -v uv >/dev/null 2>&1; then
    echo "[env] uv not found on PATH; installing to \$HOME/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv installation did not put uv on PATH."
    echo "        Add ~/.local/bin to PATH permanently in ~/.bashrc:"
    echo "            export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "        Then re-run: bash env/setup_env.sh"
    exit 1
fi

echo "[env] uv version   : $(uv --version)"

# ---- Create venv (uv will download a portable Python 3.10 if the system one is missing)
if [ ! -d "$VENV_DIR" ]; then
    echo "[env] creating venv at $VENV_DIR (Python 3.10)"
    uv venv --python 3.10 "$VENV_DIR"
else
    echo "[env] reusing existing venv at $VENV_DIR"
fi

# ---- Activate for the rest of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- PyTorch 2.5.1 + CUDA 12.4 (matches paper's environment; supports H100 sm_90)
echo "[env] installing torch==2.5.1 torchvision==0.20.1 (cu124 wheels)"
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1

# ---- Remaining dependencies
echo "[env] installing dependencies from $REQ_FILE"
uv pip install -r "$REQ_FILE"

# ---- Sanity check
echo
echo "[env] sanity check:"
python - <<'PY'
import platform
import torch
import torchvision
import numpy
import cv2
import lmdb
import sklearn
import tqdm

print("python     ", platform.python_version())
print("torch      ", torch.__version__)
print("cuda avail?", torch.cuda.is_available())
print("cuda build ", torch.version.cuda)
if hasattr(torch.backends, "cudnn"):
    v = torch.backends.cudnn.version()
    print("cudnn      ", v)
print("n_devices  ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"  gpu[{i}]  = {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")
print("torchvision", torchvision.__version__)
print("numpy      ", numpy.__version__)
print("opencv     ", cv2.__version__)
print("lmdb       ", lmdb.__version__)
print("sklearn    ", sklearn.__version__)
print("tqdm       ", tqdm.__version__)
PY

echo
echo "[env] done. Activate later with: source $VENV_DIR/bin/activate"
