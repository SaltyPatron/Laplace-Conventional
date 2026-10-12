#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
INSTALL_OFFLOAD="${INSTALL_OFFLOAD:-0}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

command -v "$PYTHON" >/dev/null || { echo "python3 is required" >&2; exit 2; }
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
PY

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ -z "$TORCH_INDEX_URL" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  if "$VENV/bin/python" -c 'from laplace_conventional.host_policy import gpu_allowed; raise SystemExit(0 if gpu_allowed() else 1)'; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
  fi
fi
if [[ "$INSTALL_OFFLOAD" != 0 || "$TORCH_INDEX_URL" != https://download.pytorch.org/whl/cpu ]]; then
  "$VENV/bin/python" -m laplace_conventional.host_policy --require
fi

"$VENV/bin/python" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
extras="train,media,rl,test"
if [[ "$INSTALL_OFFLOAD" != "0" ]]; then
  extras="$extras,offload"
fi
"$VENV/bin/python" -m pip install -e "$ROOT[$extras]"

"$VENV/bin/python" - <<'PY'
import json, sys, torch
import imageio_ffmpeg
from PIL import Image
from laplace_conventional.host_policy import gpu_allowed
info={
    "python":sys.version.split()[0],
    "torch":torch.__version__,
    "gpu_allowed":gpu_allowed(),
    "cuda_available":gpu_allowed() and torch.cuda.is_available(),
    "ffmpeg":imageio_ffmpeg.get_ffmpeg_exe(),
    "pillow":Image.__version__,
}
if info["cuda_available"]:
    info["gpu"]=torch.cuda.get_device_name(0)
    info["compute_capability"]=torch.cuda.get_device_capability(0)
    info["gpu_memory_bytes"]=torch.cuda.get_device_properties(0).total_memory
    if info["compute_capability"][0] < 6:
        raise SystemExit(f"GPU compute capability too old: {info['compute_capability']}")
print(json.dumps(info))
PY

if [[ "$INSTALL_OFFLOAD" != "0" ]]; then
  "$VENV/bin/python" - <<'PY'
import deepspeed
print(f"deepspeed={deepspeed.__version__}")
PY
fi

echo "environment ready: $VENV"
