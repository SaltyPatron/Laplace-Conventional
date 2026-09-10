#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
STATE="${1:-$ROOT/.state}"
OUT="${2:-$ROOT/checkpoints}"
PROJECT_CONFIG="${3:-$ROOT/config/default.toml}"
if [[ -z "${MICRO_BATCH_SIZE:-}" ]]; then
  MICRO_BATCH_SIZE="$("$VENV/bin/python" - "$PROJECT_CONFIG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(int(cfg.get("execution", {}).get("micro_batch_size", 1)))
PY
)"
  export MICRO_BATCH_SIZE
fi
exec "$VENV/bin/python" -m laplace_conventional.train \
  --config "$STATE/training.json" \
  --data "$STATE/data" \
  --dataset-report "$STATE/data/dataset-report.json" \
  --tokenizer "$STATE/tokenizer/tokenizer.model" \
  --output "$OUT" "${@:4}"
