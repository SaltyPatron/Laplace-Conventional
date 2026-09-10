#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
STATE="${1:-$ROOT/.state}"
OUT="${2:-$ROOT/checkpoints}"
exec "$VENV/bin/python" -m laplace_conventional.train \
  --config "$STATE/training.json" \
  --data "$STATE/data" \
  --dataset-report "$STATE/data/dataset-report.json" \
  --tokenizer "$STATE/tokenizer/tokenizer.model" \
  --output "$OUT" "${@:3}"
