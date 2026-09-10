#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
STATE="${1:-$ROOT/.state}"
OUT="${2:-$ROOT/checkpoints}"
PROJECT_CONFIG="${3:-$ROOT/config/default.toml}"

for required in "$STATE/training.json" "$STATE/execution.json" "$STATE/data/dataset-report.json" "$STATE/tokenizer/tokenizer.model"; do
  [[ -f "$required" ]] || { echo "missing configured state: $required" >&2; exit 2; }
done

if [[ -z "${MICRO_BATCH_SIZE:-}" ]]; then
  MICRO_BATCH_SIZE="$("$VENV/bin/python" - "$STATE/execution.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["micro_batch_size"]))
PY
)"
  export MICRO_BATCH_SIZE
fi

exec "$VENV/bin/python" -m laplace_conventional.train \
  --config "$STATE/training.json" \
  --execution-plan "$STATE/execution.json" \
  --data "$STATE/data" \
  --dataset-report "$STATE/data/dataset-report.json" \
  --tokenizer "$STATE/tokenizer/tokenizer.model" \
  --output "$OUT" "${@:4}"
