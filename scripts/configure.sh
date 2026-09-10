#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
DATA_ROOT="${1:?usage: scripts/configure.sh /path/to/corpus [state-dir]}"
STATE="${2:-$ROOT/.state}"
mkdir -p "$STATE"
LC="$VENV/bin/laplace-conventional"

"$LC" inventory "$DATA_ROOT" --out "$STATE/corpus"
"$LC" tokenizer "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --out "$STATE/tokenizer"
"$LC" prepare "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --tokenizer "$STATE/tokenizer/tokenizer.model" --out "$STATE/data"
"$LC" derive-config --dataset-report "$STATE/data/dataset-report.json" --out "$STATE/training.json"
"$LC" hardware --out "$STATE/hardware.json"

"$VENV/bin/python" - "$STATE/corpus/summary.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
if s["unsupported_unique_bytes"]:
    raise SystemExit(
        f"configuration stopped: {s['unsupported_unique_bytes']} unique corpus bytes are not trainable by the current generic readers. "
        "Inspect corpus/summary.json; do not start a 'whole corpus' training run until coverage is explicit."
    )
PY

echo "configuration complete: $STATE"
