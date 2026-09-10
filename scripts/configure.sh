#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
DATA_ROOT="${1:?usage: scripts/configure.sh /path/to/corpus [state-dir] [project-config]}"
STATE="${2:-$ROOT/.state}"
PROJECT_CONFIG="${3:-$ROOT/config/default.toml}"
mkdir -p "$STATE"
LC=("$VENV/bin/laplace-conventional" --project-config "$PROJECT_CONFIG")

"${LC[@]}" inventory "$DATA_ROOT" --out "$STATE/corpus"
"${LC[@]}" tokenizer "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --out "$STATE/tokenizer"
"${LC[@]}" prepare "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --tokenizer "$STATE/tokenizer/tokenizer.model" --out "$STATE/data"
"${LC[@]}" derive-config --dataset-report "$STATE/data/dataset-report.json" --out "$STATE/training.json"
"${LC[@]}" hardware --out "$STATE/hardware.json"

"$VENV/bin/python" - "$STATE/corpus/summary.json" "$PROJECT_CONFIG" <<'PY'
import json, sys, tomllib
summary = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "rb") as f:
    cfg = tomllib.load(f)
require = bool(cfg.get("corpus", {}).get("require_full_coverage", True))
unsupported = int(summary["unsupported_selected_bytes"])
if require and unsupported:
    raise SystemExit(
        f"configuration stopped: {unsupported} selected corpus bytes are unsupported. "
        "Inspect corpus/summary.json; implement coverage or make an explicit policy change in the project config."
    )
PY

echo "configuration complete: $STATE"
