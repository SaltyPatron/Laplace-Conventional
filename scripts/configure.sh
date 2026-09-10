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
"$VENV/bin/python" - "$STATE/corpus/summary.json" "$PROJECT_CONFIG" <<'PY'
import json, sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
summary = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "rb") as f:
    cfg = tomllib.load(f)
require = bool(cfg.get("corpus", {}).get("require_full_coverage", True))
unsupported = int(summary["unsupported_selected_bytes"])
inaccessible = int(summary["inaccessible_selected_bytes"])
if require and (unsupported or inaccessible):
    raise SystemExit(
        "configuration stopped before tokenizer/preparation: "
        f"unsupported_selected_bytes={unsupported}, inaccessible_selected_bytes={inaccessible}. "
        "Inspect corpus/summary.json and manifest.jsonl; implement coverage/access or make an explicit project-config policy change."
    )
PY

"${LC[@]}" tokenizer "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --out "$STATE/tokenizer"
"${LC[@]}" prepare "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --tokenizer "$STATE/tokenizer/tokenizer.model" --out "$STATE/data"
"${LC[@]}" derive-config --dataset-report "$STATE/data/dataset-report.json" --out "$STATE/training.json"
"${LC[@]}" hardware --out "$STATE/hardware.json"

echo "configuration complete: $STATE"
