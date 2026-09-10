#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
DATA_ROOT="${1:?usage: scripts/configure.sh /path/to/corpus [state-dir] [project-config]}"
STATE="${2:-$ROOT/.state}"
PROJECT_CONFIG="${3:-$ROOT/config/default.toml}"
mkdir -p "$STATE"
LC=("$VENV/bin/laplace-conventional" --project-config "$PROJECT_CONFIG")

# Account for the physical estate first, then apply explicit training admission.
# Downstream stages consume only the selected manifest, while the physical
# manifest and every inclusion/exclusion decision remain receipted.
"${LC[@]}" inventory "$DATA_ROOT" --out "$STATE/inventory"
"${LC[@]}" select --manifest "$STATE/inventory/manifest.jsonl" --out "$STATE/corpus"

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
        "configuration stopped before training preparation: "
        f"unsupported_selected_bytes={unsupported}, inaccessible_selected_bytes={inaccessible}. "
        "Inspect inventory/manifest.jsonl, corpus/decisions.jsonl, and corpus/summary.json; "
        "implement a training provider, restore access, or make an explicit selection rule with a reason."
    )
PY

# Text/code/structured lane.
"${LC[@]}" tokenizer "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --out "$STATE/tokenizer"
"${LC[@]}" prepare "$DATA_ROOT" --manifest "$STATE/corpus/manifest.jsonl" --tokenizer "$STATE/tokenizer/tokenizer.model" --out "$STATE/data"
"${LC[@]}" derive-config --dataset-report "$STATE/data/dataset-report.json" --out "$STATE/training.json"

# One measured host receipt drives both text and modality execution planning.
"${LC[@]}" hardware --out "$STATE/hardware.json"
"${LC[@]}" plan-execution \
  --training-config "$STATE/training.json" \
  --hardware "$STATE/hardware.json" \
  --out "$STATE/execution.json" \
  --require-fit
"${LC[@]}" plan-modalities \
  --manifest "$STATE/corpus/manifest.jsonl" \
  --hardware "$STATE/hardware.json" \
  --out "$STATE/modalities.json" \
  --require-fit

echo "configuration complete: $STATE"
