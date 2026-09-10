#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAPLACE_CONVENTIONAL_VENV:-$ROOT/.venv}"
DATA_ROOT="${1:?usage: scripts/train-modalities.sh /path/to/corpus [state-dir] [output-dir] [project-config]}"
STATE="${2:-$ROOT/.state}"
OUT="${3:-$ROOT/checkpoints/modalities}"
PROJECT_CONFIG="${4:-$ROOT/config/default.toml}"
AUTO_RESUME="${AUTO_RESUME:-1}"

for required in "$STATE/corpus/manifest.jsonl" "$STATE/modalities.json"; do
  [[ -f "$required" ]] || { echo "missing configured state: $required" >&2; exit 2; }
done

VALIDATION_PER_10K="$("$VENV/bin/python" - "$PROJECT_CONFIG" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], 'rb') as f:
    cfg=tomllib.load(f)
print(int(cfg.get('corpus', {}).get('validation_per_10k', 100)))
PY
)"

mapfile -t MODALITIES < <("$VENV/bin/python" - "$STATE/modalities.json" <<'PY'
import json, sys
plan=json.load(open(sys.argv[1], encoding='utf-8'))
for name in ('image','audio','video'):
    item=plan.get(name, {})
    if item.get('enabled') and item.get('present'):
        print(name)
PY
)

for modality in "${MODALITIES[@]}"; do
  echo "==== train modality: $modality ===="
  resume_args=()
  latest="$OUT/$modality/latest-checkpoint.txt"
  if [[ "$AUTO_RESUME" != "0" && -f "$latest" ]]; then
    checkpoint="$(cat "$latest")"
    [[ -d "$checkpoint" ]] || { echo "latest checkpoint path is missing: $checkpoint" >&2; exit 2; }
    resume_args=(--resume "$checkpoint")
  fi
  "$VENV/bin/python" -m laplace_conventional.modality_train \
    --modality "$modality" \
    --root "$DATA_ROOT" \
    --manifest "$STATE/corpus/manifest.jsonl" \
    --plan "$STATE/modalities.json" \
    --output "$OUT/$modality" \
    --validation-per-10k "$VALIDATION_PER_10K" \
    "${resume_args[@]}"
done
