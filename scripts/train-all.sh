#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/paths.sh"
DATA_ROOT="${1:?usage: scripts/train-all.sh /path/to/corpus [state-dir] [output-dir] [project-config]}"
STATE="${2:-$STATE_DEFAULT}"
OUT="${3:-$OUTPUT_DEFAULT}"
PROJECT_CONFIG="${4:-$ROOT/config/default.toml}"
mkdir -p "$OUT"

# The selected corpus is not considered trained if only one admitted provider
# ran. Each lane gets its own checkpoint namespace and execution receipt.
"$ROOT/scripts/train.sh" "$STATE" "$OUT/text" "$PROJECT_CONFIG"
"$ROOT/scripts/train-modalities.sh" "$DATA_ROOT" "$STATE" "$OUT/modalities" "$PROJECT_CONFIG"

python_bin="$VENV/bin/python"
"$python_bin" - "$STATE" "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
state=Path(sys.argv[1]); out=Path(sys.argv[2])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt={
    "state": {
        name: {"path": str(state/name), "sha256": sha(state/name)}
        for name in ("training.json", "execution.json", "modalities.json", "hardware.json")
    },
    "outputs": {
        "text": str(out/"text"),
        "modalities": str(out/"modalities"),
    },
}
(out/"run-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
PY
