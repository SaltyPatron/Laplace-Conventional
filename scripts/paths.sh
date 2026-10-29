# Shared output locations. Source and corpus inputs never own generated state.
VENV="${LAPLACE_CONVENTIONAL_VENV:-${LAPLACE_BUILD:-${XDG_CACHE_HOME:-$HOME/.cache}/laplace/build}/Laplace-Conventional/venv}"
STATE_DEFAULT="${LAPLACE_CONVENTIONAL_STATE:-${LAPLACE_WORK:-${XDG_STATE_HOME:-$HOME/.local/state}/laplace}/conventional}"
OUTPUT_DEFAULT="${LAPLACE_CONVENTIONAL_OUTPUT:-${LAPLACE_MODELS:-${XDG_DATA_HOME:-$HOME/.local/share}/laplace/models}/conventional/checkpoints}"
for conventional_path in "$VENV" "$STATE_DEFAULT" "$OUTPUT_DEFAULT"; do
  [[ "$conventional_path" = /* ]] || { echo "Conventional output roots must be absolute: $conventional_path" >&2; return 2; }
done
unset conventional_path
