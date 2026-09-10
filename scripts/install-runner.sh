#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-SaltyPatron/Laplace-Conventional}"
RUNNER_VERSION="${RUNNER_VERSION:-2.337.0}"
RUNNER_ROOT="${RUNNER_ROOT:-$HOME/actions-runner-laplace-conventional}"
WORK_ROOT="${WORK_ROOT:-$HOME/actions-runner-laplace-conventional-work}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-conventional}"
LABELS="${LABELS:-laplace-conventional,gpu,pascal,1080ti}"
TOKEN="${RUNNER_TOKEN:-}"

for c in curl tar; do command -v "$c" >/dev/null || { echo "$c is required" >&2; exit 2; }; done
if [[ -z "$TOKEN" ]]; then
  command -v gh >/dev/null || { echo "Set RUNNER_TOKEN or install/authenticate gh" >&2; exit 2; }
  TOKEN="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)"
fi
mkdir -p "$RUNNER_ROOT" "$WORK_ROOT"
if [[ ! -x "$RUNNER_ROOT/config.sh" ]]; then
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' EXIT
  curl -fL "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz" -o "$tmp"
  tar -xzf "$tmp" -C "$RUNNER_ROOT"
fi
cd "$RUNNER_ROOT"
./config.sh --unattended --replace --url "https://github.com/$REPO" --token "$TOKEN" --name "$RUNNER_NAME" --work "$WORK_ROOT" --labels "$LABELS"
if [[ "${INSTALL_SERVICE:-0}" == "1" ]]; then
  sudo ./svc.sh install "$(id -un)"
  sudo ./svc.sh start
else
  echo "registered; run ./run.sh or set INSTALL_SERVICE=1 to install the service"
fi
