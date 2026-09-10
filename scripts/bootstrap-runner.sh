#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-SaltyPatron/Laplace-Conventional}"
RUNNER_VERSION="${RUNNER_VERSION:-2.336.0}"
RUNNER_ROOT="${RUNNER_ROOT:-/var/lib/agents/laplace-runner/actions-runner-conventional}"
WORK_ROOT="${WORK_ROOT:-/var/lib/agents/laplace-runner/actions-runner-conventional-work}"
RUNNER_NAME="${RUNNER_NAME:-hart-server-conventional}"
LABELS="${LABELS:-laplace-conventional,gpu,pascal,1080ti}"

if [[ "$(id -u)" -eq 0 ]]; then echo "Run as the service account, not root." >&2; exit 2; fi
command -v gh >/dev/null
command -v curl >/dev/null
command -v tar >/dev/null
token="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"
mkdir -p "$RUNNER_ROOT" "$WORK_ROOT"
if [[ ! -x "$RUNNER_ROOT/config.sh" ]]; then
  tmp="$(mktemp)"
  curl -fL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -o "$tmp"
  tar -xzf "$tmp" -C "$RUNNER_ROOT"
  rm -f "$tmp"
fi
cd "$RUNNER_ROOT"
./config.sh --unattended --replace --url "https://github.com/${REPO}" --token "$token" --name "$RUNNER_NAME" --work "$WORK_ROOT" --labels "$LABELS"
echo "Runner configured. Install/start service with:"
echo "  sudo ./svc.sh install $(id -un)"
echo "  sudo ./svc.sh start"
