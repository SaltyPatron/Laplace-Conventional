#!/usr/bin/env bash
# sudo bash scripts/install-runner.sh: register this repository's service runner.
set -euo pipefail
umask 0002
REPO=SaltyPatron/Laplace-Conventional
RUNNER_USER=laplace-runner
RUNNER_GROUP=laplace-runner
RUNNER_VERSION=2.337.0
RUNNER_SHA256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
RUNNER_ROOT=/var/lib/agents/laplace-runner/actions-runner-conventional
WORK_ROOT=/build/laplace/work/conventional-runner
SCRATCH_ROOT=/build/laplace/work/conventional-scratch
RUNNER_NAME=hart-server-conventional
LABELS=laplace-conventional,gpu,pascal,1080ti
SERVICE=actions.runner.SaltyPatron-Laplace-Conventional.hart-server-conventional.service
OPERATOR="${LAPLACE_OPERATOR:-${SUDO_USER:-}}"

if [[ $EUID != 0 ]]; then
  exec sudo bash "$0" "$@"
fi
for command in curl tar sha256sum runuser systemctl python3 gh mountpoint; do
  command -v "$command" >/dev/null || { echo "Required command missing: $command" >&2; exit 2; }
done
[[ -n "$OPERATOR" && "$OPERATOR" != root ]] || {
  echo 'Run with sudo from the operator account so gh uses that account.' >&2; exit 2;
}
mountpoint -q /build || { echo '/build must be mounted' >&2; exit 2; }
mountpoint -q /var/lib/agents || { echo '/var/lib/agents must be mounted' >&2; exit 2; }
getent group "$RUNNER_GROUP" >/dev/null || groupadd --system "$RUNNER_GROUP"
if ! id "$RUNNER_USER" >/dev/null 2>&1; then
  useradd --system --gid "$RUNNER_GROUP" --home-dir /var/lib/agents/laplace-runner --create-home --shell /usr/sbin/nologin "$RUNNER_USER"
fi
usermod -aG "$RUNNER_GROUP" "$OPERATOR"
[[ $(id -gn "$RUNNER_USER") == "$RUNNER_GROUP" ]] || { echo 'Runner primary group differs' >&2; exit 2; }
for directory in /build/laplace /build/laplace/work "$RUNNER_ROOT" "$WORK_ROOT" "$SCRATCH_ROOT"; do
  [[ ! -L "$directory" ]] || { echo "Refusing symlink directory: $directory" >&2; exit 2; }
  install -d -g "$RUNNER_GROUP" -m 2770 "$directory"
done
export TMPDIR="$SCRATCH_ROOT" TMP="$SCRATCH_ROOT" TEMP="$SCRATCH_ROOT"
runuser -u "$OPERATOR" -- gh repo view "$REPO" --json viewerPermission --jq .viewerPermission | grep -qx ADMIN
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'GTX 1080 Ti'

# Fetch only the upstream distribution, never another runner's registration or credentials.
archive="$SCRATCH_ROOT/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz"
if [[ ! -x "$RUNNER_ROOT/config.sh" ]]; then
  if ! printf '%s  %s\n' "$RUNNER_SHA256" "$archive" | sha256sum --check --status 2>/dev/null; then
    curl --fail --location --retry 3 "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz" -o "$archive.download"
    printf '%s  %s\n' "$RUNNER_SHA256" "$archive.download" | sha256sum --check --status
    mv "$archive.download" "$archive"
  fi
  printf '%s  %s\n' "$RUNNER_SHA256" "$archive" | sha256sum --check --status
  runuser -u "$RUNNER_USER" -- tar -xzf "$archive" -C "$RUNNER_ROOT"
fi
cd "$RUNNER_ROOT"
if [[ -f .runner ]]; then
  python3 - "$REPO" "$RUNNER_NAME" "$WORK_ROOT" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path('.runner').read_text())
repo, name, work = sys.argv[1:]
if (config.get('gitHubUrl', '').rstrip('/').lower() != ('https://github.com/' + repo).lower()
        or config.get('agentName') != name or config.get('workFolder') != work):
    raise SystemExit('Existing runner registration differs; it was not replaced.')
PY
else
  token="$(runuser -u "$OPERATOR" -- gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token)"
  runuser -u "$RUNNER_USER" -- ./config.sh --unattended \
    --url "https://github.com/$REPO" --token "$token" --name "$RUNNER_NAME" \
    --work "$WORK_ROOT" --labels "$LABELS"
  unset token
fi
if ! systemctl cat "$SERVICE" >/dev/null 2>&1; then
  ./svc.sh install "$RUNNER_USER"
fi
[[ $(systemctl show "$SERVICE" -p User --value) == "$RUNNER_USER" ]] || {
  echo 'Existing service has a different user' >&2; exit 2;
}
install -d -m 0755 "/etc/systemd/system/$SERVICE.d"
cat > "/etc/systemd/system/$SERVICE.d/50-laplace-storage.conf" <<UNIT
[Unit]
RequiresMountsFor=/build /var/lib/agents

[Service]
Group=$RUNNER_GROUP
UMask=0002
Environment=TMPDIR=$SCRATCH_ROOT
Environment=TMP=$SCRATCH_ROOT
Environment=TEMP=$SCRATCH_ROOT
UNIT
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"
systemctl is-active --quiet "$SERVICE"
for _attempt in {1..20}; do
  status="$(runuser -u "$OPERATOR" -- gh api "repos/$REPO/actions/runners" --jq '.runners[] | select(.name == "hart-server-conventional") | .status')"
  if [[ "$status" == online ]]; then
    echo "ONLINE: $RUNNER_NAME, user=$RUNNER_USER group=$RUNNER_GROUP work=$WORK_ROOT scratch=$SCRATCH_ROOT"
    exit 0
  fi
  sleep 2
done
echo 'Service started but GitHub has not reported the runner online.' >&2
exit 1
