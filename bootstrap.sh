#!/usr/bin/env bash
# One-shot bootstrap for the Raspberry Pi (or Mac) data-collection wizard.
# Ensures Python 3.10+ exists, creates/uses ./venv, then invokes rpi_wizard.py.
# Usage:
#   ./bootstrap.sh           # run wizard
#   ./bootstrap.sh stop      # stop running node loop (from state file or process name)

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
STATE_FILE="$ROOT/logs/rpi_wizard_state.json"
SYSTEMD_UNIT="trademl-node.service"

stop_node() {
  # Try state file pid first
  if [ -f "$STATE_FILE" ]; then
    pid=$(python3 - "$STATE_FILE" <<'PY'
import json, sys
path=sys.argv[1]
try:
    data=json.loads(open(path).read())
    pid=int(data.get("last_node_pid","0"))
    if pid>0:
        print(pid)
except Exception:
    pass
PY
)
    if [ -n "${pid:-}" ]; then
      if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping node pid $pid"
        kill "$pid" || true
        sleep 1
      fi
    fi
  fi
  # Fallback: kill by process name under repo
  pids=$(pgrep -f "$ROOT/scripts/node.py" || true)
  if [ -n "$pids" ]; then
    echo "Stopping node processes: $pids"
    kill $pids || true
  fi

  stop_systemd_unit
  exit 0
}

stop_systemd_unit() {
  if [ "$(uname -s)" != "Linux" ]; then
    return
  fi

  systemctl_bin=$(command -v systemctl || true)
  if [ -z "$systemctl_bin" ]; then
    echo "systemctl not found; skipping systemd stop"
    return
  fi

  unit_path="$HOME/.config/systemd/user/$SYSTEMD_UNIT"
  if ! "$systemctl_bin" --user list-unit-files "$SYSTEMD_UNIT" >/dev/null 2>&1 && [ ! -f "$unit_path" ]; then
    echo "Systemd unit $SYSTEMD_UNIT not found; nothing to stop/disable"
    return
  fi

  if "$systemctl_bin" --user is-active "$SYSTEMD_UNIT" >/dev/null 2>&1; then
    echo "Stopping systemd unit $SYSTEMD_UNIT"
    "$systemctl_bin" --user stop "$SYSTEMD_UNIT" || echo "Warning: failed to stop $SYSTEMD_UNIT"
  else
    echo "Systemd unit $SYSTEMD_UNIT is not active"
  fi

  if "$systemctl_bin" --user is-enabled "$SYSTEMD_UNIT" >/dev/null 2>&1; then
    echo "Disabling systemd unit $SYSTEMD_UNIT"
    "$systemctl_bin" --user disable "$SYSTEMD_UNIT" || echo "Warning: failed to disable $SYSTEMD_UNIT"
  else
    echo "Systemd unit $SYSTEMD_UNIT already disabled"
  fi

  "$systemctl_bin" --user daemon-reload || true
}

if [ "${1:-}" = "stop" ]; then
  stop_node
fi

choose_python() {
  for cmd in python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ver=$("$cmd" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
      case "$ver" in
        3.10|3.11|3.12|3.13) echo "$cmd"; return 0 ;;
      esac
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [ -z "${PYTHON_BIN:-}" ]; then
  echo "Python 3.10+ not found. Install python3.11 (apt-get install python3.11 python3.11-venv or brew install python@3.11) and re-run." >&2
  exit 1
fi

validate_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    echo "Venv missing or not executable at $VENV/bin/python"
    return 1
  fi

  venv_info=$("$VENV/bin/python" - <<'PY' || true
import platform, sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
print(sys.platform)
print(platform.machine())
PY
)
  if [ -z "$venv_info" ]; then
    echo "Existing venv python failed to start"
    return 1
  fi

  host_info=$("$PYTHON_BIN" - <<'PY'
import platform, sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
print(sys.platform)
print(platform.machine())
PY
)

  venv_version=$(printf '%s\n' "$venv_info" | sed -n '1p')
  venv_platform=$(printf '%s\n' "$venv_info" | sed -n '2p')
  venv_machine=$(printf '%s\n' "$venv_info" | sed -n '3p')
  host_version=$(printf '%s\n' "$host_info" | sed -n '1p')
  host_platform=$(printf '%s\n' "$host_info" | sed -n '2p')
  host_machine=$(printf '%s\n' "$host_info" | sed -n '3p')

  if [ "$venv_version" != "$host_version" ] || [ "$venv_platform" != "$host_platform" ] || [ "$venv_machine" != "$host_machine" ]; then
    echo "Existing venv is built for $venv_version/$venv_platform/$venv_machine, host expects $host_version/$host_platform/$host_machine"
    return 1
  fi
  return 0
}

if ! validate_venv; then
  echo "Resetting venv at $VENV using $PYTHON_BIN"
  rm -rf "$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
fi

echo "Running wizard with $VENV/bin/python"
exec "$VENV/bin/python" "$ROOT/rpi_wizard.py" "$@"
