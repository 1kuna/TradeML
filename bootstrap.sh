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
  exit 0
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

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating venv at $VENV using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
fi

echo "Running wizard with $VENV/bin/python"
exec "$VENV/bin/python" "$ROOT/rpi_wizard.py" "$@"
