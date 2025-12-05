#!/usr/bin/env bash
#
# Bootstrap script for Pi Data-Node Wizard
#
# Sets up the Python virtual environment, installs dependencies,
# and runs the interactive wizard for configuring the data node.
#
# Usage:
#   ./scripts/bootstrap_data_node.sh             # Run wizard
#   ./scripts/bootstrap_data_node.sh --resume    # Resume from saved state
#   ./scripts/bootstrap_data_node.sh --dry-run   # Show planned actions
#
# This script ensures all dependencies are installed before running
# the Python wizard, avoiding import errors.

set -euo pipefail

# Resolve repo root (parent of scripts/)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"
WIZARD="$ROOT/scripts/pi_data_node_wizard.py"

# Minimum required dependencies for the wizard
DEPS="loguru pyyaml python-dotenv"

echo "============================================================"
echo "  Pi Data-Node Bootstrap"
echo "============================================================"
echo ""

# Find a suitable Python 3.10+
choose_python() {
  for cmd in python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
      case "$ver" in
        3.10|3.11|3.12|3.13)
          echo "$cmd"
          return 0
          ;;
      esac
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [ -z "${PYTHON_BIN:-}" ]; then
  echo "  ERROR: Python 3.10+ not found."
  echo ""
  echo "  Install Python 3.11:"
  echo "    Debian/Ubuntu: sudo apt-get install python3.11 python3.11-venv"
  echo "    macOS:         brew install python@3.11"
  echo ""
  exit 1
fi

echo "  Python: $PYTHON_BIN"

# Validate existing venv or create new one
validate_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    return 1
  fi

  # Check version and platform match
  venv_info=$("$VENV/bin/python" -c 'import sys, platform; print(f"{sys.version_info.major}.{sys.version_info.minor}"); print(sys.platform); print(platform.machine())' 2>/dev/null || echo "")
  if [ -z "$venv_info" ]; then
    return 1
  fi

  host_info=$("$PYTHON_BIN" -c 'import sys, platform; print(f"{sys.version_info.major}.{sys.version_info.minor}"); print(sys.platform); print(platform.machine())')

  venv_version=$(echo "$venv_info" | sed -n '1p')
  venv_platform=$(echo "$venv_info" | sed -n '2p')
  venv_machine=$(echo "$venv_info" | sed -n '3p')
  host_version=$(echo "$host_info" | sed -n '1p')
  host_platform=$(echo "$host_info" | sed -n '2p')
  host_machine=$(echo "$host_info" | sed -n '3p')

  if [ "$venv_version" != "$host_version" ] || [ "$venv_platform" != "$host_platform" ] || [ "$venv_machine" != "$host_machine" ]; then
    echo "  Venv mismatch: $venv_version/$venv_platform/$venv_machine vs $host_version/$host_platform/$host_machine"
    return 1
  fi

  return 0
}

if ! validate_venv; then
  echo "  Creating virtual environment..."
  rm -rf "$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi

echo "  Venv: $VENV"

# Upgrade pip and install dependencies
echo ""
echo "  Installing dependencies..."
"$VENV/bin/pip" install --upgrade pip || { echo "  ERROR: pip upgrade failed"; exit 1; }
"$VENV/bin/pip" install $DEPS || { echo "  ERROR: dependency install failed"; exit 1; }

# Verify installation
echo ""
echo "  Verifying installation..."
"$VENV/bin/python" -c "import loguru; print(f'    loguru: {loguru.__version__}')" || { echo "  ERROR: loguru not installed"; exit 1; }
"$VENV/bin/python" -c "import yaml; print('    pyyaml: OK')" || { echo "  ERROR: pyyaml not installed"; exit 1; }

echo "  Done."
echo ""
echo "============================================================"
echo ""

# Run the wizard
exec "$VENV/bin/python" "$WIZARD" "$@"
