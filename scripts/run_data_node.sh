#!/usr/bin/env bash
#
# Run data_node commands using the virtual environment Python.
#
# Usage:
#   ./scripts/run_data_node.sh              # Start the data node service
#   ./scripts/run_data_node.sh --status     # Check queue status
#   ./scripts/run_data_node.sh --help       # Show help
#
# This wrapper ensures you use the correct Python with all dependencies installed.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "Error: Virtual environment not found at $VENV"
    echo "Run ./scripts/bootstrap_data_node.sh first to set up the environment."
    exit 1
fi

exec "$VENV/bin/python" -m data_node "$@"
