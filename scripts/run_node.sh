#!/usr/bin/env bash
set -euo pipefail

# One-click Pi node launcher
# Wrapper that boots the full node (provision + loop) with zero flags.

cd "$(dirname "$0")/.."
exec bash scripts/pi_node.sh up

