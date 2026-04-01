#!/usr/bin/env sh
set -eu

SERVICE_PATH="${1:-/etc/systemd/system/trademl-node.service}"
WORKSPACE_ROOT="${2:-$HOME/trademl-node}"

"${HOME}/.local/bin/trademl" node install-service --workspace-root "${WORKSPACE_ROOT}" --config "${WORKSPACE_ROOT}/node.yml" --env-file "${WORKSPACE_ROOT}/.env" --service-path "${SERVICE_PATH}"
echo "TradeML systemd unit written to ${SERVICE_PATH}"
echo "Run: sudo systemctl daemon-reload && sudo systemctl enable --now trademl-node.service"
