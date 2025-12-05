#!/usr/bin/env bash
#
# Run data_node commands using the virtual environment Python.
#
# Usage:
#   ./scripts/run_data_node.sh start         # Start the data node (foreground with UI)
#   ./scripts/run_data_node.sh start --bg    # Start in background (no UI, writes PID file)
#   ./scripts/run_data_node.sh stop          # Stop the running data node
#   ./scripts/run_data_node.sh restart       # Stop and start in background
#   ./scripts/run_data_node.sh status        # Check queue and service status
#   ./scripts/run_data_node.sh logs          # Tail the log file
#   ./scripts/run_data_node.sh enable        # Enable systemd service (Linux only)
#   ./scripts/run_data_node.sh disable       # Disable systemd service (Linux only)
#   ./scripts/run_data_node.sh selfcheck     # Run self-checks
#   ./scripts/run_data_node.sh --help        # Show this help
#
# This wrapper ensures you use the correct Python with all dependencies installed.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
PID_FILE="$DATA_ROOT/data_node.pid"
LOG_FILE="$DATA_ROOT/logs/data_node.log"
SERVICE_NAME="trademl-data-node"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

check_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        error "Virtual environment not found at $VENV"
        echo "Run ./scripts/bootstrap_data_node.sh first to set up the environment."
        exit 1
    fi
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

is_running() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

do_start() {
    local background=false

    # Check for --bg flag
    for arg in "$@"; do
        if [ "$arg" = "--bg" ] || [ "$arg" = "-b" ]; then
            background=true
        fi
    done

    if is_running; then
        warn "Data node is already running (PID: $(get_pid))"
        return 1
    fi

    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$PID_FILE")"

    if [ "$background" = true ]; then
        info "Starting data node in background..."
        nohup "$VENV/bin/python" -m data_node --no-ui >> "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        sleep 1

        if kill -0 "$pid" 2>/dev/null; then
            info "Data node started (PID: $pid)"
            info "Logs: $LOG_FILE"
        else
            error "Data node failed to start. Check logs: $LOG_FILE"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        info "Starting data node (foreground with UI)..."
        info "Press Ctrl-C to stop"
        exec "$VENV/bin/python" -m data_node
    fi
}

do_stop() {
    if ! is_running; then
        warn "Data node is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(get_pid)
    info "Stopping data node (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    local count=0
    while [ $count -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        warn "Process still running, sending SIGKILL..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$PID_FILE"
    info "Data node stopped"
}

do_restart() {
    do_stop
    sleep 1
    do_start --bg
}

do_status() {
    echo "============================================================"
    echo "  Pi Data-Node Status"
    echo "============================================================"
    echo ""

    # Service status
    if is_running; then
        info "Service: RUNNING (PID: $(get_pid))"
    else
        warn "Service: STOPPED"
    fi

    # Systemd status (Linux only)
    if command -v systemctl >/dev/null 2>&1; then
        if systemctl is-enabled "$SERVICE_NAME" 2>/dev/null | grep -q "enabled"; then
            info "Systemd: ENABLED (starts on boot)"
        else
            warn "Systemd: DISABLED (not auto-start)"
        fi
    fi

    echo ""

    # Queue status from Python
    "$VENV/bin/python" -m data_node --status 2>/dev/null || true
}

do_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        error "Log file not found: $LOG_FILE"
        exit 1
    fi
}

do_selfcheck() {
    "$VENV/bin/python" -m data_node --selfcheck
}

do_enable() {
    if ! command -v systemctl >/dev/null 2>&1; then
        error "systemctl not found - this command is for Linux with systemd"
        exit 1
    fi

    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"

    if [ ! -f "$service_file" ]; then
        info "Creating systemd service file..."

        # Get current user
        local user
        user=$(whoami)

        sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=TradeML Pi Data-Node Service
After=network.target

[Service]
Type=simple
User=$user
WorkingDirectory=$ROOT
Environment=DATA_ROOT=$DATA_ROOT
ExecStart=$ROOT/scripts/run_data_node.sh start --bg
ExecStop=$ROOT/scripts/run_data_node.sh stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
    fi

    sudo systemctl enable "$SERVICE_NAME"
    info "Systemd service enabled"
    info "The data node will start automatically on boot"
    echo ""
    echo "Commands:"
    echo "  sudo systemctl start $SERVICE_NAME   # Start now"
    echo "  sudo systemctl stop $SERVICE_NAME    # Stop"
    echo "  sudo systemctl status $SERVICE_NAME  # Status"
}

do_disable() {
    if ! command -v systemctl >/dev/null 2>&1; then
        error "systemctl not found - this command is for Linux with systemd"
        exit 1
    fi

    # Stop if running via systemd
    if systemctl is-active "$SERVICE_NAME" 2>/dev/null | grep -q "active"; then
        info "Stopping systemd service..."
        sudo systemctl stop "$SERVICE_NAME"
    fi

    # Disable
    if systemctl is-enabled "$SERVICE_NAME" 2>/dev/null | grep -q "enabled"; then
        sudo systemctl disable "$SERVICE_NAME"
        info "Systemd service disabled"
    else
        warn "Systemd service was not enabled"
    fi

    # Also stop any process we started
    do_stop
}

show_help() {
    cat << EOF
Pi Data-Node Runner

Usage: $(basename "$0") <command> [options]

Commands:
  start          Start the data node (foreground with Rich UI)
  start --bg     Start in background (no UI, for production)
  stop           Stop the running data node
  restart        Stop and restart in background
  status         Show service and queue status
  logs           Tail the log file
  selfcheck      Run self-checks
  enable         Enable systemd auto-start (Linux)
  disable        Disable systemd and stop service

Examples:
  $(basename "$0") start          # Interactive mode with dashboard
  $(basename "$0") start --bg     # Background mode for production
  $(basename "$0") status         # Check what's happening
  $(basename "$0") logs           # Watch the logs

Files:
  PID file: $PID_FILE
  Log file: $LOG_FILE
EOF
}

# Main
check_venv

case "${1:-}" in
    start)
        shift
        do_start "$@"
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    logs)
        do_logs
        ;;
    selfcheck)
        do_selfcheck
        ;;
    enable)
        do_enable
        ;;
    disable)
        do_disable
        ;;
    -h|--help|help)
        show_help
        ;;
    "")
        # No command - default to start
        do_start
        ;;
    *)
        # Pass through to python module for other flags
        exec "$VENV/bin/python" -m data_node "$@"
        ;;
esac
