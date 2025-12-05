#!/usr/bin/env bash
#
# Unified Pi Data-Node CLI
#
# Handles setup, service management, and operations all in one script.
# Auto-bootstraps the virtual environment if needed.
#
# Usage:
#   ./scripts/data_node.sh setup          # Run setup wizard (first time or reconfigure)
#   ./scripts/data_node.sh start          # Start with Rich UI (foreground)
#   ./scripts/data_node.sh start --bg     # Start in background
#   ./scripts/data_node.sh stop           # Stop the data node
#   ./scripts/data_node.sh restart        # Stop + start in background
#   ./scripts/data_node.sh status         # Show service and queue status
#   ./scripts/data_node.sh logs           # Tail the log file
#   ./scripts/data_node.sh selfcheck      # Run self-checks
#   ./scripts/data_node.sh enable         # Enable systemd auto-start (Linux)
#   ./scripts/data_node.sh disable        # Disable systemd and stop
#   ./scripts/data_node.sh --help         # Show help

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"
WIZARD="$ROOT/scripts/pi_data_node_wizard.py"

# Load .env file if it exists (for DATA_ROOT and other config)
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# Set defaults after loading .env
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
PID_FILE="$DATA_ROOT/data_node.pid"
LOG_FILE="$DATA_ROOT/logs/data_node.log"
SERVICE_NAME="trademl-data-node"

# Requirements file - use lightweight Pi version on ARM
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    REQUIREMENTS="$ROOT/requirements-pi.txt"
else
    REQUIREMENTS="$ROOT/requirements.txt"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header() { echo -e "${BLUE}$*${NC}"; }

# =============================================================================
# Bootstrap Functions
# =============================================================================

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

validate_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        return 1
    fi

    # Check version and platform match
    local python_bin
    python_bin=$(choose_python) || return 1

    venv_info=$("$VENV/bin/python" -c 'import sys, platform; print(f"{sys.version_info.major}.{sys.version_info.minor}"); print(sys.platform); print(platform.machine())' 2>/dev/null || echo "")
    if [ -z "$venv_info" ]; then
        return 1
    fi

    host_info=$("$python_bin" -c 'import sys, platform; print(f"{sys.version_info.major}.{sys.version_info.minor}"); print(sys.platform); print(platform.machine())')

    venv_version=$(echo "$venv_info" | sed -n '1p')
    host_version=$(echo "$host_info" | sed -n '1p')
    venv_machine=$(echo "$venv_info" | sed -n '3p')
    host_machine=$(echo "$host_info" | sed -n '3p')

    if [ "$venv_version" != "$host_version" ] || [ "$venv_machine" != "$host_machine" ]; then
        return 1
    fi

    return 0
}

ensure_venv() {
    # Quick check - if venv exists and is valid, we're done
    if validate_venv; then
        return 0
    fi

    header "============================================================"
    header "  Pi Data-Node Bootstrap"
    header "============================================================"
    echo ""

    local python_bin
    python_bin=$(choose_python || true)
    if [ -z "${python_bin:-}" ]; then
        error "Python 3.10+ not found."
        echo ""
        echo "  Install Python 3.11:"
        echo "    Debian/Ubuntu: sudo apt-get install python3.11 python3.11-venv"
        echo "    macOS:         brew install python@3.11"
        echo ""
        exit 1
    fi

    info "Python: $python_bin"
    info "Arch: $ARCH"
    info "Requirements: $(basename "$REQUIREMENTS")"

    # Create venv
    info "Creating virtual environment..."
    rm -rf "$VENV"
    "$python_bin" -m venv "$VENV"

    # Upgrade pip and install dependencies
    echo ""
    info "Installing dependencies (this may take a few minutes)..."
    "$VENV/bin/pip" install --upgrade pip -q || { error "pip upgrade failed"; exit 1; }

    if [ -f "$REQUIREMENTS" ]; then
        "$VENV/bin/pip" install -r "$REQUIREMENTS" -q || { error "requirements install failed"; exit 1; }
    else
        warn "requirements file not found, installing minimal deps"
        "$VENV/bin/pip" install loguru pyyaml python-dotenv rich exchange-calendars -q || { error "dependency install failed"; exit 1; }
    fi

    # Verify installation
    echo ""
    info "Verifying installation..."
    "$VENV/bin/python" -c "import loguru; print(f'  loguru: {loguru.__version__}')" || { error "loguru not installed"; exit 1; }
    "$VENV/bin/python" -c "import rich; print('  rich: OK')" || { error "rich not installed"; exit 1; }

    info "Bootstrap complete!"
    echo ""
    header "============================================================"
    echo ""
}

# =============================================================================
# Service Management Functions
# =============================================================================

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

do_setup() {
    ensure_venv

    if [ ! -f "$WIZARD" ]; then
        error "Wizard script not found: $WIZARD"
        exit 1
    fi

    exec "$VENV/bin/python" "$WIZARD" "$@"
}

do_start() {
    ensure_venv

    local background=false

    for arg in "$@"; do
        if [ "$arg" = "--bg" ] || [ "$arg" = "-b" ]; then
            background=true
        fi
    done

    if is_running; then
        warn "Data node is already running (PID: $(get_pid))"
        return 1
    fi

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

    kill -TERM "$pid" 2>/dev/null || true

    local count=0
    while [ $count -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        count=$((count + 1))
    done

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
    ensure_venv

    header "============================================================"
    header "  Pi Data-Node Status"
    header "============================================================"
    echo ""

    if is_running; then
        info "Service: RUNNING (PID: $(get_pid))"
    else
        warn "Service: STOPPED"
    fi

    if command -v systemctl >/dev/null 2>&1; then
        if systemctl is-enabled "$SERVICE_NAME" 2>/dev/null | grep -q "enabled"; then
            info "Systemd: ENABLED (starts on boot)"
        else
            warn "Systemd: DISABLED"
        fi
    fi

    echo ""
    "$VENV/bin/python" -m data_node --status 2>/dev/null || true
}

do_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        error "Log file not found: $LOG_FILE"
        echo "Start the node first with: $(basename "$0") start --bg"
        exit 1
    fi
}

do_selfcheck() {
    ensure_venv
    "$VENV/bin/python" -m data_node --selfcheck
}

do_enable() {
    if ! command -v systemctl >/dev/null 2>&1; then
        error "systemctl not found - this command is for Linux with systemd"
        exit 1
    fi

    local service_file="/etc/systemd/system/${SERVICE_NAME}.service"
    local script_path
    script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

    if [ ! -f "$service_file" ]; then
        info "Creating systemd service file..."
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
ExecStart=$script_path start --bg
ExecStop=$script_path stop
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

    if systemctl is-active "$SERVICE_NAME" 2>/dev/null | grep -q "active"; then
        info "Stopping systemd service..."
        sudo systemctl stop "$SERVICE_NAME"
    fi

    if systemctl is-enabled "$SERVICE_NAME" 2>/dev/null | grep -q "enabled"; then
        sudo systemctl disable "$SERVICE_NAME"
        info "Systemd service disabled"
    else
        warn "Systemd service was not enabled"
    fi

    do_stop
}

show_help() {
    cat << EOF
Pi Data-Node CLI

Usage: $(basename "$0") <command> [options]

Setup:
  setup          Run the interactive setup wizard (first time or reconfigure)

Service:
  start          Start data node (foreground with Rich UI)
  start --bg     Start in background (for production)
  stop           Stop the running data node
  restart        Stop and restart in background
  status         Show service and queue status
  logs           Tail the log file
  selfcheck      Run self-checks

Systemd (Linux/Pi):
  enable         Enable auto-start on boot
  disable        Disable auto-start and stop service

Examples:
  $(basename "$0") setup             # First-time setup
  $(basename "$0") start             # Interactive mode
  $(basename "$0") start --bg        # Production mode
  $(basename "$0") status            # Check what's happening

The script auto-bootstraps the Python virtual environment if needed.
EOF
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    setup|wizard)
        shift || true
        do_setup "$@"
        ;;
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
    logs|log)
        do_logs
        ;;
    selfcheck|check)
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
        show_help
        ;;
    *)
        # Unknown command - show help
        error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
