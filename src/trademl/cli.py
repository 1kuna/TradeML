"""User-facing CLI for dashboard and node operations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trademl.dashboard.controller import (
    collect_dashboard_snapshot,
    resolve_node_settings,
    restart_node,
    start_node,
    stop_node,
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch TradeML CLI commands."""
    parser = argparse.ArgumentParser(prog="trademl", description="TradeML operator CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit operator dashboard.")
    dashboard_parser.add_argument("--workspace-root", default=None)
    dashboard_parser.add_argument("--config", default=None)
    dashboard_parser.add_argument("--env-file", default=None)
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument("--no-browser", action="store_true")

    node_parser = subparsers.add_parser("node", help="Control the data-node service.")
    node_parser.add_argument("--workspace-root", default=None)
    node_parser.add_argument("--config", default=None)
    node_parser.add_argument("--env-file", default=None)
    node_subparsers = node_parser.add_subparsers(dest="node_command", required=True)
    node_subparsers.add_parser("status", help="Print current node status as JSON.")
    node_subparsers.add_parser("start", help="Start the node in the background.")
    node_subparsers.add_parser("stop", help="Stop the node if it is running.")
    node_subparsers.add_parser("restart", help="Restart the node.")

    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return _launch_dashboard(args)
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    if args.node_command == "status":
        print(json.dumps(collect_dashboard_snapshot(settings), indent=2, default=str))
        return 0
    if args.node_command == "start":
        print(json.dumps(start_node(settings), indent=2, default=str))
        return 0
    if args.node_command == "stop":
        print(json.dumps(stop_node(settings), indent=2, default=str))
        return 0
    if args.node_command == "restart":
        print(json.dumps(restart_node(settings), indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported node command: {args.node_command}")


def _launch_dashboard(args: argparse.Namespace) -> int:
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    if args.no_browser:
        command.extend(["--server.headless", "true"])
    command.extend(
        [
            "--",
            "--workspace-root",
            str(settings.workspace_root),
            "--config",
            str(settings.config_path),
            "--env-file",
            str(settings.env_path),
        ]
    )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
