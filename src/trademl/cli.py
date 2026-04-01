"""User-facing CLI for dashboard and node operations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from trademl.dashboard.controller import (
    collect_dashboard_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    rebuild_cluster_state,
    resolve_node_settings,
    restart_node,
    rotate_cluster_passphrase,
    start_node,
    stop_node,
    update_cluster_secrets,
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
    join_parser = node_subparsers.add_parser("join-cluster", help="Bootstrap or join the NAS-backed worker cluster.")
    join_parser.add_argument("--passphrase", default=None)
    rebuild_parser = node_subparsers.add_parser("rebuild-state", help="Rebuild local disposable state from NAS truth.")
    rebuild_parser.add_argument("--passphrase", default=None)
    node_subparsers.add_parser("leave-cluster", help="Release leases and mark this worker inactive.")
    install_parser = node_subparsers.add_parser("install-service", help="Install the Linux systemd unit.")
    install_parser.add_argument("--service-path", default=None)
    rotate_parser = node_subparsers.add_parser("rotate-passphrase", help="Rotate the cluster secret bundle passphrase.")
    rotate_parser.add_argument("--old-passphrase", default=None)
    rotate_parser.add_argument("--new-passphrase", default=None)
    secrets_parser = node_subparsers.add_parser("update-secrets", help="Update NAS-backed encrypted shared secrets.")
    secrets_parser.add_argument("--passphrase", default=None)
    secrets_parser.add_argument("--set", action="append", default=[])
    release_parser = node_subparsers.add_parser("force-release", help="Force-release a lease.")
    release_parser.add_argument("lease_id")

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
    if args.node_command == "join-cluster":
        print(json.dumps(join_cluster(settings, passphrase=args.passphrase), indent=2, default=str))
        return 0
    if args.node_command == "rebuild-state":
        print(json.dumps(rebuild_cluster_state(settings, passphrase=args.passphrase), indent=2, default=str))
        return 0
    if args.node_command == "leave-cluster":
        print(json.dumps(leave_cluster(settings), indent=2, default=str))
        return 0
    if args.node_command == "install-service":
        print(json.dumps(install_service(settings, service_path=args.service_path), indent=2, default=str))
        return 0
    if args.node_command == "rotate-passphrase":
        old_passphrase = args.old_passphrase or os.getenv("TRADEML_CLUSTER_PASSPHRASE")
        new_passphrase = args.new_passphrase or os.getenv("TRADEML_CLUSTER_NEW_PASSPHRASE")
        if not old_passphrase or not new_passphrase:
            raise SystemExit("old/new cluster passphrases must be provided")
        print(json.dumps(rotate_cluster_passphrase(settings, old_passphrase=old_passphrase, new_passphrase=new_passphrase), indent=2, default=str))
        return 0
    if args.node_command == "update-secrets":
        passphrase = args.passphrase or os.getenv("TRADEML_CLUSTER_PASSPHRASE")
        if not passphrase:
            raise SystemExit("cluster passphrase required")
        updates: dict[str, str] = {}
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"invalid secret assignment: {item}")
            key, value = item.split("=", 1)
            updates[key] = value
        print(json.dumps(update_cluster_secrets(settings, passphrase=passphrase, updates=updates), indent=2, default=str))
        return 0
    if args.node_command == "force-release":
        print(json.dumps({"released": force_release_lease(settings, args.lease_id), "lease_id": args.lease_id}, indent=2, default=str))
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
