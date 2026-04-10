"""User-facing CLI for dashboard and node operations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from trademl.dashboard.controller import (
    bootstrap_canonical_ledger,
    collect_dashboard_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    replan_coverage,
    repair_canonical_backlog,
    rebuild_cluster_state,
    resolve_node_settings,
    restart_node,
    run_vendor_audit,
    rotate_cluster_passphrase,
    start_node,
    stop_node,
    reset_worker,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
)
from trademl.data_node.training_control import (
    launch_training_process,
    read_training_runtime,
    shared_training_runtime_path,
    training_preflight,
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch TradeML CLI commands."""
    parser = argparse.ArgumentParser(prog="trademl", description="TradeML operator CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the operator dashboard server.")
    dashboard_parser.add_argument("--workspace-root", default=None)
    dashboard_parser.add_argument("--config", default=None)
    dashboard_parser.add_argument("--env-file", default=None)
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument("--no-browser", action="store_true")

    train_parser = subparsers.add_parser("train", help="Launch or inspect off-node NAS-backed training.")
    train_parser.add_argument("--data-root", default=None)
    train_parser.add_argument("--local-state", default=None)
    train_parser.add_argument("--env-file", default=None)
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_status_parser = train_subparsers.add_parser("status", help="Show local and NAS-visible training runtime.")
    train_status_parser.add_argument("--phase", type=int, default=1)
    train_preflight_parser = train_subparsers.add_parser("preflight", help="Run the training preflight against NAS data.")
    train_preflight_parser.add_argument("--phase", type=int, default=1)
    train_start_parser = train_subparsers.add_parser("start", help="Start a detached DGX/workstation training run.")
    train_start_parser.add_argument("--phase", type=int, default=1)
    train_start_parser.add_argument("--report-date", default=None)
    train_start_parser.add_argument("--python-executable", default=sys.executable)

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
    reset_parser = node_subparsers.add_parser("reset", help="Wipe local disposable worker state and rebuild from NAS.")
    reset_parser.add_argument("--passphrase", default=None)
    node_subparsers.add_parser("update", help="Update the local worker installation.")
    node_subparsers.add_parser("uninstall", help="Remove local worker artifacts from this machine.")
    node_subparsers.add_parser("run-audit", help="Run live vendor capability canaries and persist the report.")
    node_subparsers.add_parser("replan-coverage", help="Materialize the current auxiliary coverage plan.")
    node_subparsers.add_parser("bootstrap-canonical-ledger", help="Seed the durable canonical ledger from raw partitions.")
    repair_parser = node_subparsers.add_parser("repair-canonical", help="Repair stale partial canonical backlog state.")
    repair_parser.add_argument("--date", default=None)
    repair_parser.add_argument("--start-date", default=None)
    repair_parser.add_argument("--end-date", default=None)
    repair_parser.add_argument("--symbol", default=None)
    repair_parser.add_argument("--verify-only", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return _launch_dashboard(args)
    if args.command == "train":
        return _dispatch_train(args)
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
    if args.node_command == "reset":
        print(json.dumps(reset_worker(settings, passphrase=args.passphrase), indent=2, default=str))
        return 0
    if args.node_command == "update":
        print(json.dumps(update_worker(settings), indent=2, default=str))
        return 0
    if args.node_command == "uninstall":
        print(json.dumps(uninstall_worker(settings), indent=2, default=str))
        return 0
    if args.node_command == "run-audit":
        print(json.dumps(run_vendor_audit(settings), indent=2, default=str))
        return 0
    if args.node_command == "replan-coverage":
        print(json.dumps(replan_coverage(settings), indent=2, default=str))
        return 0
    if args.node_command == "bootstrap-canonical-ledger":
        print(json.dumps(bootstrap_canonical_ledger(settings), indent=2, default=str))
        return 0
    if args.node_command == "repair-canonical":
        repair_kwargs = {"trading_date": args.date}
        if args.start_date is not None:
            repair_kwargs["start_date"] = args.start_date
        if args.end_date is not None:
            repair_kwargs["end_date"] = args.end_date
        if args.symbol is not None:
            repair_kwargs["symbol"] = args.symbol
        if args.verify_only:
            repair_kwargs["verify_only"] = True
        print(
            json.dumps(
                repair_canonical_backlog(settings, **repair_kwargs),
                indent=2,
                default=str,
            )
        )
        return 0
    raise SystemExit(f"unsupported node command: {args.node_command}")


def _launch_dashboard(args: argparse.Namespace) -> int:
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    command = [
        sys.executable,
        "-m",
        "trademl.dashboard.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.no_browser:
        command.append("--no-browser")
    command.extend(
        [
            "--workspace-root",
            str(settings.workspace_root),
            "--config",
            str(settings.config_path),
            "--env-file",
            str(settings.env_path),
        ]
    )
    return subprocess.run(command, check=False).returncode


def _dispatch_train(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root or os.getenv("TRADEML_DATA_ROOT") or os.getenv("NAS_MOUNT") or ".").expanduser()
    local_state = Path(args.local_state or os.getenv("TRADEML_TRAIN_STATE") or "~/.trademl-training").expanduser()
    env_path = Path(args.env_file).expanduser() if args.env_file else Path(".env")
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "equities_xs.yml"
    if args.train_command == "status":
        payload = {
            "local": read_training_runtime(local_state=local_state, phase=args.phase),
            "shared": read_training_runtime(path=shared_training_runtime_path(data_root=data_root, phase=args.phase)),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.train_command == "preflight":
        print(json.dumps(training_preflight(data_root=data_root, config_path=config_path), indent=2, default=str))
        return 0
    if args.train_command == "start":
        payload = launch_training_process(
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            phase=args.phase,
            model_suite="ridge_only" if args.phase == 1 else "full",
            python_executable=args.python_executable,
            report_date=args.report_date,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported train command: {args.train_command}")


if __name__ == "__main__":
    raise SystemExit(main())
