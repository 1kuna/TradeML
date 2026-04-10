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
    repair_status,
    run_vendor_audit,
    rotate_cluster_passphrase,
    start_node,
    start_training_run,
    stop_node,
    stop_training_run,
    lane_health,
    show_leases,
    training_preflight_status,
    training_runtime_logs,
    training_runtime_status,
    reset_worker,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
    verify_recent_canonical_dates,
)
from trademl.data_node.training_control import (
    resolve_training_target,
)
from trademl.experiments import compare_experiment, experiment_status, launch_experiment, plan_experiment, render_experiment_report


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
    train_status_parser.add_argument("--target", default=None)
    train_preflight_parser = train_subparsers.add_parser("preflight", help="Run the training preflight against NAS data.")
    train_preflight_parser.add_argument("--phase", type=int, default=1)
    train_preflight_parser.add_argument("--target", default=None)
    train_start_parser = train_subparsers.add_parser("start", help="Start a detached DGX/workstation training run.")
    train_start_parser.add_argument("--phase", type=int, default=1)
    train_start_parser.add_argument("--report-date", default=None)
    train_start_parser.add_argument("--python-executable", default=sys.executable)
    train_start_parser.add_argument("--target", default=None)
    train_stop_parser = train_subparsers.add_parser("stop", help="Stop a detached training run.")
    train_stop_parser.add_argument("--phase", type=int, default=1)
    train_stop_parser.add_argument("--target", default=None)
    train_logs_parser = train_subparsers.add_parser("logs", help="Tail detached training logs.")
    train_logs_parser.add_argument("--phase", type=int, default=1)
    train_logs_parser.add_argument("--tail", type=int, default=50)
    train_logs_parser.add_argument("--target", default=None)

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
    verify_recent_parser = node_subparsers.add_parser("verify-recent", help="Verify the most recently touched canonical dates.")
    verify_recent_parser.add_argument("--days", type=int, default=7)
    verify_recent_parser.add_argument("--dataset", default="equities_eod")
    verify_recent_parser.add_argument("--verify-only", action="store_true")
    node_subparsers.add_parser("repair-status", help="Show current repair lane health.")
    lane_health_parser = node_subparsers.add_parser("lane-health", help="Show current vendor lane health.")
    lane_health_parser.add_argument("--dataset", default="equities_eod")
    show_leases_parser = node_subparsers.add_parser("show-leases", help="Show current leased planner work.")
    show_leases_parser.add_argument("--family", default=None, choices=["canonical_bars", "canonical_repair"])

    experiments_parser = subparsers.add_parser("experiments", help="Plan and launch experiment matrices.")
    experiments_parser.add_argument("--data-root", default=None)
    experiments_parser.add_argument("--local-state", default=None)
    experiments_parser.add_argument("--env-file", default=None)
    experiments_subparsers = experiments_parser.add_subparsers(dest="experiments_command", required=True)
    experiments_plan = experiments_subparsers.add_parser("plan", help="Plan an experiment matrix.")
    experiments_plan.add_argument("--spec", required=True)
    experiments_launch = experiments_subparsers.add_parser("launch", help="Launch pending runs for an experiment.")
    experiments_launch.add_argument("--spec", required=True)
    experiments_status = experiments_subparsers.add_parser("status", help="Refresh experiment run status.")
    experiments_status.add_argument("--experiment", required=True)
    experiments_compare = experiments_subparsers.add_parser("compare", help="Compare completed experiment runs.")
    experiments_compare.add_argument("--experiment", required=True)
    experiments_report = experiments_subparsers.add_parser("report", help="Write experiment comparison reports.")
    experiments_report.add_argument("--experiment", required=True)

    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return _launch_dashboard(args)
    if args.command == "train":
        return _dispatch_train(args)
    if args.command == "experiments":
        return _dispatch_experiments(args)
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
    if args.node_command == "verify-recent":
        print(
            json.dumps(
                verify_recent_canonical_dates(
                    settings,
                    days=args.days,
                    dataset=args.dataset,
                    verify_only=args.verify_only,
                ),
                indent=2,
                default=str,
            )
        )
        return 0
    if args.node_command == "repair-status":
        print(json.dumps(repair_status(settings), indent=2, default=str))
        return 0
    if args.node_command == "lane-health":
        print(json.dumps(lane_health(settings, dataset=args.dataset), indent=2, default=str))
        return 0
    if args.node_command == "show-leases":
        print(json.dumps(show_leases(settings, family=args.family), indent=2, default=str))
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
    repo_root = Path(__file__).resolve().parents[2]
    targets_config_path = repo_root / "configs" / "node.yml"
    settings = resolve_node_settings(config_path=targets_config_path)
    settings.nas_mount = data_root
    settings.local_state = local_state
    if args.env_file:
        settings.env_path = Path(args.env_file).expanduser()
    if args.train_command == "status":
        payload = training_runtime_status(settings, phase=args.phase, target=args.target)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.train_command == "preflight":
        print(json.dumps(training_preflight_status(settings, phase=args.phase, target=args.target), indent=2, default=str))
        return 0
    if args.train_command == "start":
        payload = start_training_run(settings, phase=args.phase, report_date=args.report_date, target=args.target)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.train_command == "stop":
        payload = stop_training_run(settings, phase=args.phase, target=args.target)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.train_command == "logs":
        payload = training_runtime_logs(settings, phase=args.phase, target=args.target, tail_lines=args.tail)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported train command: {args.train_command}")


def _dispatch_experiments(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root or os.getenv("TRADEML_DATA_ROOT") or os.getenv("NAS_MOUNT") or ".").expanduser()
    local_state = Path(args.local_state or os.getenv("TRADEML_TRAIN_STATE") or "~/.trademl-training").expanduser()
    env_path = Path(args.env_file).expanduser() if args.env_file else Path(".env")
    repo_root = Path(__file__).resolve().parents[2]
    targets_config_path = repo_root / "configs" / "node.yml"
    common = {
        "repo_root": repo_root,
        "data_root": data_root,
        "local_state": local_state,
        "env_path": env_path,
        "targets_config_path": targets_config_path,
        "python_executable": sys.executable,
    }
    if args.experiments_command == "plan":
        payload = plan_experiment(spec_path=Path(args.spec).expanduser(), **common)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "launch":
        payload = launch_experiment(spec_path=Path(args.spec).expanduser(), **common)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "status":
        payload = experiment_status(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "compare":
        payload = compare_experiment(experiment_id=args.experiment, local_state=local_state)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "report":
        payload = render_experiment_report(experiment_id=args.experiment, local_state=local_state)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported experiments command: {args.experiments_command}")


if __name__ == "__main__":
    raise SystemExit(main())
