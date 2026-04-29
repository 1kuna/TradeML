"""User-facing CLI for dashboard and node operations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from trademl.dashboard.controller import (
    bootstrap_canonical_ledger,
    collect_dashboard_live_snapshot,
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
    write_deployed_version,
)
from trademl.env import load_dotenv
from trademl.fleet.autopilot import collect_fleet_health
from trademl.fleet.observability import build_fleet_observability, write_fleet_observability
from trademl.experiments import (
    backtest_experiment_survivors,
    compare_experiment,
    evaluate_experiment,
    experiment_status,
    launch_experiment,
    pause_experiment_supervisor,
    plan_experiment,
    propose_next_experiment_family,
    render_experiment_report,
    resume_experiment_supervisor,
    run_experiment_until_idle,
    stop_experiment_supervisor,
    supervise_experiment,
)
from trademl.fleet.launchd import install_research_launch_agent, launch_agent_status, unload_launch_agent
from trademl.research import (
    build_research_features,
    list_research_alerts,
    latest_research_program_summary,
    paper_account_smoke,
    pause_research_program,
    read_research_incumbent,
    read_research_program_state,
    research_health,
    resume_research_program,
    run_research_canary,
    start_research_program,
    steer_research_program,
    stop_research_program,
    submit_paper_orders,
    write_research_review_packet,
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
    train_status_parser.add_argument("--target", default=None)
    train_preflight_parser = train_subparsers.add_parser("preflight", help="Run the training preflight against NAS data.")
    train_preflight_parser.add_argument("--phase", type=int, default=1)
    train_preflight_parser.add_argument("--target", default=None)
    train_preflight_parser.add_argument("--model-suite", default=None)
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
    deployed_parser = node_subparsers.add_parser("write-deployed-version", help="Write deploy provenance for this worker.")
    deployed_parser.add_argument("--commit", default=None)
    deployed_parser.add_argument("--test-evidence", default=None)
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
    experiments_supervise = experiments_subparsers.add_parser("supervise", help="Run or spawn the bounded experiment supervisor.")
    experiments_supervise.add_argument("--spec", required=True)
    experiments_supervise.add_argument("--poll-seconds", type=int, default=None)
    experiments_supervise.add_argument("--detach", action="store_true")
    experiments_run_until_idle = experiments_subparsers.add_parser("run-until-idle", help="Keep launching experiment runs until the queue is drained.")
    experiments_run_until_idle.add_argument("--spec", required=True)
    experiments_run_until_idle.add_argument("--poll-seconds", type=int, default=30)
    experiments_status = experiments_subparsers.add_parser("status", help="Refresh experiment run status.")
    experiments_status.add_argument("--experiment", required=True)
    experiments_pause = experiments_subparsers.add_parser("pause", help="Pause a running experiment supervisor.")
    experiments_pause.add_argument("--experiment", required=True)
    experiments_resume = experiments_subparsers.add_parser("resume", help="Resume a paused experiment supervisor.")
    experiments_resume.add_argument("--experiment", required=True)
    experiments_stop = experiments_subparsers.add_parser("stop", help="Stop an experiment supervisor.")
    experiments_stop.add_argument("--experiment", required=True)
    experiments_evaluate = experiments_subparsers.add_parser("evaluate", help="Evaluate completed runs for an experiment.")
    experiments_evaluate.add_argument("--experiment", required=True)
    experiments_backtest = experiments_subparsers.add_parser("backtest-survivors", help="Run backtests for predictive survivors.")
    experiments_backtest.add_argument("--experiment", required=True)
    experiments_propose = experiments_subparsers.add_parser("propose-next", help="Generate the next bounded experiment family proposal.")
    experiments_propose.add_argument("--experiment", required=True)
    experiments_compare = experiments_subparsers.add_parser("compare", help="Compare completed experiment runs.")
    experiments_compare.add_argument("--experiment", required=True)
    experiments_report = experiments_subparsers.add_parser("report", help="Write experiment comparison reports.")
    experiments_report.add_argument("--experiment", required=True)

    research_parser = subparsers.add_parser("research", help="Run perpetual research programs.")
    research_parser.add_argument("--data-root", default=None)
    research_parser.add_argument("--local-state", default=None)
    research_parser.add_argument("--env-file", default=None)
    research_subparsers = research_parser.add_subparsers(dest="research_command", required=True)
    research_start = research_subparsers.add_parser("start", help="Start a research program supervisor.")
    research_start.add_argument("--program", required=True)
    research_start.add_argument("--poll-seconds", type=int, default=None)
    research_start.add_argument("--detach", action="store_true")
    research_canary = research_subparsers.add_parser("canary", help="Run one bounded autonomous research canary family.")
    research_canary.add_argument("--program", required=True)
    research_canary.add_argument("--poll-seconds", type=int, default=None)
    research_canary.add_argument("--detach", action="store_true")
    research_canary.add_argument("--feature-version", default=None)
    research_canary.add_argument("--label-horizon", type=int, default=None)
    research_build_features = research_subparsers.add_parser("build-features", help="Build PIT modeling feature and label artifacts.")
    research_build_features.add_argument("--program", default="configs/research/perpetual_macmini.yml")
    research_build_features.add_argument("--feature-version", default=None)
    research_build_features.add_argument("--report-date", default=None)
    research_paper_smoke = research_subparsers.add_parser("paper-smoke", help="Run a read-only Alpaca paper account smoke check.")
    research_paper_smoke.add_argument("--program", required=True)
    research_paper_submit = research_subparsers.add_parser("paper-submit", help="Submit generated Alpaca paper payloads with explicit guards.")
    research_paper_submit.add_argument("--program", required=True)
    research_paper_submit.add_argument("--payloads", required=True)
    research_launchd = research_subparsers.add_parser("install-launchd", help="Install a macOS LaunchAgent for a research program.")
    research_launchd.add_argument("--program", required=True)
    research_launchd.add_argument("--poll-seconds", type=int, default=None)
    research_launchd.add_argument("--label", default=None)
    research_launchd.add_argument("--plist-path", default=None)
    research_launchd.add_argument("--python-executable", default=None)
    research_launchd.add_argument("--load", action="store_true")
    research_launchd_status = research_subparsers.add_parser("launchd-status", help="Show macOS LaunchAgent status for a research program.")
    research_launchd_status.add_argument("--program", default=None)
    research_launchd_status.add_argument("--label", default=None)
    research_launchd_unload = research_subparsers.add_parser("unload-launchd", help="Unload a macOS LaunchAgent for a research program.")
    research_launchd_unload.add_argument("--program", default=None)
    research_launchd_unload.add_argument("--label", default=None)
    research_status = research_subparsers.add_parser("status", help="Show research program state.")
    research_status.add_argument("--program-id", required=True)
    research_pause = research_subparsers.add_parser("pause", help="Pause a research program.")
    research_pause.add_argument("--program-id", required=True)
    research_resume = research_subparsers.add_parser("resume", help="Resume a research program.")
    research_resume.add_argument("--program-id", required=True)
    research_stop = research_subparsers.add_parser("stop", help="Stop a research program.")
    research_stop.add_argument("--program-id", required=True)
    research_frontier = research_subparsers.add_parser("frontier", help="Show the current program frontier summary.")
    research_frontier.add_argument("--program-id", default=None)
    research_review = research_subparsers.add_parser("review-packet", help="Write a research review packet.")
    research_review.add_argument("--program-id", required=True)
    research_health_parser = research_subparsers.add_parser("health", help="Show hardened research health.")
    research_health_parser.add_argument("--program-id", required=True)
    research_incumbent = research_subparsers.add_parser("incumbent", help="Show the current research incumbent.")
    research_incumbent.add_argument("--program-id", required=True)
    research_alerts = research_subparsers.add_parser("alerts", help="Show recent research alerts.")
    research_alerts.add_argument("--program-id", required=True)
    research_alerts.add_argument("--limit", type=int, default=20)
    research_steer = research_subparsers.add_parser("steer", help="Persist manual steering for a research program.")
    research_steer.add_argument("--program-id", required=True)
    research_steer.add_argument("--prefer-architecture", action="append", default=None)
    research_steer.add_argument("--avoid-architecture", action="append", default=None)
    research_steer.add_argument("--prefer-data-family", action="append", default=None)
    research_steer.add_argument("--avoid-data-family", action="append", default=None)
    research_steer.add_argument("--freeze-phase", type=int, default=None)
    research_steer.add_argument("--force-pivot", action="store_true")
    research_steer.add_argument("--exploration-breadth", default=None)

    fleet_parser = subparsers.add_parser("fleet", help="Inspect fleet-level autopilot health.")
    fleet_parser.add_argument("--workspace-root", default=None)
    fleet_parser.add_argument("--config", default=None)
    fleet_parser.add_argument("--env-file", default=None)
    fleet_subparsers = fleet_parser.add_subparsers(dest="fleet_command", required=True)
    fleet_health = fleet_subparsers.add_parser("health", help="Show one current-state verdict for Pi, Mac, NAS, and research.")
    fleet_health.add_argument("--data-root", default=None)
    fleet_health.add_argument("--pi-host", default=None)
    fleet_health.add_argument("--pi-user", default="zach")
    fleet_health.add_argument("--pi-password-env", default="TRADEML_PI_PASSWORD")
    fleet_health.add_argument("--mac-host", default=None)
    fleet_health.add_argument("--mac-user", default="openclaw")
    fleet_health.add_argument("--mac-password-env", default="TRADEML_MAC_PASSWORD")
    fleet_health.add_argument("--heal", action="store_true")
    fleet_observability = fleet_subparsers.add_parser("observability", help="Write and print the fleet observability snapshot.")
    fleet_observability.add_argument("--data-root", default=None)
    fleet_observability.add_argument("--pi-host", default=None)
    fleet_observability.add_argument("--pi-user", default="zach")
    fleet_observability.add_argument("--pi-password-env", default="TRADEML_PI_PASSWORD")
    fleet_observability.add_argument("--mac-host", default=None)
    fleet_observability.add_argument("--mac-user", default="openclaw")
    fleet_observability.add_argument("--mac-password-env", default="TRADEML_MAC_PASSWORD")
    fleet_observability.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "dashboard":
        return _launch_dashboard(args)
    if args.command == "train":
        return _dispatch_train(args)
    if args.command == "experiments":
        return _dispatch_experiments(args)
    if args.command == "research":
        return _dispatch_research(args)
    if args.command == "fleet":
        return _dispatch_fleet(args)
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
    if args.node_command == "write-deployed-version":
        evidence: dict[str, object] = {}
        if args.test_evidence:
            try:
                evidence = json.loads(args.test_evidence)
            except json.JSONDecodeError:
                evidence = {"summary": args.test_evidence}
        print(
            json.dumps(
                write_deployed_version(
                    settings,
                    commit=args.commit,
                    test_evidence=evidence,
                ),
                indent=2,
                default=str,
            )
        )
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


def _dispatch_fleet(args: argparse.Namespace) -> int:
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    load_dotenv(settings.env_path)
    if args.fleet_command == "health":
        snapshot = _collect_fleet_local_snapshot(settings)
        data_root = Path(args.data_root).expanduser() if args.data_root else settings.nas_mount
        pi_host = args.pi_host or os.getenv("TRADEML_PI_HOST") or os.getenv("TRADEML_PI_TAILSCALE_HOST")
        mac_host = args.mac_host or os.getenv("TRADEML_MAC_HOST") or os.getenv("TRADEML_MAC_TAILSCALE_HOST")
        pi = (
            {"host": pi_host, "user": os.getenv("TRADEML_PI_USER") or args.pi_user, "password_env": args.pi_password_env}
            if pi_host
            else None
        )
        mac = (
            {"host": mac_host, "user": os.getenv("TRADEML_MAC_USER") or args.mac_user, "password_env": args.mac_password_env}
            if mac_host
            else None
        )
        payload = collect_fleet_health(local_snapshot=snapshot, data_root=data_root, pi=pi, mac=mac, heal=bool(args.heal))
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.fleet_command == "observability":
        snapshot = _collect_fleet_local_snapshot(settings)
        data_root = Path(args.data_root).expanduser() if args.data_root else settings.nas_mount
        pi_host = args.pi_host or os.getenv("TRADEML_PI_HOST") or os.getenv("TRADEML_PI_TAILSCALE_HOST")
        mac_host = args.mac_host or os.getenv("TRADEML_MAC_HOST") or os.getenv("TRADEML_MAC_TAILSCALE_HOST")
        pi = (
            {"host": pi_host, "user": os.getenv("TRADEML_PI_USER") or args.pi_user, "password_env": args.pi_password_env}
            if pi_host
            else None
        )
        mac = (
            {"host": mac_host, "user": os.getenv("TRADEML_MAC_USER") or args.mac_user, "password_env": args.mac_password_env}
            if mac_host
            else None
        )
        if pi or mac:
            payload = collect_fleet_health(
                local_snapshot=snapshot,
                data_root=data_root,
                pi=pi,
                mac=mac,
                heal=False,
            )["observability"]
            print(json.dumps(payload, indent=2, default=str))
            return 0
        remote = {}
        payload = build_fleet_observability(
            snapshot={**snapshot, "fleet_remote": remote},
            data_root=data_root,
            remote=remote,
            config=yaml.safe_load(settings.config_path.read_text(encoding="utf-8")) or {},
        )
        write_fleet_observability(data_root=data_root, payload=payload)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported fleet command: {args.fleet_command}")


def _collect_fleet_local_snapshot(settings) -> dict[str, object]:  # noqa: ANN001
    """Collect fleet command state without triggering remote training probes."""
    return collect_dashboard_live_snapshot(settings)


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
        print(json.dumps(training_preflight_status(settings, phase=args.phase, target=args.target, model_suite=args.model_suite), indent=2, default=str))
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
    if args.experiments_command == "supervise":
        payload = supervise_experiment(
            spec_path=Path(args.spec).expanduser(),
            poll_seconds=args.poll_seconds,
            detach=bool(args.detach),
            **common,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "run-until-idle":
        payload = run_experiment_until_idle(
            spec_path=Path(args.spec).expanduser(),
            poll_seconds=int(args.poll_seconds),
            **common,
        )
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
    if args.experiments_command == "pause":
        payload = pause_experiment_supervisor(local_state=local_state, experiment_id=args.experiment)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "resume":
        payload = resume_experiment_supervisor(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "stop":
        payload = stop_experiment_supervisor(
            local_state=local_state,
            experiment_id=args.experiment,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "evaluate":
        payload = evaluate_experiment(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "backtest-survivors":
        payload = backtest_experiment_survivors(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "propose-next":
        payload = propose_next_experiment_family(
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
        payload = compare_experiment(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.experiments_command == "report":
        payload = render_experiment_report(
            experiment_id=args.experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported experiments command: {args.experiments_command}")


def _dispatch_research(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root or os.getenv("TRADEML_DATA_ROOT") or os.getenv("NAS_MOUNT") or ".").expanduser()
    local_state = Path(args.local_state or os.getenv("TRADEML_TRAIN_STATE") or "~/.trademl-training").expanduser()
    env_path = Path(args.env_file).expanduser() if args.env_file else Path(".env")
    load_dotenv(env_path)
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
    if args.research_command == "start":
        payload = start_research_program(
            program_path=Path(args.program).expanduser(),
            poll_seconds=args.poll_seconds,
            detach=bool(args.detach),
            **common,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "canary":
        payload = run_research_canary(
            program_path=Path(args.program).expanduser(),
            poll_seconds=args.poll_seconds,
            detach=bool(args.detach),
            feature_version=args.feature_version,
            label_horizon=args.label_horizon,
            **common,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "build-features":
        payload = build_research_features(
            program_path=Path(args.program).expanduser(),
            data_root=data_root,
            feature_version=args.feature_version,
            report_date=args.report_date,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "paper-smoke":
        spec = yaml.safe_load(Path(args.program).expanduser().read_text(encoding="utf-8")) or {}
        payload = paper_account_smoke(policy=dict(spec.get("paper_policy") or {}))
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "paper-submit":
        spec = yaml.safe_load(Path(args.program).expanduser().read_text(encoding="utf-8")) or {}
        payload = submit_paper_orders(
            payloads_path=Path(args.payloads).expanduser(),
            policy=dict(spec.get("paper_policy") or {}),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "install-launchd":
        program_path = Path(args.program).expanduser()
        spec = yaml.safe_load(program_path.read_text(encoding="utf-8")) or {}
        program_id = str(spec.get("program_id") or program_path.stem)
        payload = install_research_launch_agent(
            label=str(args.label or f"com.trademl.research.{program_id}"),
            python_executable=str(args.python_executable or sys.executable),
            repo_root=repo_root,
            program_path=program_path,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            poll_seconds=int(args.poll_seconds or spec.get("poll_seconds") or 60),
            plist_path=Path(args.plist_path).expanduser() if args.plist_path else None,
            load=bool(args.load),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command in {"launchd-status", "unload-launchd"}:
        label = _research_launchd_label(program_path=args.program, label=args.label)
        payload = launch_agent_status(label) if args.research_command == "launchd-status" else unload_launch_agent(label)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "status":
        payload = read_research_program_state(local_state=local_state, program_id=args.program_id)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "pause":
        payload = pause_research_program(local_state=local_state, program_id=args.program_id)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "resume":
        payload = resume_research_program(
            program_id=args.program_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "stop":
        payload = stop_research_program(
            local_state=local_state,
            program_id=args.program_id,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "frontier":
        payload = (
            read_research_program_state(local_state=local_state, program_id=args.program_id)
            if args.program_id
            else latest_research_program_summary(local_state=local_state)
        )
        print(json.dumps(payload.get("frontier", payload), indent=2, default=str))
        return 0
    if args.research_command == "review-packet":
        payload = write_research_review_packet(
            program_id=args.program_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "health":
        payload = research_health(
            program_id=args.program_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "incumbent":
        payload = read_research_incumbent(local_state=local_state, program_id=args.program_id)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "alerts":
        payload = list_research_alerts(local_state=local_state, program_id=args.program_id, limit=int(args.limit))
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "steer":
        payload = steer_research_program(
            local_state=local_state,
            program_id=args.program_id,
            prefer_architecture_families=args.prefer_architecture,
            avoid_architecture_families=args.avoid_architecture,
            prefer_data_families=args.prefer_data_family,
            avoid_data_families=args.avoid_data_family,
            freeze_phase=args.freeze_phase,
            force_pivot=bool(args.force_pivot),
            exploration_breadth=args.exploration_breadth,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    raise SystemExit(f"unsupported research command: {args.research_command}")


def _research_launchd_label(*, program_path: str | None, label: str | None) -> str:
    """Resolve the launchd label for a research program."""
    if label:
        return str(label)
    if not program_path:
        raise SystemExit("--program or --label is required")
    resolved_program_path = Path(program_path).expanduser()
    spec = yaml.safe_load(resolved_program_path.read_text(encoding="utf-8")) or {}
    program_id = str(spec.get("program_id") or resolved_program_path.stem)
    return f"com.trademl.research.{program_id}"


if __name__ == "__main__":
    raise SystemExit(main())
