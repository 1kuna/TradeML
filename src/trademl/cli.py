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
from trademl.data_node.compaction import compact_archive_partitions
from trademl.data_node.db import DataNodeDB
from trademl.env import load_dotenv
from trademl.fleet.audit import run_fleet_audit
from trademl.fleet.autopilot import collect_fleet_health
from trademl.fleet.data_quality import run_data_quality_audit
from trademl.fleet.observability import build_fleet_observability, write_fleet_observability
from trademl.fleet.watchdog import run_fleet_watchdog_once
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
from trademl.events.form4_candidates import run_form4_candidate_curation
from trademl.events.form4_event_study import run_form4_event_study
from trademl.events.form4_fixture_gate import run_form4_fixture_gate_from_env
from trademl.events.form4_ingest import run_form4_ingest_from_env
from trademl.events.form4_labels import run_form4_label_curation
from trademl.events.form4_market_backfill import run_form4_market_backfill_from_env
from trademl.events.form4_rework import run_form4_rework_study
from trademl.events.sec8k import (
    run_sec8k_candidate_curation,
    run_sec8k_event_study,
    run_sec8k_research_decision,
)
from trademl.events.sec8k_ingest import run_sec8k_ingest_from_env
from trademl.events.sec8k_market_backfill import run_sec8k_market_backfill_from_env
from trademl.events.sec8k_coverage import (
    run_sec8k_coverage_audit,
    run_sec8k_coverage_expand,
    run_sec_event_semantic_coverage_gate,
)
from trademl.events.sec8k_semantic import (
    run_sec_event_semantic_classification,
    run_sec_event_semantic_labelability_audit,
    run_sec_event_semantic_scaled_gate,
    run_sec_event_semantic_study,
)
from trademl.events.semantic_classifier import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_SEC_EVENT_MODEL,
    run_sec_event_semantic_fixture_gate,
)
from trademl.fleet.launchd import install_research_launch_agent, launch_agent_status, unload_launch_agent
from trademl.research import (
    build_research_features,
    list_research_alerts,
    latest_research_program_summary,
    pause_research_program,
    read_research_incumbent,
    read_research_program_state,
    research_health,
    reload_research_program,
    resume_research_program,
    run_and_persist_paper_account_smoke,
    run_feature_version_canary_batch,
    run_research_canary,
    start_research_program,
    steer_research_program,
    stop_research_program,
    submit_paper_orders,
    write_research_feature_source_contract,
    write_research_review_packet,
)
from trademl.research_audit import run_research_progression_audit


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
    compact_parser = node_subparsers.add_parser("compact-archives", help="Compact bounded raw/archive parquet partitions.")
    compact_parser.add_argument("--dataset", action="append", default=None)
    compact_parser.add_argument("--max-partitions", type=int, default=10)
    compact_parser.add_argument("--dry-run", action="store_true")
    node_subparsers.add_parser("repair-status", help="Show current repair lane health.")
    lane_health_parser = node_subparsers.add_parser("lane-health", help="Show current vendor lane health.")
    lane_health_parser.add_argument("--dataset", default="equities_eod")
    controller_status_parser = node_subparsers.add_parser("controller-status", help="Show recent saturation-controller decisions.")
    controller_status_parser.add_argument("--minutes", type=int, default=15)
    controller_status_parser.add_argument("--json", action="store_true")
    data_quality_parser = node_subparsers.add_parser("data-quality", help="Run and persist data-quality checks for high-value datasets.")
    data_quality_parser.add_argument("--dataset", action="append", default=None)
    data_quality_parser.add_argument("--json", action="store_true")
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
    research_source_contract = research_subparsers.add_parser("source-contract", help="Write feature source contract paths.")
    research_source_contract.add_argument("--source-root", default=None)
    research_source_contract.add_argument(
        "--dataset-path",
        action="append",
        default=None,
        metavar="DATASET=PATH",
        help="Override one dataset source path; repeat for multiple paths/datasets.",
    )
    research_form4_fixture_gate = research_subparsers.add_parser(
        "form4-fixture-gate",
        help="Run the SEC Form 4 retrieval/parser fixture gate.",
    )
    research_form4_fixture_gate.add_argument("--limit", type=int, default=None)
    research_form4_fixture_gate.add_argument("--user-agent", default=None)
    research_form4_ingest = research_subparsers.add_parser(
        "form4-ingest",
        help="Fetch SEC Form 4 manifests, raw filings, parser outputs, and candidates.",
    )
    research_form4_ingest.add_argument("--start-date", required=True)
    research_form4_ingest.add_argument("--end-date", required=True)
    research_form4_ingest.add_argument("--limit", type=int, default=None)
    research_form4_ingest.add_argument("--user-agent", default=None)
    research_form4_ingest.add_argument("--max-retrieval-attempts", type=int, default=6)
    research_form4_ingest.add_argument("--rate-limit-pause-seconds", type=float, default=60.0)
    research_form4_ingest.add_argument(
        "--no-cache",
        action="store_true",
        help="Refetch raw Form 4 filings instead of reusing existing raw artifacts.",
    )
    research_subparsers.add_parser(
        "form4-candidates",
        help="Build SEC Form 4 candidate events from curated parser outputs.",
    )
    research_form4_market_backfill = research_subparsers.add_parser(
        "form4-market-backfill",
        help="Fetch bounded market-data slices needed by current Form 4 labels.",
    )
    research_form4_market_backfill.add_argument(
        "--horizon", action="append", type=int, default=None
    )
    research_form4_market_backfill.add_argument(
        "--round-trip-cost-bps", type=float, default=50.0
    )
    research_form4_market_backfill.add_argument("--limit-events", type=int, default=None)
    research_form4_market_backfill.add_argument(
        "--no-controls",
        action="store_true",
        help="Do not include Form 4 negative-control candidates in market backfill.",
    )
    research_form4_market_backfill.add_argument("--max-fetch-attempts", type=int, default=6)
    research_form4_market_backfill.add_argument(
        "--rate-limit-pause-seconds",
        type=float,
        default=60.0,
    )
    research_form4_market_backfill.add_argument(
        "--daily-symbol-batch-size",
        type=int,
        default=100,
    )
    research_form4_labels = research_subparsers.add_parser(
        "form4-labels",
        help="Build PIT Form 4 event labels from candidate events and market bars.",
    )
    research_form4_labels.add_argument("--horizon", action="append", type=int, default=None)
    research_form4_labels.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_form4_labels.add_argument(
        "--market-data-root",
        action="append",
        default=None,
        help="Additional root to search for market-data partitions; repeatable.",
    )
    research_form4_labels.add_argument(
        "--source-contract",
        default=None,
        help="Explicit feature source contract JSON path for market-data source mapping.",
    )
    research_form4_event_study = research_subparsers.add_parser(
        "form4-event-study",
        help="Write Form 4 event-study, controls, and decision packet.",
    )
    research_form4_event_study.add_argument("--primary-horizon", type=int, default=5)
    research_form4_event_study.add_argument(
        "--horizon", action="append", type=int, default=None
    )
    research_form4_event_study.add_argument(
        "--round-trip-cost-bps", type=float, default=50.0
    )
    research_form4_event_study.add_argument(
        "--min-historical-sample", type=int, default=300
    )
    research_form4_event_study.add_argument(
        "--market-data-root",
        action="append",
        default=None,
        help="Additional root to search for market-data partitions; repeatable.",
    )
    research_form4_event_study.add_argument(
        "--source-contract",
        default=None,
        help="Explicit feature source contract JSON path for market-data source mapping.",
    )
    research_subparsers.add_parser(
        "form4-rework-study",
        help="Run the bounded Form 4 diagnostic rework gate from existing artifacts.",
    )
    research_sec8k_ingest = research_subparsers.add_parser(
        "sec8k-ingest",
        help="Fetch SEC 8-K full-index rows, complete texts, and item candidates.",
    )
    research_sec8k_ingest.add_argument("--start-date", required=True)
    research_sec8k_ingest.add_argument("--end-date", required=True)
    research_sec8k_ingest.add_argument("--limit", type=int, default=None)
    research_sec8k_ingest.add_argument("--user-agent", default=None)
    research_sec8k_ingest.add_argument("--max-retrieval-attempts", type=int, default=6)
    research_sec8k_ingest.add_argument("--rate-limit-pause-seconds", type=float, default=60.0)
    research_sec8k_ingest.add_argument(
        "--no-cache",
        action="store_true",
        help="Refetch raw SEC 8-K complete text instead of reusing existing raw artifacts.",
    )
    research_sec8k_candidates = research_subparsers.add_parser(
        "sec8k-candidates",
        help="Build deterministic SEC 8-K item candidates from existing filing artifacts.",
    )
    research_sec8k_candidates.add_argument(
        "--filings-path",
        default=None,
        help="Optional parquet path for SEC filing-index rows.",
    )
    research_sec8k_market_backfill = research_subparsers.add_parser(
        "sec8k-market-backfill",
        help="Fetch bounded market-data slices needed by current SEC 8-K labels.",
    )
    research_sec8k_market_backfill.add_argument(
        "--horizon", action="append", type=int, default=None
    )
    research_sec8k_market_backfill.add_argument(
        "--round-trip-cost-bps", type=float, default=50.0
    )
    research_sec8k_market_backfill.add_argument("--limit-events", type=int, default=None)
    research_sec8k_market_backfill.add_argument(
        "--no-timestamp-placebo",
        action="store_true",
        help="Do not include SEC 8-K timestamp-placebo candidates in market backfill.",
    )
    research_sec8k_market_backfill.add_argument("--max-fetch-attempts", type=int, default=6)
    research_sec8k_market_backfill.add_argument(
        "--rate-limit-pause-seconds",
        type=float,
        default=60.0,
    )
    research_sec8k_market_backfill.add_argument(
        "--daily-symbol-batch-size",
        type=int,
        default=100,
    )
    research_sec8k_market_backfill.add_argument(
        "--candidate-source",
        choices=("sec8k_item_events", "sec_event_semantic_candidates"),
        default="sec8k_item_events",
    )
    research_sec8k_market_backfill.add_argument(
        "--target-item",
        action="append",
        default=None,
        help="SEC item number to include in market backfill; repeatable.",
    )
    research_sec8k_market_backfill.add_argument("--accepted-from", default=None)
    research_sec8k_market_backfill.add_argument("--accepted-to", default=None)
    research_sec8k_event_study = research_subparsers.add_parser(
        "sec8k-event-study",
        help="Write SEC 8-K item-event labels, controls, and decision packet.",
    )
    research_sec8k_event_study.add_argument("--primary-horizon", type=int, default=5)
    research_sec8k_event_study.add_argument("--horizon", action="append", type=int, default=None)
    research_sec8k_event_study.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_sec8k_event_study.add_argument(
        "--market-data-root",
        action="append",
        default=None,
        help="Additional root to search for market-data partitions; repeatable.",
    )
    research_sec8k_event_study.add_argument(
        "--source-contract",
        default=None,
        help="Explicit feature source contract JSON path for market-data source mapping.",
    )
    research_sec8k_decision = research_subparsers.add_parser(
        "sec8k-decision",
        help="Write the continue/kill decision for broad deterministic SEC 8-K item events.",
    )
    research_sec8k_decision.add_argument("--min-labeled-events", type=int, default=300)
    research_sec8k_decision.add_argument("--min-family-events", type=int, default=100)
    research_sec8k_decision.add_argument("--min-mean-abret", type=float, default=0.005)
    research_sec8k_decision.add_argument(
        "--min-control-separation",
        type=float,
        default=0.0075,
    )
    research_sec8k_decision.add_argument(
        "--max-top5-abs-contribution",
        type=float,
        default=0.35,
    )
    research_sec8k_coverage_audit = research_subparsers.add_parser(
        "sec8k-coverage-audit",
        help="Audit SEC 8-K month/raw/candidate/labelability coverage before LLM spend.",
    )
    research_sec8k_coverage_audit.add_argument("--start-date", default="2024-01-01")
    research_sec8k_coverage_audit.add_argument("--end-date", default="2025-12-31")
    research_sec8k_coverage_audit.add_argument("--target-item", action="append", default=None)
    research_sec8k_coverage_audit.add_argument(
        "--fallback-target-item",
        action="append",
        default=None,
    )
    research_sec8k_coverage_audit.add_argument("--horizon", action="append", type=int, default=None)
    research_sec8k_coverage_audit.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_sec8k_coverage_expand = research_subparsers.add_parser(
        "sec8k-coverage-expand",
        help="Ingest missing SEC 8-K months and rebuild global item candidates.",
    )
    research_sec8k_coverage_expand.add_argument("--start-date", default="2024-01-01")
    research_sec8k_coverage_expand.add_argument("--end-date", default="2025-12-31")
    research_sec8k_coverage_expand.add_argument("--target-item", action="append", default=None)
    research_sec8k_coverage_expand.add_argument(
        "--fallback-target-item",
        action="append",
        default=None,
    )
    research_sec8k_coverage_expand.add_argument("--horizon", action="append", type=int, default=None)
    research_sec8k_coverage_expand.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_sec8k_coverage_expand.add_argument("--limit-per-month", type=int, default=None)
    research_sec8k_coverage_expand.add_argument("--user-agent", default=None)
    research_sec8k_coverage_expand.add_argument("--max-retrieval-attempts", type=int, default=6)
    research_sec8k_coverage_expand.add_argument(
        "--rate-limit-pause-seconds",
        type=float,
        default=60.0,
    )
    research_sec8k_coverage_expand.add_argument("--no-cache", action="store_true")
    research_sec8k_coverage_expand.add_argument("--no-rebuild-candidates", action="store_true")
    research_semantic_gate = research_subparsers.add_parser(
        "sec-event-semantic-gate",
        help="Run the LLM semantic classifier fixture gate for SEC/public-event excerpts.",
    )
    research_semantic_gate.add_argument("--model", default=DEFAULT_SEC_EVENT_MODEL)
    research_semantic_gate.add_argument("--base-url", default=DEFAULT_LMSTUDIO_BASE_URL)
    research_semantic_gate.add_argument("--timeout-seconds", type=float, default=60.0)
    research_semantic_gate.add_argument(
        "--response-format-mode",
        choices=("json_schema", "prompt_json"),
        default="json_schema",
    )
    research_semantic_gate.add_argument("--batch-size", type=int, default=4)
    research_semantic_gate.add_argument("--limit", type=int, default=None)
    research_semantic_classify = research_subparsers.add_parser(
        "sec-event-semantic-classify",
        help="Classify real SEC 8-K snippets with the semantic event model.",
    )
    research_semantic_classify.add_argument("--model", default=DEFAULT_SEC_EVENT_MODEL)
    research_semantic_classify.add_argument("--base-url", default=DEFAULT_LMSTUDIO_BASE_URL)
    research_semantic_classify.add_argument("--timeout-seconds", type=float, default=180.0)
    research_semantic_classify.add_argument(
        "--response-format-mode",
        choices=("json_schema", "prompt_json"),
        default="json_schema",
    )
    research_semantic_classify.add_argument("--batch-size", type=int, default=4)
    research_semantic_classify.add_argument("--limit", type=int, default=None)
    research_semantic_classify.add_argument("--max-snippet-chars", type=int, default=4000)
    research_semantic_classify.add_argument(
        "--routing-mode",
        choices=("broad", "targeted"),
        default="broad",
        help="Use broad 8-K snippets or targeted SEC item routing before LLM classification.",
    )
    research_semantic_classify.add_argument(
        "--target-item",
        action="append",
        default=None,
        help="SEC item number to include in targeted routing; repeatable.",
    )
    research_semantic_classify.add_argument(
        "--accepted-from",
        default=None,
        help="Inclusive accepted/filed date lower bound for semantic routing, YYYY-MM-DD.",
    )
    research_semantic_classify.add_argument(
        "--accepted-to",
        default=None,
        help="Inclusive accepted/filed date upper bound for semantic routing, YYYY-MM-DD.",
    )
    research_semantic_classify.add_argument(
        "--snippet-kind",
        choices=("all", "item_section", "exhibit"),
        default="all",
        help="Restrict semantic snippets by source kind before LLM classification.",
    )
    research_semantic_classify.add_argument(
        "--labelability-mode",
        choices=("all", "prefer-labelable", "labelable-only"),
        default="all",
        help="Use market-label feasibility to order or filter SEC rows before LLM classification.",
    )
    research_semantic_classify.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful classifications from the semantic checkpoint.",
    )
    research_semantic_classify.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional semantic classification checkpoint parquet path.",
    )
    research_labelability = research_subparsers.add_parser(
        "sec-event-semantic-labelability-audit",
        help="Audit which deterministic SEC 8-K rows can be market-labeled before LLM classification.",
    )
    research_labelability.add_argument(
        "--routing-mode",
        choices=("broad", "targeted"),
        default="targeted",
    )
    research_labelability.add_argument("--target-item", action="append", default=None)
    research_labelability.add_argument("--accepted-from", default=None)
    research_labelability.add_argument("--accepted-to", default=None)
    research_labelability.add_argument("--snippet-kind", default="item_section")
    research_labelability.add_argument("--horizon", action="append", type=int, default=None)
    research_labelability.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_scaled_gate = research_subparsers.add_parser(
        "sec-event-semantic-scaled-gate",
        help="Run the labelability-first scaled SEC 8-K semantic decision gate.",
    )
    research_scaled_gate.add_argument("--model", default=DEFAULT_SEC_EVENT_MODEL)
    research_scaled_gate.add_argument("--base-url", default="http://127.0.0.1:1235/v1")
    research_scaled_gate.add_argument("--timeout-seconds", type=float, default=300.0)
    research_scaled_gate.add_argument(
        "--response-format-mode",
        choices=("json_schema", "prompt_json"),
        default="prompt_json",
    )
    research_scaled_gate.add_argument("--batch-size", type=int, default=1)
    research_scaled_gate.add_argument("--target-item", action="append", default=None)
    research_scaled_gate.add_argument("--fallback-target-item", action="append", default=None)
    research_scaled_gate.add_argument("--year", action="append", type=int, default=None)
    research_scaled_gate.add_argument("--max-snippets", type=int, default=None)
    research_scaled_gate.add_argument("--no-resume", action="store_true")
    research_scaled_gate.add_argument("--primary-horizon", type=int, default=5)
    research_scaled_gate.add_argument("--min-sample", type=int, default=100)
    research_scaled_gate.add_argument("--min-mean-abret", type=float, default=0.005)
    research_scaled_gate.add_argument("--min-control-separation", type=float, default=0.0075)
    research_scaled_gate.add_argument("--max-top5-abs-contribution", type=float, default=0.35)
    research_coverage_gate = research_subparsers.add_parser(
        "sec-event-semantic-coverage-gate",
        help="Expand SEC 8-K coverage, repair labelability, then run the semantic scaled gate.",
    )
    research_coverage_gate.add_argument("--start-date", default="2024-01-01")
    research_coverage_gate.add_argument("--end-date", default="2025-12-31")
    research_coverage_gate.add_argument("--model", default=DEFAULT_SEC_EVENT_MODEL)
    research_coverage_gate.add_argument("--base-url", default="http://127.0.0.1:1235/v1")
    research_coverage_gate.add_argument("--timeout-seconds", type=float, default=300.0)
    research_coverage_gate.add_argument(
        "--response-format-mode",
        choices=("json_schema", "prompt_json"),
        default="prompt_json",
    )
    research_coverage_gate.add_argument("--batch-size", type=int, default=1)
    research_coverage_gate.add_argument("--target-item", action="append", default=None)
    research_coverage_gate.add_argument("--fallback-target-item", action="append", default=None)
    research_coverage_gate.add_argument("--max-snippets", type=int, default=None)
    research_coverage_gate.add_argument("--no-resume", action="store_true")
    research_coverage_gate.add_argument("--primary-horizon", type=int, default=5)
    research_coverage_gate.add_argument("--min-sample", type=int, default=100)
    research_coverage_gate.add_argument("--min-mean-abret", type=float, default=0.005)
    research_coverage_gate.add_argument("--min-control-separation", type=float, default=0.0075)
    research_coverage_gate.add_argument("--max-top5-abs-contribution", type=float, default=0.35)
    research_coverage_gate.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_coverage_gate.add_argument("--no-expand", action="store_true")
    research_coverage_gate.add_argument("--no-market-repair", action="store_true")
    research_coverage_gate.add_argument("--limit-per-month", type=int, default=None)
    research_coverage_gate.add_argument("--user-agent", default=None)
    research_coverage_gate.add_argument("--max-retrieval-attempts", type=int, default=6)
    research_coverage_gate.add_argument(
        "--sec-rate-limit-pause-seconds",
        type=float,
        default=60.0,
    )
    research_coverage_gate.add_argument("--market-max-fetch-attempts", type=int, default=6)
    research_coverage_gate.add_argument(
        "--market-rate-limit-pause-seconds",
        type=float,
        default=60.0,
    )
    research_coverage_gate.add_argument("--daily-symbol-batch-size", type=int, default=100)
    research_semantic_study = research_subparsers.add_parser(
        "sec-event-semantic-study",
        help="Label SEC 8-K semantic events and write a study/decision packet.",
    )
    research_semantic_study.add_argument("--primary-horizon", type=int, default=5)
    research_semantic_study.add_argument("--horizon", action="append", type=int, default=None)
    research_semantic_study.add_argument("--round-trip-cost-bps", type=float, default=50.0)
    research_semantic_study.add_argument(
        "--market-data-root",
        action="append",
        default=None,
        help="Additional root to search for market-data partitions; repeatable.",
    )
    research_semantic_study.add_argument(
        "--source-contract",
        default=None,
        help="Explicit feature source contract JSON path for market-data source mapping.",
    )
    research_feature_canary = research_subparsers.add_parser("feature-canary", help="Build and canary configured modeling feature versions.")
    research_feature_canary.add_argument("--program", default="configs/research/perpetual_macmini.yml")
    research_feature_canary.add_argument("--poll-seconds", type=int, default=None)
    research_feature_canary.add_argument("--detach", action="store_true")
    research_feature_canary.add_argument("--feature-version", action="append", default=None)
    research_feature_canary.add_argument("--label-horizon", type=int, default=None)
    research_feature_canary.add_argument("--report-date", default=None)
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
    research_reload = research_subparsers.add_parser("reload", help="Reload a research supervisor after code/config updates.")
    research_reload.add_argument("--program-id", required=True)
    research_reload.add_argument("--program", default=None)
    research_reload.add_argument("--label", default=None)
    research_reload.add_argument("--no-interrupt-active", action="store_true")
    research_reload.add_argument("--no-requeue-interrupted", action="store_true")
    research_stop = research_subparsers.add_parser("stop", help="Stop a research program.")
    research_stop.add_argument("--program-id", required=True)
    research_frontier = research_subparsers.add_parser("frontier", help="Show the current program frontier summary.")
    research_frontier.add_argument("--program-id", default=None)
    research_review = research_subparsers.add_parser("review-packet", help="Write a research review packet.")
    research_review.add_argument("--program-id", required=True)
    research_health_parser = research_subparsers.add_parser("health", help="Show hardened research health.")
    research_health_parser.add_argument("--program-id", required=True)
    research_audit_parser = research_subparsers.add_parser("audit", help="Write and print the research progression audit.")
    research_audit_parser.add_argument("--program-id", required=True)
    research_audit_parser.add_argument("--json", action="store_true")
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
    fleet_watchdog = fleet_subparsers.add_parser("watchdog", help="Run one scheduled fleet watchdog pass.")
    fleet_watchdog.add_argument("--data-root", default=None)
    fleet_watchdog.add_argument("--pi-host", default=None)
    fleet_watchdog.add_argument("--pi-user", default="zach")
    fleet_watchdog.add_argument("--pi-password-env", default="TRADEML_PI_PASSWORD")
    fleet_watchdog.add_argument("--mac-host", default=None)
    fleet_watchdog.add_argument("--mac-user", default="openclaw")
    fleet_watchdog.add_argument("--mac-password-env", default="TRADEML_MAC_PASSWORD")
    fleet_watchdog.add_argument("--heal", action="store_true")
    fleet_watchdog.add_argument("--json", action="store_true")
    fleet_audit = fleet_subparsers.add_parser("audit", help="Run a full fleet audit and write the Codex curation feed.")
    fleet_audit.add_argument("--data-root", default=None)
    fleet_audit.add_argument("--pi-host", default=None)
    fleet_audit.add_argument("--pi-user", default="zach")
    fleet_audit.add_argument("--pi-password-env", default="TRADEML_PI_PASSWORD")
    fleet_audit.add_argument("--mac-host", default=None)
    fleet_audit.add_argument("--mac-user", default="openclaw")
    fleet_audit.add_argument("--mac-password-env", default="TRADEML_MAC_PASSWORD")
    fleet_audit.add_argument("--program-id", default="perpetual-macmini")
    fleet_audit.add_argument("--json", action="store_true")

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
    if args.node_command == "compact-archives":
        print(
            json.dumps(
                compact_archive_partitions(
                    data_root=settings.nas_mount,
                    datasets=args.dataset,
                    max_partitions=args.max_partitions,
                    dry_run=bool(args.dry_run),
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
    if args.node_command == "controller-status":
        payload = DataNodeDB(settings.db_path).summarize_controller_decisions(
            minutes=int(args.minutes)
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.node_command == "data-quality":
        payload = run_data_quality_audit(
            data_root=settings.nas_mount,
            db=DataNodeDB(settings.db_path),
            datasets=args.dataset,
        )
        print(json.dumps(payload, indent=2, default=str))
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
    if args.fleet_command == "watchdog":
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
        config = yaml.safe_load(settings.config_path.read_text(encoding="utf-8")) or {}
        payload = run_fleet_watchdog_once(
            local_snapshot=snapshot,
            data_root=data_root,
            pi=pi,
            mac=mac,
            heal=bool(args.heal),
            policy=(config.get("fleet") or {}).get("watchdog") if isinstance(config, dict) else None,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.fleet_command == "audit":
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
        config = yaml.safe_load(settings.config_path.read_text(encoding="utf-8")) or {}
        payload = run_fleet_audit(
            local_snapshot=snapshot,
            data_root=data_root,
            repo_root=settings.repo_root,
            local_state=settings.local_state,
            targets_config_path=settings.config_path,
            python_executable=sys.executable,
            program_id=args.program_id,
            pi=pi,
            mac=mac,
            config=config if isinstance(config, dict) else {},
        )
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
    if args.research_command == "source-contract":
        dataset_paths: dict[str, list[Path | str]] = {}
        for raw_item in list(args.dataset_path or []):
            if "=" not in str(raw_item):
                raise SystemExit(f"--dataset-path must be DATASET=PATH, got {raw_item!r}")
            dataset, raw_path = str(raw_item).split("=", 1)
            dataset_paths.setdefault(dataset, []).append(Path(raw_path).expanduser())
        payload = write_research_feature_source_contract(
            data_root=data_root,
            source_root=Path(args.source_root).expanduser() if args.source_root else None,
            dataset_paths=dataset_paths or None,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-fixture-gate":
        try:
            payload = run_form4_fixture_gate_from_env(
                data_root=data_root,
                limit=args.limit,
                user_agent=args.user_agent,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-ingest":
        try:
            payload = run_form4_ingest_from_env(
                data_root=data_root,
                start_date=args.start_date,
                end_date=args.end_date,
                limit=args.limit,
                user_agent=args.user_agent,
                max_retrieval_attempts=args.max_retrieval_attempts,
                rate_limit_pause_seconds=args.rate_limit_pause_seconds,
                use_cache=not args.no_cache,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-candidates":
        payload = run_form4_candidate_curation(data_root=data_root)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-market-backfill":
        payload = run_form4_market_backfill_from_env(
            data_root=data_root,
            horizons=tuple(args.horizon or [1, 5, 10, 20]),
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            limit_events=args.limit_events,
            include_controls=not args.no_controls,
            max_fetch_attempts=args.max_fetch_attempts,
            rate_limit_pause_seconds=args.rate_limit_pause_seconds,
            daily_symbol_batch_size=args.daily_symbol_batch_size,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-labels":
        payload = run_form4_label_curation(
            data_root=data_root,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            market_data_roots=[
                Path(item).expanduser() for item in list(args.market_data_root or [])
            ],
            source_contract_path=(
                Path(args.source_contract).expanduser()
                if args.source_contract
                else None
            ),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-event-study":
        payload = run_form4_event_study(
            data_root=data_root,
            primary_horizon=args.primary_horizon,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            min_historical_sample=args.min_historical_sample,
            market_data_roots=[
                Path(item).expanduser() for item in list(args.market_data_root or [])
            ],
            source_contract_path=(
                Path(args.source_contract).expanduser()
                if args.source_contract
                else None
            ),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "form4-rework-study":
        payload = run_form4_rework_study(data_root=data_root)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-ingest":
        try:
            payload = run_sec8k_ingest_from_env(
                data_root=data_root,
                start_date=args.start_date,
                end_date=args.end_date,
                limit=args.limit,
                user_agent=args.user_agent,
                max_retrieval_attempts=args.max_retrieval_attempts,
                rate_limit_pause_seconds=args.rate_limit_pause_seconds,
                use_cache=not args.no_cache,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-candidates":
        payload = run_sec8k_candidate_curation(
            data_root=data_root,
            filings_path=Path(args.filings_path).expanduser()
            if args.filings_path
            else None,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-market-backfill":
        payload = run_sec8k_market_backfill_from_env(
            data_root=data_root,
            horizons=tuple(args.horizon or [1, 5, 10, 20]),
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            limit_events=args.limit_events,
            include_timestamp_placebo=not args.no_timestamp_placebo,
            max_fetch_attempts=args.max_fetch_attempts,
            rate_limit_pause_seconds=args.rate_limit_pause_seconds,
            daily_symbol_batch_size=args.daily_symbol_batch_size,
            candidate_source=args.candidate_source,
            target_items=tuple(args.target_item or ()),
            accepted_from=args.accepted_from,
            accepted_to=args.accepted_to,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-event-study":
        payload = run_sec8k_event_study(
            data_root=data_root,
            primary_horizon=args.primary_horizon,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            market_data_roots=[
                Path(item).expanduser() for item in list(args.market_data_root or [])
            ],
            source_contract_path=(
                Path(args.source_contract).expanduser()
                if args.source_contract
                else None
            ),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-decision":
        payload = run_sec8k_research_decision(
            data_root=data_root,
            min_labeled_events=args.min_labeled_events,
            min_family_events=args.min_family_events,
            min_mean_abret=args.min_mean_abret,
            min_control_separation=args.min_control_separation,
            max_top5_abs_contribution=args.max_top5_abs_contribution,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-coverage-audit":
        payload = run_sec8k_coverage_audit(
            data_root=data_root,
            start_date=args.start_date,
            end_date=args.end_date,
            target_items=args.target_item,
            fallback_target_items=args.fallback_target_item,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec8k-coverage-expand":
        payload = run_sec8k_coverage_expand(
            data_root=data_root,
            start_date=args.start_date,
            end_date=args.end_date,
            target_items=args.target_item,
            fallback_target_items=args.fallback_target_item,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            limit_per_month=args.limit_per_month,
            user_agent=args.user_agent,
            max_retrieval_attempts=args.max_retrieval_attempts,
            rate_limit_pause_seconds=args.rate_limit_pause_seconds,
            use_cache=not args.no_cache,
            rebuild_candidates=not args.no_rebuild_candidates,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-gate":
        payload = run_sec_event_semantic_fixture_gate(
            data_root=data_root,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            response_format_mode=args.response_format_mode,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-classify":
        payload = run_sec_event_semantic_classification(
            data_root=data_root,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            response_format_mode=args.response_format_mode,
            batch_size=args.batch_size,
            limit=args.limit,
            max_snippet_chars=args.max_snippet_chars,
            routing_mode=args.routing_mode,
            target_items=args.target_item,
            accepted_from=args.accepted_from,
            accepted_to=args.accepted_to,
            snippet_kind=args.snippet_kind,
            labelability_mode=args.labelability_mode,
            resume=args.resume,
            checkpoint_path=Path(args.checkpoint_path).expanduser()
            if args.checkpoint_path
            else None,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-labelability-audit":
        payload = run_sec_event_semantic_labelability_audit(
            data_root=data_root,
            routing_mode=args.routing_mode,
            target_items=args.target_item,
            accepted_from=args.accepted_from,
            accepted_to=args.accepted_to,
            snippet_kind=args.snippet_kind,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-scaled-gate":
        payload = run_sec_event_semantic_scaled_gate(
            data_root=data_root,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            response_format_mode=args.response_format_mode,
            batch_size=args.batch_size,
            target_items=args.target_item,
            fallback_target_items=args.fallback_target_item,
            years=args.year,
            max_snippets=args.max_snippets,
            resume=not args.no_resume,
            primary_horizon=args.primary_horizon,
            min_sample=args.min_sample,
            min_mean_abret=args.min_mean_abret,
            min_control_separation=args.min_control_separation,
            max_top5_abs_contribution=args.max_top5_abs_contribution,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-coverage-gate":
        payload = run_sec_event_semantic_coverage_gate(
            data_root=data_root,
            start_date=args.start_date,
            end_date=args.end_date,
            target_items=args.target_item,
            fallback_target_items=args.fallback_target_item,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            response_format_mode=args.response_format_mode,
            batch_size=args.batch_size,
            max_snippets=args.max_snippets,
            resume=not args.no_resume,
            primary_horizon=args.primary_horizon,
            min_sample=args.min_sample,
            min_mean_abret=args.min_mean_abret,
            min_control_separation=args.min_control_separation,
            max_top5_abs_contribution=args.max_top5_abs_contribution,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            expand_missing_coverage=not args.no_expand,
            repair_market_coverage=not args.no_market_repair,
            limit_per_month=args.limit_per_month,
            user_agent=args.user_agent,
            max_retrieval_attempts=args.max_retrieval_attempts,
            sec_rate_limit_pause_seconds=args.sec_rate_limit_pause_seconds,
            market_max_fetch_attempts=args.market_max_fetch_attempts,
            market_rate_limit_pause_seconds=args.market_rate_limit_pause_seconds,
            daily_symbol_batch_size=args.daily_symbol_batch_size,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "sec-event-semantic-study":
        payload = run_sec_event_semantic_study(
            data_root=data_root,
            primary_horizon=args.primary_horizon,
            horizons=args.horizon,
            round_trip_cost_bps=float(args.round_trip_cost_bps),
            market_data_roots=[
                Path(item).expanduser() for item in list(args.market_data_root or [])
            ],
            source_contract_path=(
                Path(args.source_contract).expanduser()
                if args.source_contract
                else None
            ),
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "feature-canary":
        payload = run_feature_version_canary_batch(
            program_path=Path(args.program).expanduser(),
            poll_seconds=args.poll_seconds,
            detach=bool(args.detach),
            feature_versions=args.feature_version,
            label_horizon=args.label_horizon,
            report_date=args.report_date,
            **common,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.research_command == "paper-smoke":
        payload = run_and_persist_paper_account_smoke(
            program_path=Path(args.program).expanduser(),
            local_state=local_state,
        )
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
    if args.research_command == "reload":
        payload = reload_research_program(
            program_id=args.program_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=sys.executable,
            program_path=Path(args.program).expanduser() if args.program else None,
            label=args.label,
            interrupt_active=not bool(args.no_interrupt_active),
            requeue_interrupted=not bool(args.no_requeue_interrupted),
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
    if args.research_command == "audit":
        payload = run_research_progression_audit(
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
