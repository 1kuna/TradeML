"""CLI entry point for the TradeML data node."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path

import pandas as pd
import yaml

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.bootstrap import Stage0UniverseBuilder
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.capabilities import build_reference_jobs as build_reference_jobs_from_registry
from trademl.data_node.capabilities import default_macro_series, load_audit_state
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.runtime import build_connectors, build_connector, resolve_vendor_budgets
from trademl.data_node.service import DataNodePaths, DataNodeService
from trademl.fleet.cluster import ClusterCoordinator


def _build_reference_jobs(*, connectors: dict[str, object], symbols: list[str]) -> list[dict[str, object]]:
    """Build the default weekly reference collection plan from verified connector lanes."""
    return build_reference_jobs_from_registry(connectors=connectors, symbols=symbols)


def _load_dotenv(env_path: Path | None) -> None:
    if env_path is None or not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


def main() -> int:
    """Run one deterministic data-node cycle or the scheduled service loop."""
    parser = argparse.ArgumentParser(description="Run the TradeML data node.")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--passphrase", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    env_path = None
    if args.env_file:
        env_path = Path(args.env_file).expanduser()
    elif args.root:
        env_path = Path(args.root).expanduser() / ".env"
    elif Path(".env").exists():
        env_path = Path(".env")
    _load_dotenv(env_path)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    workspace_root = Path(args.root).expanduser() if args.root else Path(
        os.getenv("LOCAL_STATE", config["node"]["local_state"])
    ).expanduser().parent
    local_state_default = workspace_root / "control" if args.root else Path(config["node"]["local_state"]).expanduser()
    local_state = Path(os.getenv("LOCAL_STATE", str(local_state_default))).expanduser()
    vendor_limits = resolve_vendor_budgets(config)
    budgets = BudgetManager(
        vendor_limits,
        snapshot_path=local_state / "budget_state.json",
    )
    connectors = build_connectors(
        env_values=os.environ,
        vendor_limits=vendor_limits,
        budget_manager_factory=lambda _vendor: budgets,
        sec_edgar_user_agent="TradeML/0.1 test@example.com",
    )
    connector = build_connector(
        vendor="alpaca",
        env_values=os.environ,
        vendor_limits=vendor_limits,
        budget_manager_factory=lambda _vendor: budgets,
    )
    if connector is None:
        raise SystemExit("ALPACA_API_KEY is required to start the data node")
    stage0_universe_builder = Stage0UniverseBuilder(connector=connector)
    worker_id = os.getenv("EDGE_NODE_ID", config.get("node", {}).get("worker_id", os.uname().nodename if hasattr(os, "uname") else "worker"))
    db = DataNodeDB(local_state / "node.sqlite")
    data_root = Path(args.data_root or os.getenv("NAS_MOUNT", config["node"]["nas_mount"])).expanduser()
    audit_state = load_audit_state(data_root / "control" / "cluster" / "state" / "vendor_audit.json")
    symbols = args.symbols or _load_stage_symbols(workspace_root)
    stage_years = _load_stage_years(workspace_root)
    service = DataNodeService(
        db=db,
        connectors=connectors,
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=data_root / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=data_root),
        capability_audit_state=audit_state,
        worker_id=worker_id,
        stage_years=stage_years,
    )
    service.install_signal_handlers()
    reference_jobs = build_reference_jobs_from_registry(connectors=connectors, symbols=symbols, audit_state=audit_state)

    if args.once or args.date:
        if not args.date:
            raise SystemExit("--date is required with --once")
        service.run_cycle(
            trading_date=args.date,
            symbols=symbols,
            exchange="XNYS",
            audit_start=args.date,
            audit_end=args.date,
        )
        if "fred" in connectors:
            service.collect_macro_data(default_macro_series(), args.date, args.date)
        if reference_jobs:
            materialized = [service._materialize_job(job, args.date) for job in reference_jobs]
            service.collect_reference_data(materialized)
            service.curate_dates(corp_actions=service.load_corp_actions_reference())
        if {"massive", "twelve_data", "tiingo"}.intersection(connectors):
            service.run_cross_vendor_price_checks(trading_date=args.date, sample_symbols=symbols[:5])
        service.sync_partition_status()
        return 0

    coordinator = ClusterCoordinator(
        nas_root=data_root,
        workspace_root=workspace_root,
        config_path=Path(args.config).expanduser(),
        env_path=env_path or (workspace_root / ".env"),
        local_state=local_state,
        nas_share=os.getenv("NAS_SHARE", config["node"].get("nas_share", "//nas/trademl")),
        worker_id=worker_id,
        lease_ttl_seconds=int(config["node"].get("lease_ttl_seconds", 90)),
        heartbeat_interval_seconds=int(config["node"].get("heartbeat_interval_seconds", 30)),
        universe_builder=stage0_universe_builder,
    )
    manifest = coordinator.ensure_cluster_ready(passphrase=args.passphrase)
    local_db_path = local_state / "node.sqlite"
    if _should_rebuild_local_state(local_db_path=local_db_path):
        coordinator.rebuild_local_state(local_db_path=local_db_path)
    db = DataNodeDB(local_state / "node.sqlite")
    service.db = db
    service.auditor.db = db
    active_symbols = symbols or manifest["stage"]["symbols"]
    service.default_symbols = list(active_symbols)
    service._ensure_planner_backlog_seeded(
        trading_date=(args.date or datetime.now(tz=UTC).date().isoformat())
    )
    service.run_cluster_forever(
        coordinator=coordinator,
        symbols=active_symbols,
        exchange=manifest["datasets"]["equities_eod"].get("exchange", "XNYS"),
        collection_time_et=os.getenv("COLLECTION_TIME_ET", manifest["schedule"]["collection_time_et"]),
        maintenance_hour_local=int(os.getenv("MAINTENANCE_HOUR_LOCAL", manifest["schedule"]["maintenance_hour_local"])),
        poll_seconds=args.poll_seconds,
        macro_series_ids=default_macro_series() if "fred" in connectors else [],
        reference_jobs=reference_jobs,
        price_check_symbols=active_symbols[:5] if {"massive", "twelve_data", "tiingo"}.intersection(connectors) else [],
    )
    return 0


def _should_rebuild_local_state(*, local_db_path: Path) -> bool:
    """Return whether startup should rebuild the disposable local DB from NAS."""
    if os.getenv("TRADEML_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return not local_db_path.exists()


def _load_stage_symbols(root: Path) -> list[str]:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return []
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return list(stage.get("symbols", []))


def _load_stage_years(root: Path) -> int:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return 5
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return int(stage.get("years", 5) or 5)


def _resolve_vendor_budgets(config: dict[str, object]) -> dict[str, dict[str, int]]:
    return resolve_vendor_budgets(config)


if __name__ == "__main__":
    raise SystemExit(main())
