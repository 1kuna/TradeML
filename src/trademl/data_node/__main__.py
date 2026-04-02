"""CLI entry point for the TradeML data node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.connectors.tiingo import TiingoConnector
from trademl.connectors.twelve_data import TwelveDataConnector
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.bootstrap import Stage0UniverseBuilder
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.capabilities import build_reference_jobs as build_reference_jobs_from_registry
from trademl.data_node.capabilities import default_macro_series, load_audit_state
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService
from trademl.fleet.cluster import ClusterCoordinator


DEFAULT_VENDOR_LIMITS = {
    "alpaca": {"rpm": 150, "daily_cap": 10000},
    "tiingo": {"rpm": 40, "daily_cap": 400},
    "twelve_data": {"rpm": 6, "daily_cap": 600},
    "massive": {"rpm": 4, "daily_cap": 300},
    "finnhub": {"rpm": 50, "daily_cap": 10000},
    "alpha_vantage": {"rpm": 4, "daily_cap": 400},
    "fred": {"rpm": 80, "daily_cap": 5000},
    "fmp": {"rpm": 3, "daily_cap": 200},
    "sec_edgar": {"rpm": 8, "daily_cap": 5000},
}


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

    budgets = BudgetManager(_resolve_vendor_budgets(config))
    connector = AlpacaConnector(
        base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        trading_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_API_SECRET", ""),
        budget_manager=budgets,
    )
    stage0_universe_builder = Stage0UniverseBuilder(connector=connector)
    connectors = {"alpaca": connector}
    if os.getenv("MASSIVE_API_KEY"):
        from trademl.connectors.massive import MassiveConnector

        connectors["massive"] = MassiveConnector(
            base_url="https://api.polygon.io",
            api_key=os.environ["MASSIVE_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("FINNHUB_API_KEY"):
        from trademl.connectors.finnhub import FinnhubConnector

        connectors["finnhub"] = FinnhubConnector(
            base_url="https://finnhub.io",
            api_key=os.environ["FINNHUB_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("FRED_API_KEY"):
        from trademl.connectors.fred import FredConnector

        connectors["fred"] = FredConnector(
            base_url="https://api.stlouisfed.org",
            api_key=os.environ["FRED_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        connectors["alpha_vantage"] = AlphaVantageConnector(
            base_url="https://www.alphavantage.co",
            api_key=os.environ["ALPHA_VANTAGE_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("TIINGO_API_KEY"):
        connectors["tiingo"] = TiingoConnector(
            base_url="https://api.tiingo.com",
            api_key=os.environ["TIINGO_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("TWELVE_DATA_API_KEY"):
        connectors["twelve_data"] = TwelveDataConnector(
            base_url="https://api.twelvedata.com",
            api_key=os.environ["TWELVE_DATA_API_KEY"],
            budget_manager=budgets,
        )
    if os.getenv("FMP_API_KEY"):
        connectors["fmp"] = FMPConnector(
            base_url="https://financialmodelingprep.com",
            api_key=os.environ["FMP_API_KEY"],
            budget_manager=budgets,
        )
    connectors["sec_edgar"] = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent=os.getenv("SEC_EDGAR_USER_AGENT", "TradeML/0.1 test@example.com"),
        budget_manager=budgets,
    )
    workspace_root = Path(args.root).expanduser() if args.root else Path(
        os.getenv("LOCAL_STATE", config["node"]["local_state"])
    ).expanduser().parent
    local_state_default = workspace_root / "control" if args.root else Path(config["node"]["local_state"]).expanduser()
    local_state = Path(os.getenv("LOCAL_STATE", str(local_state_default))).expanduser()
    worker_id = os.getenv("EDGE_NODE_ID", config.get("node", {}).get("worker_id", os.uname().nodename if hasattr(os, "uname") else "worker"))
    db = DataNodeDB(local_state / "node.sqlite")
    data_root = Path(args.data_root or os.getenv("NAS_MOUNT", config["node"]["nas_mount"])).expanduser()
    audit_state = load_audit_state(data_root / "control" / "cluster" / "state" / "vendor_audit.json")
    symbols = args.symbols or _load_stage_symbols(workspace_root)
    service = DataNodeService(
        db=db,
        connectors=connectors,
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=data_root / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=data_root),
        capability_audit_state=audit_state,
        worker_id=worker_id,
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
            service.collect_macro_data(["DGS10"], args.date, args.date)
        if reference_jobs:
            materialized = [service._materialize_job(job, args.date) for job in reference_jobs]
            service.collect_reference_data(materialized)
            service.curate_dates(corp_actions=service.load_corp_actions_reference())
        if "massive" in connectors and "finnhub" in connectors:
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
    rebuilt = coordinator.rebuild_local_state(local_db_path=local_state / "node.sqlite")
    db = DataNodeDB(local_state / "node.sqlite")
    service.db = db
    service.auditor.db = db
    service.run_cluster_forever(
        coordinator=coordinator,
        symbols=symbols or manifest["stage"]["symbols"],
        exchange=manifest["datasets"]["equities_eod"].get("exchange", "XNYS"),
        collection_time_et=os.getenv("COLLECTION_TIME_ET", manifest["schedule"]["collection_time_et"]),
        maintenance_hour_local=int(os.getenv("MAINTENANCE_HOUR_LOCAL", manifest["schedule"]["maintenance_hour_local"])),
        poll_seconds=args.poll_seconds,
        macro_series_ids=default_macro_series() if "fred" in connectors else [],
        reference_jobs=reference_jobs,
        price_check_symbols=(symbols or manifest["stage"]["symbols"])[:5] if {"massive", "finnhub"}.issubset(connectors) else [],
    )
    return 0


def _load_stage_symbols(root: Path) -> list[str]:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return []
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return list(stage.get("symbols", []))


def _resolve_vendor_budgets(config: dict[str, object]) -> dict[str, dict[str, int]]:
    resolved = {name: limits.copy() for name, limits in DEFAULT_VENDOR_LIMITS.items()}
    for vendor, values in (config.get("vendors", {}) or {}).items():
        if not isinstance(values, dict):
            continue
        existing = resolved.get(str(vendor), {"rpm": 1, "daily_cap": 1})
        resolved[str(vendor)] = {
            "rpm": int(values.get("rpm", existing["rpm"])),
            "daily_cap": int(values.get("daily_cap", existing["daily_cap"])),
        }
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
