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
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService


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

    budgets = BudgetManager({vendor: {"rpm": values["rpm"], "daily_cap": values["daily_cap"]} for vendor, values in config["vendors"].items()})
    connector = AlpacaConnector(
        base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_API_SECRET", ""),
        budget_manager=budgets,
    )
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
    db = DataNodeDB(local_state / "node.sqlite")
    data_root = Path(args.data_root or os.getenv("NAS_MOUNT", config["node"]["nas_mount"])).expanduser()
    symbols = args.symbols or _load_stage_symbols(workspace_root)
    service = DataNodeService(
        db=db,
        connectors=connectors,
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=data_root / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=data_root),
    )
    service.install_signal_handlers()
    reference_jobs: list[dict[str, object]] = []
    if "massive" in connectors:
        reference_jobs.extend(
            [
                {"source": "massive", "dataset": "reference_tickers", "symbols": [], "output_name": "universe"},
                {"source": "massive", "dataset": "reference_splits", "symbols": symbols[:10], "output_name": "splits"},
                {"source": "massive", "dataset": "reference_dividends", "symbols": symbols[:10], "output_name": "dividends"},
            ]
        )
    if "fmp" in connectors:
        reference_jobs.extend(
            [
                {"source": "fmp", "dataset": "delistings", "symbols": [], "output_name": "delistings"},
                {"source": "fmp", "dataset": "earnings_calendar", "symbols": [], "output_name": "earnings_calendar_fmp"},
            ]
        )
    if "alpha_vantage" in connectors:
        reference_jobs.extend(
            [
                {"source": "alpha_vantage", "dataset": "listings", "symbols": [], "output_name": "listings"},
                {"source": "alpha_vantage", "dataset": "corp_actions", "symbols": symbols[:10], "output_name": "corp_actions"},
            ]
        )
    if "finnhub" in connectors:
        reference_jobs.append({"source": "finnhub", "dataset": "earnings_calendar", "symbols": [], "output_name": "earnings_calendar"})
    reference_jobs.append({"source": "sec_edgar", "dataset": "filing_index", "symbols": ["320193"], "output_name": "sec_filings"})

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

    service.run_forever(
        symbols=symbols,
        exchange="XNYS",
        collection_time_et=os.getenv("COLLECTION_TIME_ET", config["node"]["collection_time_et"]),
        maintenance_hour_local=int(os.getenv("MAINTENANCE_HOUR_LOCAL", config["node"]["maintenance_hour_local"])),
        poll_seconds=args.poll_seconds,
        macro_series_ids=["DGS10"] if "fred" in connectors else [],
        reference_jobs=reference_jobs,
        price_check_symbols=symbols[:5] if {"massive", "finnhub"}.issubset(connectors) else [],
    )
    return 0


def _load_stage_symbols(root: Path) -> list[str]:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return []
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return list(stage.get("symbols", []))


if __name__ == "__main__":
    raise SystemExit(main())
