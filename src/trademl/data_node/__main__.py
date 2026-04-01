"""CLI entry point for the TradeML data node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


def main() -> int:
    """Run one deterministic data-node cycle from config."""
    parser = argparse.ArgumentParser(description="Run the TradeML data node once.")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    args = parser.parse_args()
    _load_dotenv()

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
    local_state = Path(os.getenv("LOCAL_STATE", config["node"]["local_state"])).expanduser()
    db = DataNodeDB(local_state / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors=connectors,
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=Path(args.root) / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=Path(args.root)),
    )
    service.install_signal_handlers()
    service.run_cycle(
        trading_date=args.date,
        symbols=args.symbols,
        exchange="XNYS",
        audit_start=args.date,
        audit_end=args.date,
    )
    if "fred" in connectors:
        service.collect_macro_data(["DGS10"], args.date, args.date)
    reference_jobs: list[dict[str, object]] = []
    if "massive" in connectors:
        reference_jobs.extend(
            [
                {"source": "massive", "dataset": "reference_tickers", "symbols": [], "start_date": args.date, "end_date": args.date, "output_name": "universe"},
                {"source": "massive", "dataset": "reference_splits", "symbols": args.symbols[:10], "start_date": args.date, "end_date": args.date, "output_name": "splits"},
                {"source": "massive", "dataset": "reference_dividends", "symbols": args.symbols[:10], "start_date": args.date, "end_date": args.date, "output_name": "dividends"},
            ]
        )
    if "fmp" in connectors:
        reference_jobs.append({"source": "fmp", "dataset": "delistings", "symbols": [], "start_date": args.date, "end_date": args.date, "output_name": "delistings"})
    if reference_jobs:
        service.collect_reference_data(reference_jobs)
    if "massive" in connectors and "finnhub" in connectors:
        service.run_cross_vendor_price_checks(trading_date=args.date, sample_symbols=args.symbols[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
