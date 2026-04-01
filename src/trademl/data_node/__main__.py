"""CLI entry point for the TradeML data node."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.connectors.alpaca import AlpacaConnector
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService


def main() -> int:
    """Run one deterministic data-node cycle from config."""
    parser = argparse.ArgumentParser(description="Run the TradeML data node once.")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    budgets = BudgetManager({vendor: {"rpm": values["rpm"], "daily_cap": values["daily_cap"]} for vendor, values in config["vendors"].items()})
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        api_key="",
        budget_manager=budgets,
    )
    db = DataNodeDB(Path(args.root) / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": connector},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
