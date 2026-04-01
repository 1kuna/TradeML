from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.connectors.base import BaseConnector
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService


@dataclass
class MockConnector:
    vendor_name: str = "alpaca"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        assert dataset == "equities_eod"
        return pd.DataFrame(
            [
                {
                    "date": start_date,
                    "symbol": symbol,
                    "open": 100.0 + idx,
                    "high": 101.0 + idx,
                    "low": 99.0 + idx,
                    "close": 100.5 + idx,
                    "vwap": 100.2 + idx,
                    "volume": 1000 + idx,
                    "trade_count": 10 + idx,
                    "ingested_at": pd.Timestamp.utcnow(),
                    "source_name": "alpaca",
                    "source_uri": "/v2/stocks/bars",
                    "vendor_ts": pd.Timestamp(start_date),
                }
                for idx, symbol in enumerate(symbols or ["AAPL", "MSFT"])
            ]
        )


def test_service_cycle_writes_raw_curated_and_qc(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": MockConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    result = service.run_cycle(
        trading_date="2025-01-06",
        symbols=["AAPL", "MSFT"],
        exchange="XNYS",
        audit_start="2025-01-05",
        audit_end="2025-01-06",
        corp_actions=pd.DataFrame(),
    )

    assert result["forward_dates"] == ["2025-01-06"]
    assert (tmp_path / "data" / "raw" / "equities_bars" / "date=2025-01-06" / "data.parquet").exists()
    assert (tmp_path / "data" / "curated" / "equities_ohlcv_adj" / "date=2025-01-06" / "data.parquet").exists()
    assert (tmp_path / "data" / "qc" / "partition_status.parquet").exists()


def test_gap_tasks_are_backfilled_with_default_symbol_universe(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": MockConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    result = service.run_cycle(
        trading_date="2025-01-07",
        symbols=["AAPL", "MSFT"],
        exchange="XNYS",
        audit_start="2025-01-06",
        audit_end="2025-01-07",
        corp_actions=pd.DataFrame(),
    )

    assert "2025-01-06" in result["backfill_dates"]
    assert (tmp_path / "data" / "raw" / "equities_bars" / "date=2025-01-06" / "data.parquet").exists()


def test_auditor_creates_gap_tasks_for_missing_sessions(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    auditor = PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars"))

    result = auditor.audit_range(
        exchange="XNYS",
        source="alpaca",
        dataset="equities_eod",
        start_date="2025-01-06",
        end_date="2025-01-07",
        expected_rows=2,
    )

    assert [day.isoformat() for day in result.missing_dates] == ["2025-01-06", "2025-01-07"]
    leased = db.lease_next_task()
    assert leased is not None
    assert leased.kind == "GAP"
    assert leased.start_date == "2025-01-06"
