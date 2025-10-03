#!/usr/bin/env python3
"""
Smoke test: one-cycle collect -> curate -> audit and summarize artifacts.

Usage:
  STORAGE_BACKEND=local python scripts/smoke.py

It will:
  - Fetch last 2 days AAPL daily bars via Alpaca (feed=iex) into raw/
  - Run curator (configs/curator.yml)
  - Run audit for equities_eod & equities_minute
  - Update delistings reference (AV, FMP)
  - Try a small Finnhub chain fetch for AAPL (optional)
  - Print a summary of counts & paths
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def main():
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo / ".env", override=False)
    os.environ.setdefault("STORAGE_BACKEND", "local")

    # 1) Collect 2 days of AAPL daily bars
    try:
        from data_layer.connectors.alpaca_connector import AlpacaConnector

        conn = AlpacaConnector()
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=1)
        df = conn.fetch_bars(symbols=["AAA".replace("AAA", "AAPL")], start_date=start, end_date=end, timeframe="1Day")
        if not df.empty:
            out = Path("data_layer/raw/equities_bars/alpaca") / f"date={start.isoformat()}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "data.parquet").write_bytes(b"")
            df[df["date"] == start].to_parquet(out / "data.parquet", index=False)
            out2 = Path("data_layer/raw/equities_bars/alpaca") / f"date={end.isoformat()}"
            out2.mkdir(parents=True, exist_ok=True)
            df[df["date"] == end].to_parquet(out2 / "data.parquet", index=False)
            logger.info(f"Collected AAPL bars for {start} and {end}")
        else:
            logger.warning("Alpaca returned empty result for smoke")
    except Exception as e:
        logger.warning(f"Alpaca smoke failed: {e}")

    # 2) Curator
    try:
        from scripts.curator import Curator

        Curator("configs/curator.yml").run()
    except Exception as e:
        logger.warning(f"Curator smoke failed: {e}")

    # 3) Audit
    try:
        from ops.ssot import audit_scan

        audit_scan(["equities_eod", "equities_minute"])
    except Exception as e:
        logger.warning(f"Audit smoke failed: {e}")

    # 4) Reference delistings
    try:
        from ops.ssot.reference import update_reference

        update_reference()
    except Exception as e:
        logger.warning(f"Reference smoke failed: {e}")

    # 5) Options (best-effort)
    try:
        from ops.ssot.backfill import _backfill_options_chains
        _backfill_options_chains(["AAPL"], budget=None, per_day_limit=1)
    except Exception as e:
        logger.warning(f"Options chain smoke failed: {e}")

    # Summary
    try:
        import pandas as pd
        parts = list(Path("data_layer/raw/alpaca/equities_bars").glob("date=*/data.parquet"))
        curated = list(Path("data_layer/curated/equities_ohlcv_adj").glob("date=*/data.parquet"))
        ledger = Path("data_layer/qc/partition_status.parquet")
        counts = {
            "raw_bars": len(parts),
            "curated_days": len(curated),
            "ledger": "present" if ledger.exists() else "missing",
        }
        logger.info(f"Smoke summary: {counts}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

