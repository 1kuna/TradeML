#!/usr/bin/env python
"""
Build a tiny intraday fixture from archive.zip for manual pipeline testing.

Reads the 1-minute SPY CSV inside archive.zip, trims it to a small date window,
normalizes timestamps to UTC, and writes parquet fixtures in a curated-style
layout:
    ml_testing/fixtures/curated/
        equities_minute/date=YYYY-MM-DD/data.parquet
        equities_ohlcv_adj/SPY.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

DEFAULT_START = "2021-05-03"
DEFAULT_END = "2021-05-06"
DEFAULT_SYMBOL = "SPY"
DEFAULT_SOURCE_TZ = "America/Denver"  # archive timestamps look like 07:30–13:59 (MST/MDT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small intraday fixtures from archive.zip")
    parser.add_argument("--zip-path", default="archive.zip", help="Path to archive.zip with 1_min_SPY_2008-2021.csv")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol to stamp into fixtures")
    parser.add_argument(
        "--output-root",
        default="ml_testing/fixtures/curated",
        help="Root dir for curated-style outputs (equities_minute/, equities_ohlcv_adj/)",
    )
    parser.add_argument("--source-tz", default=DEFAULT_SOURCE_TZ, help="Timezone of timestamps in the CSV")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows to stream per chunk from the zip")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional hard cap on rows (debug only)")
    return parser.parse_args()


def _load_window(
    zip_path: Path,
    start_date: str,
    end_date: str,
    symbol: str,
    source_tz: str,
    chunksize: int,
    limit_rows: Optional[int],
) -> pd.DataFrame:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    rows = []
    total_in = 0
    total_out = 0

    for chunk in pd.read_csv(zip_path, chunksize=chunksize):
        total_in += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str).str.strip(), format="%Y%m%d %H:%M:%S")
        mask = (chunk["date"].dt.date >= start) & (chunk["date"].dt.date <= end)
        if not mask.any():
            continue

        sub = chunk.loc[mask].copy()
        sub["timestamp"] = sub["date"].dt.tz_localize(source_tz).dt.tz_convert("UTC")
        sub["symbol"] = symbol

        keep = sub[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
        rows.append(keep)
        total_out += len(keep)

        if limit_rows is not None and total_out >= limit_rows:
            break

    if not rows:
        raise ValueError(f"No rows found in {zip_path} between {start} and {end}")

    df = pd.concat(rows, ignore_index=True)
    if limit_rows is not None and len(df) > limit_rows:
        df = df.iloc[:limit_rows]

    logger.info(f"Loaded {len(df):,} rows from {zip_path} (scanned {total_in:,})")
    return df


def _write_minute(df: pd.DataFrame, out_root: Path) -> None:
    minute_root = out_root / "equities_minute"
    minute_root.mkdir(parents=True, exist_ok=True)

    grouped = df.copy()
    grouped["date"] = grouped["timestamp"].dt.tz_convert("America/New_York").dt.date

    for day, g in grouped.groupby("date"):
        day_dir = minute_root / f"date={day.isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        g_sorted = g.sort_values("timestamp")
        g_sorted.to_parquet(day_dir / "data.parquet", index=False)
        logger.info(f"Wrote minute fixture: {day_dir}/data.parquet ({len(g_sorted):,} rows)")


def _write_daily(df: pd.DataFrame, symbol: str, out_root: Path) -> None:
    daily_root = out_root / "equities_ohlcv_adj"
    daily_root.mkdir(parents=True, exist_ok=True)

    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.tz_convert("America/New_York").dt.date
    daily = daily.sort_values("timestamp")

    agg = (
        daily.groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    agg["symbol"] = symbol
    cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    agg = agg[cols]

    out_path = daily_root / f"{symbol}.parquet"
    agg.to_parquet(out_path, index=False)
    logger.info(f"Wrote daily fixture: {out_path} ({len(agg):,} rows)")


def main() -> None:
    args = _parse_args()
    zip_path = Path(args.zip_path)
    out_root = Path(args.output_root)

    df = _load_window(
        zip_path=zip_path,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        source_tz=args.source_tz,
        chunksize=args.chunksize,
        limit_rows=args.limit_rows,
    )

    _write_minute(df, out_root)
    _write_daily(df, args.symbol, out_root)
    logger.info("Fixture build complete.")


if __name__ == "__main__":
    main()
