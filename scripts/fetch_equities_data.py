"""
Standalone script to fetch equities data from Alpaca.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    sys.exit(1)


def fetch_bars(symbols: list[str], start_date: str, end_date: str, output_dir: str):
    """Fetch daily bars and save to parquet."""

    # Initialize Alpaca client
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Create output directories
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    curated_dir = Path("data_layer/curated/equities_ohlcv_adj")
    curated_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching bars for {len(symbols)} symbols from {start_date} to {end_date}")

    all_data = []

    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol}...")

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d"),
            )

            bars = client.get_stock_bars(request_params)

            if symbol not in bars.data:
                logger.warning(f"No data returned for {symbol}")
                continue

            # Convert to dataframe
            df = bars.df.reset_index()

            # Rename columns to match our schema
            df = df.rename(columns={
                'timestamp': 'date',
                'trade_count': 'trades',
            })

            # Ensure symbol column exists
            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            # Add metadata
            df['ingested_at'] = datetime.utcnow()
            df['source_name'] = 'alpaca'
            df['source_uri'] = f'alpaca://data.alpaca.markets/v2/stocks/{symbol}/bars'

            # Add adjustment factor placeholder (1.0 = no adjustment yet)
            df['adj_factor'] = 1.0

            # Convert date to date type
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Save raw data (symbol-partitioned)
            symbol_raw_dir = raw_dir / symbol
            symbol_raw_dir.mkdir(exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(df),
                symbol_raw_dir / "data.parquet"
            )

            # Save curated data (same for now, will apply corp actions later)
            symbol_curated_dir = curated_dir / symbol
            symbol_curated_dir.mkdir(exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(df),
                symbol_curated_dir / "data.parquet"
            )

            all_data.append(df)
            logger.info(f"✓ {symbol}: {len(df)} bars fetched")

        except Exception as e:
            logger.error(f"✗ {symbol}: {e}")
            continue

    if not all_data:
        logger.error("No data fetched for any symbol!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    logger.info(f"✅ Fetched {len(combined_df)} total bars for {len(all_data)} symbols")
    logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    logger.info(f"Saved to: {raw_dir} and {curated_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch equities data from Alpaca")
    parser.add_argument("--universe", default="data_layer/reference/universe_symbols.txt",
                        help="Path to universe file (one symbol per line)")
    parser.add_argument("--start-date", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-10-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data_layer/raw/equities_bars", help="Output directory")

    args = parser.parse_args()

    # Load universe from file
    if Path(args.universe).exists():
        with open(args.universe, "r") as f:
            UNIVERSE = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(UNIVERSE)} symbols from {args.universe}")
    else:
        # Fallback: use basic list
        UNIVERSE = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
            "UNH", "JNJ", "PFE", "ABBV", "XOM", "CVX", "WMT", "HD", "PG",
        ]
        logger.warning(f"Universe file not found, using default {len(UNIVERSE)} symbols")

    # Fetch data
    fetch_bars(
        symbols=UNIVERSE,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output
    )
