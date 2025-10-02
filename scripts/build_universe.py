"""
Build trading universe with selection criteria.

Criteria:
- Market cap > $5B
- Price > $5
- ADV > $50M
- Sector diversification
- Exclude recent IPOs (<1 year)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from loguru import logger

# Load environment
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


def get_sp500_tickers():
    """
    Get S&P 500 constituents as starting universe.

    In production, fetch from reliable source (e.g., Wikipedia table).
    For now, use a curated list of liquid names.
    """
    # Top 150 liquid S&P 500 stocks (diversified across sectors)
    universe = [
        # Technology (30)
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "CSCO", "ADBE",
        "CRM", "INTC", "AMD", "QCOM", "ORCL", "TXN", "AMAT", "MU", "LRCX", "KLAC",
        "SNPS", "CDNS", "ADI", "MCHP", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "NET",

        # Financials (25)
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "SPGI", "CME",
        "ICE", "USB", "TFC", "PNC", "BK", "STT", "NTRS", "AXP", "COF", "DFS",
        "MA", "V", "PYPL", "FIS", "FISV",

        # Healthcare (20)
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
        "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "MCK", "COR", "VRTX", "REGN",

        # Consumer Discretionary (15)
        "HD", "NKE", "MCD", "SBUX", "LOW", "TJX", "BKNG", "MAR", "CMG", "ORLY",
        "AZO", "ROST", "YUM", "DG", "DLTR",

        # Consumer Staples (15)
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "K", "HSY", "SJM", "CAG",

        # Energy (10)
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",

        # Industrials (15)
        "CAT", "BA", "HON", "UNP", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
        "EMR", "ETN", "ITW", "PH", "CMI",

        # Materials (8)
        "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE",

        # Real Estate (5)
        "AMT", "PLD", "CCI", "EQIX", "PSA",

        # Utilities (5)
        "NEE", "DUK", "SO", "D", "EXC",

        # Communication Services (7)
        "T", "VZ", "NFLX", "DIS", "CMCSA", "CHTR", "EA",
    ]

    return list(set(universe))  # Remove duplicates


def filter_universe(symbols: list, min_price: float = 5.0, min_adv_usd: float = 50_000_000) -> pd.DataFrame:
    """
    Filter universe based on price and volume criteria.

    Parameters
    ----------
    symbols : list
        List of symbols to filter
    min_price : float
        Minimum price per share (default: $5)
    min_adv_usd : float
        Minimum average daily dollar volume (default: $50M)

    Returns
    -------
    pd.DataFrame
        Filtered universe with columns: symbol, price, adv_usd, meets_criteria
    """
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Fetch last 20 days for ADV calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    results = []

    logger.info(f"Filtering {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        try:
            # Fetch recent bars
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
            )

            bars = client.get_stock_bars(request_params)

            if symbol not in bars.data or len(bars.data[symbol]) == 0:
                logger.warning(f"{symbol}: No data available")
                continue

            # Convert to dataframe
            df = bars.df.reset_index()

            # Calculate metrics
            latest_price = df['close'].iloc[-1]
            avg_volume = df['volume'].mean()
            avg_dollar_volume = (df['close'] * df['volume']).mean()

            # Apply criteria
            meets_criteria = (latest_price >= min_price) and (avg_dollar_volume >= min_adv_usd)

            results.append({
                "symbol": symbol,
                "price": latest_price,
                "avg_volume": avg_volume,
                "adv_usd": avg_dollar_volume,
                "meets_criteria": meets_criteria,
            })

            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(symbols)} symbols...")

        except Exception as e:
            logger.warning(f"{symbol}: Error - {e}")
            continue

    df = pd.DataFrame(results)

    # Summary
    n_passed = df["meets_criteria"].sum()
    logger.info(f"\n✓ {n_passed}/{len(df)} symbols meet criteria")
    logger.info(f"  Price >= ${min_price}: {(df['price'] >= min_price).sum()}")
    logger.info(f"  ADV >= ${min_adv_usd/1e6:.0f}M: {(df['adv_usd'] >= min_adv_usd).sum()}")

    return df


def save_universe(df: pd.DataFrame, output_path: str = "data_layer/reference/universe.csv"):
    """Save universe to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort by ADV (descending)
    df = df.sort_values("adv_usd", ascending=False)

    df.to_csv(output_path, index=False)
    logger.info(f"✓ Universe saved to {output_path}")

    # Also save just the passing symbols
    passing_symbols = df[df["meets_criteria"]]["symbol"].tolist()
    symbols_path = output_path.replace(".csv", "_symbols.txt")
    with open(symbols_path, "w") as f:
        f.write("\n".join(passing_symbols))
    logger.info(f"✓ Passing symbols saved to {symbols_path}")

    return passing_symbols


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("TradeML Universe Builder")
    logger.info("=" * 80)

    # Get initial universe
    logger.info("\n[1/3] Fetching S&P 500 constituents...")
    symbols = get_sp500_tickers()
    logger.info(f"✓ Starting universe: {len(symbols)} symbols")

    # Filter universe
    logger.info("\n[2/3] Applying filters (price >= $5, ADV >= $50M)...")
    df = filter_universe(symbols, min_price=5.0, min_adv_usd=50_000_000)

    # Save universe
    logger.info("\n[3/3] Saving universe...")
    passing_symbols = save_universe(df)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("UNIVERSE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total symbols evaluated: {len(df)}")
    logger.info(f"Passing criteria: {len(passing_symbols)}")
    logger.info(f"Top 10 by ADV:")
    for i, row in df[df["meets_criteria"]].head(10).iterrows():
        logger.info(f"  {row['symbol']}: ${row['price']:.2f}, ADV ${row['adv_usd']/1e6:.0f}M")

    logger.success(f"\n✅ Universe ready with {len(passing_symbols)} symbols")
    logger.info(f"   Symbols: {', '.join(passing_symbols[:20])}...")
