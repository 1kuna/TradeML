"""
Alpaca Markets connector for equities bars (REST API only).

Free tier: Unlimited market data (paper trading account required)
API Docs: https://docs.alpaca.markets/docs/market-data

Supports:
- Historical bars (1Min, 5Min, 15Min, 1Hour, 1Day)
- Point-in-time safe (no forward-looking adjustments)
"""

import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd
import requests
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


# Valid timeframes for Alpaca REST API
VALID_TIMEFRAMES = {"1Min", "5Min", "15Min", "1Hour", "1Day"}


class AlpacaConnector(BaseConnector):
    """
    Connector for Alpaca Markets historical data via REST API.

    Free tier provides:
    - Historical bars (minute through daily)
    - Real-time quotes and trades (paper account)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        rate_limit_per_sec: float = 3.25,  # ~195 rpm (97.5% of 200, aggressive)
    ):
        """
        Initialize Alpaca connector.

        Args:
            api_key: Alpaca API key (or from environment ALPACA_API_KEY)
            secret_key: Alpaca secret key (or from environment ALPACA_SECRET_KEY)
            rate_limit_per_sec: Max requests per second
        """
        # Get credentials from env if not provided
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ConnectorError(
                "Alpaca API credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        super().__init__(
            source_name="alpaca",
            api_key=api_key,
            base_url="https://data.alpaca.markets",
            rate_limit_per_sec=rate_limit_per_sec,
        )

        self.secret_key = secret_key
        logger.info("Alpaca connector initialized (REST only)")

    def _fetch_raw(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1Day",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch raw bars from Alpaca REST API.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')

        Returns:
            Dict mapping symbol -> list of bar dicts
        """
        if timeframe not in VALID_TIMEFRAMES:
            raise ConnectorError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid: {list(VALID_TIMEFRAMES)}"
            )

        url = f"{self.base_url}/v2/stocks/bars"
        params = {
            "timeframe": timeframe,
            "start": f"{start_date.isoformat()}T00:00:00Z",
            "end": f"{end_date.isoformat()}T23:59:59Z",
            "feed": "sip",
            "symbols": ",".join(symbols),
            "limit": 10000,
        }
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        bucket: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
        page_token: Optional[str] = None

        while True:
            p = dict(params)
            if page_token:
                p["page_token"] = page_token

            try:
                resp = self._get(url, params=p, headers=headers)
            except requests.exceptions.RequestException as rexc:
                raise ConnectorError(f"Failed to fetch data from Alpaca: {rexc}")

            data = resp.json()
            bars = data.get("bars") or {}

            def _append_bar(sym: str, bar: Dict[str, Any]) -> None:
                if not isinstance(bar, dict):
                    return
                bucket.setdefault(sym, []).append(
                    {
                        "timestamp": bar.get("t") or bar.get("timestamp"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "vwap": bar.get("vw"),
                        "volume": bar.get("v") or bar.get("volume"),
                        "trade_count": bar.get("n") or bar.get("trade_count"),
                    }
                )

            if isinstance(bars, dict):
                if bars:
                    logger.debug(f"Alpaca bars symbols: {list(bars.keys())[:5]}")
                for sym, bar_list in bars.items():
                    if not bar_list:
                        continue
                    for bar in bar_list:
                        _append_bar(sym, bar)
            elif isinstance(bars, list):
                if bars:
                    logger.debug(f"Alpaca bars list length={len(bars)}")
                for bar in bars:
                    if not isinstance(bar, dict):
                        continue
                    sym = bar.get("S") or bar.get("symbol")
                    if not sym:
                        continue
                    _append_bar(sym, bar)
            else:
                logger.warning(f"Unexpected Alpaca bars payload type: {type(bars).__name__}")

            page_token = data.get("next_page_token")
            if not page_token:
                break

        # Remove symbols with no data
        return {k: v for k, v in bucket.items() if v}

    def _transform(
        self,
        raw_data: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform Alpaca bars to our schema.

        Args:
            raw_data: Dict mapping symbol -> list of bar dicts
            **kwargs: Additional context

        Returns:
            DataFrame conforming to EQUITY_BARS_SCHEMA
        """
        rows = []

        for symbol, bars in raw_data.items():
            if not bars:
                continue

            for bar in bars:
                ts = pd.to_datetime(bar["timestamp"])

                row = {
                    "date": ts.date(),
                    "symbol": symbol,
                    "session_id": ts.strftime("%Y%m%d"),
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "vwap": bar.get("vwap"),
                    "volume": bar["volume"],
                    "nbbo_spread": None,  # Not provided by Alpaca bars
                    "trades": bar.get("trade_count"),
                    "imbalance": None,  # Not provided
                }
                rows.append(row)

        if not rows:
            logger.warning("No data returned from Alpaca")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Ensure date column is proper type
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Sort by date and symbol
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        return df

    def fetch_bars(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch and transform stock bars.

        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            timeframe: Bar timeframe

        Returns:
            DataFrame with bars and metadata
        """
        logger.info(
            f"Fetching {timeframe} bars for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Fetch in batches to avoid hitting limits
        BATCH_SIZE = 100
        all_data = []

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            logger.debug(f"Fetching batch {i // BATCH_SIZE + 1} ({len(batch)} symbols)")

            try:
                self._rate_limit()
            except Exception:
                pass

            raw_data = self._fetch_raw(
                symbols=batch,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )

            df = self._transform(raw_data)

            if not df.empty:
                source_uri = f"alpaca://{timeframe}/{start_date}/{end_date}"
                df = self._add_metadata(df, source_uri=source_uri)
                all_data.append(df)

        if not all_data:
            logger.warning("No data fetched from Alpaca")
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)

        logger.info(f"Fetched {len(result)} bars for {len(symbols)} symbols")
        return result

    def fetch_universe(
        self,
        min_price: float = 5.0,
        min_volume: int = 1_000_000,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[str]:
        """
        Get tradable universe based on price and volume filters.

        Args:
            min_price: Minimum stock price
            min_volume: Minimum average daily volume
            start_date: Start date for volume calculation
            end_date: End date for volume calculation

        Returns:
            List of symbols meeting criteria
        """
        logger.info(
            f"Building universe with min_price={min_price}, "
            f"min_volume={min_volume}"
        )

        # Starter universe (top US equities)
        # TODO: Implement full universe construction with ADV calculation
        starter_universe = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "UNH", "JNJ", "V", "XOM", "WMT", "JPM", "MA", "PG", "AVGO", "HD",
            "CVX", "MRK", "ABBV", "COST", "PEP", "KO", "TMO", "BAC", "CSCO",
            "ACN", "MCD", "ADBE", "LLY", "ABT", "DIS", "VZ", "NKE", "CMCSA",
            "WFC", "INTC", "NFLX", "AMD", "CRM", "PM", "TXN", "NEE", "DHR",
            "UPS", "ORCL", "QCOM", "RTX", "HON", "BMY"
        ]

        logger.info(f"Starter universe: {len(starter_universe)} symbols")
        return starter_universe


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch Alpaca market data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1Day")
    parser.add_argument("--output", type=str, default="data_layer/raw/equities_bars")

    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    connector = AlpacaConnector()

    df = connector.fetch_bars(
        symbols=args.symbols,
        start_date=start,
        end_date=end,
        timeframe=args.timeframe,
    )

    if not df.empty:
        output_path = f"{args.output}/alpaca_{args.timeframe}_{start}_{end}.parquet"
        connector.write_parquet(
            df,
            path=output_path,
            schema=get_schema(DataType.EQUITY_BARS),
            partition_cols=["date", "symbol"],
        )
        print(f"[OK] Wrote {len(df)} rows to {output_path}")
    else:
        print("[WARN] No data fetched")
