"""
Finnhub connector for options chains and market data.

Free tier: 60 API calls/minute
API Docs: https://finnhub.io/docs/api

Supports:
- Options chains (calls/puts)
- Quote data
- Company fundamentals
- Daily OHLC candles (free tier)
"""

import os
from datetime import datetime, date, timezone
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


class FinnhubConnector(BaseConnector):
    """
    Connector for Finnhub market data.

    Free tier provides:
    - Options chains with strikes and expiries
    - Real-time quotes
    - Company profiles
    """

    API_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_sec: float = 0.85,  # ~51 per minute (85% of 60)
    ):
        """
        Initialize Finnhub connector.

        Args:
            api_key: Finnhub API key (or from environment)
            rate_limit_per_sec: Max requests per second
        """
        api_key = api_key or os.getenv("FINNHUB_API_KEY")

        if not api_key:
            raise ConnectorError(
                "Finnhub API key not found. Set FINNHUB_API_KEY in .env"
            )

        super().__init__(
            source_name="finnhub",
            api_key=api_key,
            base_url=self.API_URL,
            rate_limit_per_sec=rate_limit_per_sec,
        )

        logger.info("Finnhub connector initialized")

    def _fetch_raw(self, endpoint: str, **params) -> Dict:
        """
        Fetch raw data from Finnhub API.

        Args:
            endpoint: API endpoint path
            **params: Query parameters

        Returns:
            Raw JSON response
        """
        url = f"{self.API_URL}/{endpoint}"

        api_params = {
            "token": self.api_key,
            **params
        }

        try:
            response = self._get(url, params=api_params)
            data = response.json()

            # Check for API errors
            if "error" in data:
                raise ConnectorError(f"Finnhub error: {data['error']}")

            return data

        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            raise ConnectorError(f"Failed to fetch from Finnhub: {e}")

    def _transform(self, raw_data: Dict, **kwargs) -> pd.DataFrame:
        """
        Transform Finnhub response to our schema.

        Implemented by specific fetch methods.
        """
        raise NotImplementedError("Use specific fetch methods")

    def fetch_options_chain(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch options chain for a symbol.

        Args:
            symbol: Underlying symbol
            date: Expiry date (YYYY-MM-DD) or None for all

        Returns:
            DataFrame with options chain
        """
        logger.info(f"Fetching options chain for {symbol}")

        params = {"symbol": symbol}
        if date:
            params["date"] = date

        raw_data = self._fetch_raw("stock/option-chain", **params)

        # Parse options data
        rows = []
        options_data = raw_data.get("data", [])

        for opt in options_data:
            # Expiry can be epoch seconds or ISO string depending on plan/version
            exp_raw = opt.get("expirationDate")
            expiry = None
            if exp_raw:
                try:
                    # Epoch seconds
                    expiry = datetime.fromtimestamp(int(exp_raw)).date()
                except Exception:
                    try:
                        # ISO string
                        expiry = pd.to_datetime(exp_raw).date()
                    except Exception:
                        expiry = None

            row = {
                "underlier": symbol,
                "expiry": expiry,
                "strike": float(opt.get("strike")) if opt.get("strike") is not None else None,
                "cp_flag": "C" if str(opt.get("type", "")).lower().startswith("c") else "P",
                "bid": float(opt.get("bid", 0)) if opt.get("bid") is not None else None,
                "ask": float(opt.get("ask", 0)) if opt.get("ask") is not None else None,
                "bid_size": int(opt.get("bidSize", 0)) if opt.get("bidSize") is not None else None,
                "ask_size": int(opt.get("askSize", 0)) if opt.get("askSize") is not None else None,
                "nbbo_mid": None,  # Will compute
                "ts_ns": int(datetime.now().timestamp() * 1e9),
            }

            # Compute mid if bid/ask available
            if row["bid"] is not None and row["ask"] is not None:
                row["nbbo_mid"] = (row["bid"] + row["ask"]) / 2

            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            source_uri = f"finnhub://option-chain/{symbol}"
            df = self._add_metadata(df, source_uri=source_uri)
            df = df.sort_values(["expiry", "strike", "cp_flag"]).reset_index(drop=True)

        return df

    def fetch_quote(self, symbol: str) -> Dict:
        """
        Fetch real-time quote for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with quote data
        """
        logger.info(f"Fetching quote for {symbol}")

        raw_data = self._fetch_raw("quote", symbol=symbol)
        return raw_data

    def fetch_company_profile(self, symbol: str) -> Dict:
        """
        Fetch company profile/fundamentals.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with company info
        """
        logger.info(f"Fetching company profile for {symbol}")

        raw_data = self._fetch_raw("stock/profile2", symbol=symbol)
        return raw_data

    def fetch_options_expiries(self, symbol: str) -> List[date]:
        """
        Get available expiry dates for options.

        Args:
            symbol: Underlying symbol

        Returns:
            List of expiry dates
        """
        logger.info(f"Fetching options expiries for {symbol}")

        # Fetch chain without date filter to get all expiries
        df = self.fetch_options_chain(symbol)

        if df.empty:
            return []

        expiries = df["expiry"].dropna().unique().tolist()
        expiries.sort()

        return expiries

    def fetch_candle_daily(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV candles for an equity.

        Args:
            symbol: Equity ticker
            start_date: Inclusive start date
            end_date: Inclusive end date

        Returns:
            DataFrame matching EQUITY_BARS schema (or empty when no data).
        """
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        def _to_unix(dt: date) -> int:
            return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())

        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": _to_unix(start_date),
            "to": _to_unix(end_date),
        }

        raw = self._fetch_raw("stock/candle", **params)
        if not raw or raw.get("s") != "ok":
            logger.debug(f"Finnhub candle daily returned status={raw.get('s')} for {symbol}")
            return pd.DataFrame()

        closes = raw.get("c", [])
        opens = raw.get("o", [])
        highs = raw.get("h", [])
        lows = raw.get("l", [])
        volumes = raw.get("v", [])
        timestamps = raw.get("t", [])

        rows = []
        for idx, ts in enumerate(timestamps or []):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            rows.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "session_id": dt.strftime("%Y%m%d"),
                    "open": float(opens[idx]) if idx < len(opens) else float("nan"),
                    "high": float(highs[idx]) if idx < len(highs) else float("nan"),
                    "low": float(lows[idx]) if idx < len(lows) else float("nan"),
                    "close": float(closes[idx]) if idx < len(closes) else float("nan"),
                    "vwap": None,
                    "volume": int(volumes[idx]) if idx < len(volumes) else 0,
                    "nbbo_spread": None,
                    "trades": None,
                    "imbalance": None,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = self._add_metadata(
            df,
            source_uri=f"finnhub://stock/candle?symbol={symbol}&resolution=D",
        )
        df = df.sort_values("date").reset_index(drop=True)
        return df


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch Finnhub options data")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to fetch")
    parser.add_argument("--expiry", type=str, help="Expiry date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data_layer/raw/options_nbbo")

    args = parser.parse_args()

    connector = FinnhubConnector()

    # Fetch options chain
    df = connector.fetch_options_chain(args.symbol, date=args.expiry)

    if not df.empty:
        output_path = f"{args.output}/{args.symbol}_options.parquet"
        connector.write_parquet(
            df,
            path=output_path,
            schema=get_schema(DataType.OPTIONS_NBBO),
        )
        print(f"[OK] Fetched {len(df)} options contracts for {args.symbol} to {output_path}")

        # Print summary
        expiries = df["expiry"].unique()
        print(f"\nExpiries found: {len(expiries)}")
        for exp in sorted(expiries)[:5]:  # Show first 5
            count = len(df[df["expiry"] == exp])
            print(f"  {exp}: {count} contracts")
    else:
        print(f"[WARN] No options data found for {args.symbol}")
