"""
Financial Modeling Prep (FMP) connector for delisting data and fundamentals.

Free tier: 250 requests/day
API Docs: https://financialmodelingprep.com/developer/docs/

Supports:
- Delisted companies
- Historical price data
- Financial statements
- Company profiles
"""

import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


class FMPConnector(BaseConnector):
    """
    Connector for Financial Modeling Prep data.

    Free tier provides:
    - Delisted companies list
    - Historical price data
    - Financial statements
    """

    API_URL = "https://financialmodelingprep.com/api/v3"
    STABLE_URL = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_sec: float = 0.5,  # push limits but gated by 250/day budget
    ):
        """
        Initialize FMP connector.

        Args:
            api_key: FMP API key (or from environment)
            rate_limit_per_sec: Max requests per second
        """
        api_key = api_key or os.getenv("FMP_API_KEY")

        if not api_key:
            raise ConnectorError(
                "FMP API key not found. Set FMP_API_KEY in .env"
            )

        super().__init__(
            source_name="fmp",
            api_key=api_key,
            base_url=self.API_URL,
            rate_limit_per_sec=rate_limit_per_sec,
        )

        logger.info("FMP connector initialized")

    def _fetch_raw(self, endpoint: str, **params) -> Any:
        """
        Fetch raw data from FMP API.

        Args:
            endpoint: API endpoint path
            **params: Query parameters

        Returns:
            Raw JSON response
        """
        url = f"{self.API_URL}/{endpoint}"

        api_params = {
            "apikey": self.api_key,
            **params
        }

        try:
            response = self._get(url, params=api_params)
            data = response.json()

            # Check for API errors
            if isinstance(data, dict) and "Error Message" in data:
                raise ConnectorError(f"FMP error: {data['Error Message']}")

            return data

        except Exception as e:
            logger.error(f"FMP API error: {e}")
            raise ConnectorError(f"Failed to fetch from FMP: {e}")

    def _fetch_stable(self, endpoint: str, **params) -> Any:
        """Fetch raw data from FMP stable API.

        Args:
            endpoint: Stable API endpoint path
            **params: Query parameters

        Returns:
            Raw JSON response
        """
        url = f"{self.STABLE_URL}/{endpoint}"
        api_params = {"apikey": self.api_key, **params}
        try:
            response = self._get(url, params=api_params)
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"FMP stable API error: {e}")
            raise ConnectorError(f"Failed to fetch from FMP stable: {e}")

    def _transform(self, raw_data: Any, **kwargs) -> pd.DataFrame:
        """
        Transform FMP response to our schema.

        Implemented by specific fetch methods.
        """
        raise NotImplementedError("Use specific fetch methods")

    def fetch_delisted_companies(self) -> pd.DataFrame:
        """
        Fetch list of delisted companies.

        Returns:
            DataFrame with delisted symbols
        """
        logger.info("Fetching delisted companies from FMP (stable)")

        raw_data = self._fetch_stable("delisted-companies")

        # Parse delisted data
        rows = []

        if isinstance(raw_data, list):
            for item in raw_data:
                row = {
                    "symbol": item.get("symbol"),
                    "delist_date": datetime.strptime(item["delistedDate"], "%Y-%m-%d").date() if item.get("delistedDate") else None,
                    "reason": item.get("reason", "unknown"),
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            source_uri = "fmp://delisted-companies"
            df = self._add_metadata(df, source_uri=source_uri)
            df = df.sort_values("delist_date", ascending=False).reset_index(drop=True)

        return df

    def fetch_historical_price(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical prices for {symbol}")

        params = {}
        if start_date:
            params["from"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["to"] = end_date.strftime("%Y-%m-%d")

        # Prefer stable light EOD if no explicit range requested; otherwise fall back to full and filter
        raw_data: Any
        if not start_date and not end_date:
            raw_data = self._fetch_stable("historical-price-eod/light", symbol=symbol)
        else:
            raw_data = self._fetch_raw(f"historical-price-full/{symbol}", **params)

        # Parse historical data
        rows = []
        historical = raw_data.get("historical", [])

        for bar in historical:
            row = {
                "date": datetime.strptime(bar["date"], "%Y-%m-%d").date(),
                "symbol": symbol,
                "session_id": bar["date"].replace("-", ""),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "vwap": float(bar.get("vwap", 0)) if bar.get("vwap") else None,
                "volume": int(bar["volume"]),
                "nbbo_spread": None,
                "trades": None,
                "imbalance": None,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            source_uri = f"fmp://historical-price/{symbol}"
            df = self._add_metadata(df, source_uri=source_uri)
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def fetch_eod_series(self, symbol: str, variant: str = "light") -> pd.DataFrame:
        """Fetch EOD OHLCV series from stable historical-price-eod (light/full).

        Args:
            symbol: Ticker
            variant: 'light' or 'full'
        Returns:
            DataFrame with columns date,symbol,session_id,open,high,low,close,volume
        """
        logger.info(f"FMP EOD {variant} for {symbol}")
        data = self._fetch_stable(f"historical-price-eod/{variant}", symbol=symbol)
        # Response may be a list or {'historical': [...]} depending on plan
        series = []
        if isinstance(data, dict):
            series = data.get("historical", []) or data.get("results", []) or []
        elif isinstance(data, list):
            series = data
        rows = []
        for bar in series:
            d = bar.get("date") or bar.get("Date")
            if not d:
                continue
            try:
                dt = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            rows.append({
                "date": dt,
                "symbol": symbol,
                "session_id": dt.strftime("%Y%m%d"),
                "open": float(bar.get("open") or bar.get("Open") or 0.0),
                "high": float(bar.get("high") or bar.get("High") or 0.0),
                "low": float(bar.get("low") or bar.get("Low") or 0.0),
                "close": float(bar.get("close") or bar.get("Close") or 0.0),
                "vwap": None,
                "volume": int(bar.get("volume") or bar.get("Volume") or 0),
                "nbbo_spread": None,
                "trades": None,
                "imbalance": None,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = self._add_metadata(df, source_uri=f"fmp://historical-price-eod/{variant}/{symbol}")
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def fetch_eod_day(self, symbol: str, day: date) -> pd.DataFrame:
        """Fetch single EOD row for a given day using stable light series."""
        df = self.fetch_eod_series(symbol, variant="light")
        if df.empty:
            return df
        sub = df[df["date"] == day]
        return sub.reset_index(drop=True)

    def fetch_company_profile(self, symbol: str) -> Dict:
        """
        Fetch company profile.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with company info
        """
        logger.info(f"Fetching company profile for {symbol}")

        raw_data = self._fetch_raw(f"profile/{symbol}")

        if isinstance(raw_data, list) and len(raw_data) > 0:
            return raw_data[0]

        return {}

    def fetch_available_symbols(self, exchange: str = "NYSE,NASDAQ") -> pd.DataFrame:
        """
        Fetch all available symbols for exchanges.

        Args:
            exchange: Exchange codes (comma-separated)

        Returns:
            DataFrame with symbol list
        """
        logger.info(f"Fetching symbols for {exchange}")

        raw_data = self._fetch_raw("stock/list")

        # Filter by exchange
        rows = []
        if isinstance(raw_data, list):
            for item in raw_data:
                if item.get("exchangeShortName") in exchange.split(","):
                    rows.append({
                        "symbol": item.get("symbol"),
                        "name": item.get("name"),
                        "exchange": item.get("exchangeShortName"),
                        "type": item.get("type"),
                        "price": item.get("price"),
                    })

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values("symbol").reset_index(drop=True)

        return df


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch FMP market data")
    parser.add_argument("--action", type=str,
                        choices=["delisted", "price", "symbols"],
                        default="delisted",
                        help="Action to perform")
    parser.add_argument("--symbol", type=str, help="Symbol for price data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data_layer/reference")

    args = parser.parse_args()

    connector = FMPConnector()

    if args.action == "delisted":
        # Fetch delisted companies
        df = connector.fetch_delisted_companies()
        if not df.empty:
            output_path = f"{args.output}/delistings_fmp.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.DELISTINGS),
            )
            print(f"[OK] Fetched {len(df)} delisted companies to {output_path}")
            print(f"\nMost recent delistings:")
            print(df.head(10)[["symbol", "delist_date", "reason"]])
        else:
            print("[WARN] No delisted companies found")

    elif args.action == "symbols":
        # Fetch available symbols
        df = connector.fetch_available_symbols()
        if not df.empty:
            print(f"[OK] Found {len(df)} symbols")
            print(df.head(20))
        else:
            print("[WARN] No symbols found")

    elif args.action == "price" and args.symbol:
        # Fetch historical prices
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

        df = connector.fetch_historical_price(args.symbol, start, end)
        if not df.empty:
            output_path = f"{args.output}/{args.symbol}_fmp.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.EQUITY_BARS),
            )
            print(f"[OK] Fetched {len(df)} bars for {args.symbol} to {output_path}")
        else:
            print(f"[WARN] No price data found for {args.symbol}")
    else:
        print("Error: Invalid arguments. Use --action delisted|price|symbols")
