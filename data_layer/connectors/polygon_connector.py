"""
Polygon.io connector (free-tier friendly, 5 req/min suggested).

Implements a small subset of endpoints used as supplemental sources:
 - Aggregates (day/minute) for equities bars
 - Reference splits and dividends
 - Reference tickers (active)
 - Market status (now)

All methods are best-effort and return empty DataFrames on errors. Use
BudgetManager to govern daily request counts under the 'polygon' vendor.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError


class PolygonConnector(BaseConnector):
    API_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, rate_limit_per_sec: float = 0.08):
        api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ConnectorError("Polygon API key not found. Set POLYGON_API_KEY in .env")
        super().__init__(source_name="polygon", api_key=api_key, base_url=self.API_URL, rate_limit_per_sec=rate_limit_per_sec)

    def _auth_params(self) -> Dict[str, str]:
        return {"apiKey": self.api_key}

    def _fetch_raw(self, **kwargs):  # unused abstract
        return {}

    def _transform(self, raw_data, **kwargs) -> pd.DataFrame:  # unused abstract
        return pd.DataFrame()

    # -------- Aggregates --------
    def fetch_aggregates(self, symbol: str, start_date: date, end_date: date, timespan: str = "day") -> pd.DataFrame:
        """Fetch aggregates for a symbol between dates (inclusive). timespan in {'day','minute'}.

        Returns columns: date, symbol, open, high, low, close, volume
        """
        ts = "day" if timespan not in ("minute", "day") else timespan
        url = f"{self.API_URL}/v2/aggs/ticker/{symbol}/range/1/{ts}/{start_date.isoformat()}/{end_date.isoformat()}"
        try:
            r = self._get(url, params=self._auth_params())
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return pd.DataFrame()
            rows = []
            for it in results:
                # Polygon uses epoch millis in 't'
                dt = pd.to_datetime(int(it.get("t", 0)), unit="ms").date()
                rows.append({
                    "date": dt,
                    "symbol": symbol,
                    "open": float(it.get("o", float('nan'))),
                    "high": float(it.get("h", float('nan'))),
                    "low": float(it.get("l", float('nan'))),
                    "close": float(it.get("c", float('nan'))),
                    "volume": float(it.get("v", float('nan'))),
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=url)
        except Exception as e:
            logger.warning(f"Polygon aggregates fetch failed for {symbol} {start_date}->{end_date} {timespan}: {e}")
            return pd.DataFrame()

    # -------- Reference: splits & dividends --------
    def fetch_splits(self, symbol: str) -> pd.DataFrame:
        url = f"{self.API_URL}/v3/reference/splits"
        try:
            r = self._get(url, params={**self._auth_params(), "ticker": symbol, "limit": 1000})
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return pd.DataFrame()
            rows = []
            for it in results:
                exd = pd.to_datetime(it.get("execution_date") or it.get("ex_date")).date()
                ratio = it.get("split_to") / max(1.0, float(it.get("split_from", 1.0))) if it.get("split_to") else None
                rows.append({
                    "symbol": symbol,
                    "event_type": "split",
                    "ex_date": exd,
                    "ratio": float(ratio) if ratio else None,
                    "amount": None,
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=f"{url}?ticker={symbol}")
        except Exception as e:
            logger.warning(f"Polygon splits fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        url = f"{self.API_URL}/v3/reference/dividends"
        try:
            r = self._get(url, params={**self._auth_params(), "ticker": symbol, "limit": 1000})
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return pd.DataFrame()
            rows = []
            for it in results:
                exd = pd.to_datetime(it.get("ex_dividend_date") or it.get("ex_date")).date()
                pay = pd.to_datetime(it.get("pay_date") or it.get("payment_date")).date() if it.get("pay_date") or it.get("payment_date") else None
                amt = it.get("cash_amount") or it.get("amount")
                rows.append({
                    "symbol": symbol,
                    "event_type": "dividend",
                    "ex_date": exd,
                    "record_date": None,
                    "pay_date": pay,
                    "amount": float(amt) if amt is not None else None,
                    "ratio": None,
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=f"{url}?ticker={symbol}")
        except Exception as e:
            logger.warning(f"Polygon dividends fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    # -------- Reference: tickers & market status --------
    def list_active_tickers(self, cursor: Optional[str] = None, limit: int = 1000) -> Tuple[pd.DataFrame, Optional[str]]:
        url = f"{self.API_URL}/v3/reference/tickers"
        try:
            params = {**self._auth_params(), "market": "stocks", "active": "true", "limit": limit}
            if cursor:
                params["cursor"] = cursor
            r = self._get(url, params=params)
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            next_cursor = data.get("next_url") or data.get("next_cursor")
            if not results:
                return pd.DataFrame(), None
            rows = []
            for it in results:
                rows.append({
                    "ticker": it.get("ticker"),
                    "name": it.get("name"),
                    "currency": it.get("currency_name") or it.get("currency_symbol"),
                    "locale": it.get("locale"),
                    "primary_exchange": it.get("primary_exchange"),
                    "active": it.get("active"),
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=url), next_cursor
        except Exception as e:
            logger.warning(f"Polygon tickers fetch failed: {e}")
            return pd.DataFrame(), None

    def market_status_now(self) -> Optional[Dict]:
        url = f"{self.API_URL}/v1/marketstatus/now"
        try:
            r = self._get(url, params=self._auth_params())
            return r.json()
        except Exception as e:
            logger.warning(f"Polygon market status failed: {e}")
            return None

    # -------- Options (v3 reference + v3 aggregates custom-bars) --------
    def list_options_contracts(self, underlying_ticker: str, as_of: Optional[date] = None, limit: int = 1000, cursor: Optional[str] = None) -> tuple[pd.DataFrame, Optional[str]]:
        """
        GET /v3/reference/options/contracts — List options contracts for an underlying.

        Params:
          - underlying_ticker: Underlying equity/ETF symbol
          - as_of: Optional date filter (YYYY-MM-DD). Note: free plan supports last ~2 years.
          - limit: Page size
          - cursor: Pagination cursor
        Returns: (DataFrame, next_cursor)
        """
        url = f"{self.API_URL}/v3/reference/options/contracts"
        params: Dict[str, object] = {**self._auth_params(), "underlying_ticker": underlying_ticker, "limit": limit}
        if as_of is not None:
            params["as_of"] = as_of.isoformat()
        if cursor:
            params["cursor"] = cursor
        try:
            r = self._get(url, params=params)
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            next_cursor = data.get("next_url") or data.get("next_cursor")
            if not results:
                return pd.DataFrame(), None
            rows = []
            for it in results:
                rows.append({
                    "ticker": it.get("ticker"),
                    "underlier": it.get("underlying_ticker") or underlying_ticker,
                    "contract_type": it.get("contract_type"),
                    "strike_price": float(it.get("strike_price")) if it.get("strike_price") is not None else None,
                    "expiration_date": pd.to_datetime(it.get("expiration_date")).date() if it.get("expiration_date") else None,
                    "exercise_style": it.get("exercise_style"),
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=url), next_cursor
        except Exception as e:
            logger.warning(f"Polygon options contracts fetch failed for {underlying_ticker}: {e}")
            return pd.DataFrame(), None

    def fetch_option_aggregates(self, option_ticker: str, start_date: date, end_date: date, multiplier: int = 1, timespan: str = "day") -> pd.DataFrame:
        """
        GET /v3/aggs/ticker/{option_ticker}/range/{multiplier}/{timespan}/{from}/{to}

        Free plan allows ~2 years back — clamp start_date accordingly.
        Returns columns: date, option_ticker, open, high, low, close, volume
        """
        # Clamp to ~2 years back
        from datetime import timedelta as _TD
        try:
            earliest = date.today() - _TD(days=730)
        except Exception:
            earliest = date.today()
        if start_date < earliest:
            start_date = earliest
        tspan = timespan if timespan in ("minute", "hour", "day", "week", "month") else "day"
        url = f"{self.API_URL}/v3/aggs/ticker/{option_ticker}/range/{multiplier}/{tspan}/{start_date.isoformat()}/{end_date.isoformat()}"
        try:
            r = self._get(url, params=self._auth_params())
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return pd.DataFrame()
            rows = []
            for it in results:
                dt = pd.to_datetime(int(it.get("t", 0)), unit="ms").date()
                rows.append({
                    "date": dt,
                    "option_ticker": option_ticker,
                    "open": float(it.get("o", float('nan'))),
                    "high": float(it.get("h", float('nan'))),
                    "low": float(it.get("l", float('nan'))),
                    "close": float(it.get("c", float('nan'))),
                    "volume": float(it.get("v", float('nan'))),
                })
            df = pd.DataFrame(rows)
            return self._add_metadata(df, source_uri=url)
        except Exception as e:
            logger.warning(f"Polygon option aggregates failed for {option_ticker}: {e}")
            return pd.DataFrame()
