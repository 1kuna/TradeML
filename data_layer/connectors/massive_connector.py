"""
Massive.com connector - free-tier friendly, 5 req/min limit.

Implements a small subset of endpoints used as supplemental sources:
 - Aggregates (day/minute) for equities bars
 - Reference splits and dividends
 - Reference tickers (active)
 - Market status (now)
 - Options contracts and aggregates

Free tier limits (as of 2025):
 - 5 requests per minute
 - 2 years historical data for minute-level granularity
 - End-of-day equities, forex, and crypto included

All methods are best-effort and return empty DataFrames on errors. Use
BudgetManager to govern daily request counts under the 'massive' vendor.

See: https://massive.com/docs/rest/stocks
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError


class MassiveConnector(BaseConnector):
    """Connector for Massive.com market data API."""

    API_URL = "https://api.massive.com"

    # Free tier historical data limit
    FREE_TIER_HISTORY_DAYS = 730  # ~2 years

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_sec: float = 0.08,
    ):
        api_key = api_key or os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise ConnectorError("Massive API key not found. Set MASSIVE_API_KEY in .env")

        super().__init__(
            source_name="massive",
            api_key=api_key,
            base_url=self.API_URL,
            rate_limit_per_sec=rate_limit_per_sec,
        )

    def _auth_params(self) -> Dict[str, str]:
        return {"apiKey": self.api_key}

    def _fetch_raw(self, **kwargs):  # unused abstract
        return {}

    def _transform(self, raw_data, **kwargs) -> pd.DataFrame:  # unused abstract
        return pd.DataFrame()

    # -------- Aggregates --------
    def fetch_aggregates(self, symbol: str, start_date: date, end_date: date, timespan: str = "day") -> pd.DataFrame:
        """Fetch aggregates for a symbol between dates (inclusive). timespan in {'day','minute'}.

        Free tier limits:
          - 2 years of historical minute data
          - EOD data may have longer history

        Returns columns: date, symbol, open, high, low, close, volume
        """
        ts = "day" if timespan not in ("minute", "day") else timespan

        # Clamp to free tier history limit for minute data
        if ts == "minute":
            earliest = date.today() - timedelta(days=self.FREE_TIER_HISTORY_DAYS)
            if start_date < earliest:
                logger.debug(f"Clamping start_date from {start_date} to {earliest} (free tier limit)")
                start_date = earliest

        url = f"{self.API_URL}/v2/aggs/ticker/{symbol}/range/1/{ts}/{start_date.isoformat()}/{end_date.isoformat()}"
        try:
            r = self._get(url, params=self._auth_params())
            data = r.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            if not results:
                return pd.DataFrame()
            rows = []
            for it in results:
                # Massive uses epoch millis in 't'
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
            logger.warning(f"Massive aggregates fetch failed for {symbol} {start_date}->{end_date} {timespan}: {e}")
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
            logger.warning(f"Massive splits fetch failed for {symbol}: {e}")
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
            logger.warning(f"Massive dividends fetch failed for {symbol}: {e}")
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
            logger.warning(f"Massive tickers fetch failed: {e}")
            return pd.DataFrame(), None

    def market_status_now(self) -> Optional[Dict]:
        url = f"{self.API_URL}/v1/marketstatus/now"
        try:
            r = self._get(url, params=self._auth_params())
            return r.json()
        except Exception as e:
            logger.warning(f"Massive market status failed: {e}")
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
            logger.warning(f"Massive options contracts fetch failed for {underlying_ticker}: {e}")
            return pd.DataFrame(), None

    def fetch_option_aggregates(self, option_ticker: str, start_date: date, end_date: date, multiplier: int = 1, timespan: str = "day") -> pd.DataFrame:
        """
        GET /v3/aggs/ticker/{option_ticker}/range/{multiplier}/{timespan}/{from}/{to}

        Free tier allows ~2 years back — clamp start_date accordingly.
        Returns columns: date, option_ticker, open, high, low, close, volume
        """
        # Clamp to free tier history limit
        earliest = date.today() - timedelta(days=self.FREE_TIER_HISTORY_DAYS)
        if start_date < earliest:
            logger.debug(f"Clamping option start_date from {start_date} to {earliest} (free tier limit)")
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
            logger.warning(f"Massive option aggregates failed for {option_ticker}: {e}")
            return pd.DataFrame()
