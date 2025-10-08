"""
Alpha Vantage connector for corporate actions and reference data.

Free tier: 25 requests/day (generous for periodic updates)
API Docs: https://www.alphavantage.co/documentation/

Supports:
- Corporate actions (splits, dividends)
- Listing status (active/delisted)
- Company overview
- Historical options with greeks (daily snapshots)
"""

import math
import os
from datetime import datetime, date
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


class AlphaVantageConnector(BaseConnector):
    """
    Connector for Alpha Vantage corporate actions and reference data.

    Free tier provides:
    - Splits and dividends history
    - Listing status (active/delisted)
    - Company fundamentals
    """

    API_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_sec: float = 0.2,  # AV: 25/day; rpm governed by 429/backoff
    ):
        """
        Initialize Alpha Vantage connector.

        Args:
            api_key: Alpha Vantage API key (or from environment)
            rate_limit_per_sec: Max requests per second
        """
        api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            raise ConnectorError(
                "Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY in .env"
            )

        super().__init__(
            source_name="alpha_vantage",
            api_key=api_key,
            base_url=self.API_URL,
            rate_limit_per_sec=rate_limit_per_sec,
        )

        logger.info("Alpha Vantage connector initialized")

    def _fetch_raw(self, function: str, symbol: Optional[str] = None, **kwargs) -> Dict:
        """
        Fetch raw data from Alpha Vantage API.

        Args:
            function: API function name
            symbol: Ticker symbol (if needed)
            **kwargs: Additional parameters

        Returns:
            Raw JSON response
        """
        params = {
            "function": function,
            "apikey": self.api_key,
            **kwargs
        }

        if symbol:
            params["symbol"] = symbol

        try:
            response = self._get(self.API_URL, params=params)
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise ConnectorError(f"Alpha Vantage error: {data['Error Message']}")

            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")

            return data

        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
            raise ConnectorError(f"Failed to fetch from Alpha Vantage: {e}")

    def _transform(self, raw_data: Dict, **kwargs) -> pd.DataFrame:
        """
        Transform Alpha Vantage response to our schema.

        Implemented by specific fetch methods.
        """
        raise NotImplementedError("Use specific fetch methods (fetch_splits, fetch_dividends, etc.)")

    def fetch_splits(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock split history.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with split events
        """
        logger.info(f"Fetching splits for {symbol}")

        raw_data = self._fetch_raw(function="SPLITS", symbol=symbol)

        # Parse splits data
        rows = []
        splits_data = raw_data.get("data", [])

        for split in splits_data:
            row = {
                "symbol": symbol,
                "event_type": "split",
                "ex_date": datetime.strptime(split["effective_date"], "%Y-%m-%d").date(),
                "record_date": None,  # Not provided by AV
                "pay_date": None,
                "ratio": float(split["split_factor"]),  # e.g., "2:1" -> 2.0
                "amount": None,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            source_uri = f"alpha_vantage://SPLITS/{symbol}"
            df = self._add_metadata(df, source_uri=source_uri)

        return df

    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        """
        Fetch dividend history.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with dividend events
        """
        logger.info(f"Fetching dividends for {symbol}")

        raw_data = self._fetch_raw(function="DIVIDENDS", symbol=symbol)

        # Parse dividends data
        rows = []
        div_data = raw_data.get("data", [])

        for div in div_data:
            row = {
                "symbol": symbol,
                "event_type": "dividend",
                "ex_date": datetime.strptime(div["ex_date"], "%Y-%m-%d").date(),
                "record_date": datetime.strptime(div["record_date"], "%Y-%m-%d").date() if div.get("record_date") else None,
                "pay_date": datetime.strptime(div["payment_date"], "%Y-%m-%d").date() if div.get("payment_date") else None,
                "ratio": None,
                "amount": float(div["amount"]),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            source_uri = f"alpha_vantage://DIVIDENDS/{symbol}"
            df = self._add_metadata(df, source_uri=source_uri)

        return df

    def fetch_listing_status(
        self,
        state: str = "delisted",
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch listing status for all symbols.

        Args:
            state: 'active' or 'delisted'
            date: Optional date filter (YYYY-MM-DD)

        Returns:
            DataFrame with listing status
        """
        logger.info(f"Fetching {state} symbols...")

        params = {"state": state}
        if date:
            params["date"] = date

        raw_data = self._fetch_raw(function="LISTING_STATUS", **params)

        # Parse CSV response (Alpha Vantage returns CSV for this endpoint)
        # Expected columns: symbol, name, exchange, assetType, ipoDate, delistingDate, status
        import io

        if isinstance(raw_data, dict):
            # If JSON error response
            logger.warning("Unexpected JSON response for LISTING_STATUS")
            return pd.DataFrame()

        # Parse CSV
        csv_data = raw_data if isinstance(raw_data, str) else raw_data.text
        df = pd.read_csv(io.StringIO(csv_data))

        # Transform to delistings schema if needed
        if state == "delisted" and not df.empty:
            delisting_rows = []
            for _, row in df.iterrows():
                delisting_rows.append({
                    "symbol": row["symbol"],
                    "delist_date": datetime.strptime(row["delistingDate"], "%Y-%m-%d").date() if pd.notna(row.get("delistingDate")) else None,
                    "reason": row.get("status", "unknown"),
                })

            df_delisting = pd.DataFrame(delisting_rows)
            source_uri = f"alpha_vantage://LISTING_STATUS/delisted"
            df_delisting = self._add_metadata(df_delisting, source_uri=source_uri)
            return df_delisting

        return df

    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """
        Fetch all corporate actions (splits + dividends) for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Combined DataFrame with all corporate actions
        """
        splits = self.fetch_splits(symbol)
        dividends = self.fetch_dividends(symbol)

        if splits.empty and dividends.empty:
            return pd.DataFrame()

        # Combine and sort by ex_date
        combined = pd.concat([splits, dividends], ignore_index=True)
        combined = combined.sort_values("ex_date").reset_index(drop=True)

        return combined

    def fetch_overview(self, symbol: str) -> Dict:
        """
        Fetch company overview/fundamentals.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with company info
        """
        logger.info(f"Fetching overview for {symbol}")

        raw_data = self._fetch_raw(function="OVERVIEW", symbol=symbol)
        return raw_data

    def fetch_historical_options(
        self,
        symbol: str,
        expiry: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical options chain snapshots with greeks.

        Args:
            symbol: Underlying ticker
            expiry: Optional expiry filter (YYYY-MM-DD)

        Returns:
            DataFrame aligned to OPTIONS_IV schema (daily snapshot rows).
        """
        params: Dict[str, Any] = {}
        if expiry:
            params["expiration"] = expiry

        raw = self._fetch_raw(function="HISTORICAL_OPTIONS", symbol=symbol, **params)
        data = raw.get("data") or raw.get("historical") or []
        underlying_price = raw.get("underlying_price") or raw.get("underlyingPrice")

        rows = []

        def _as_float(val, default=None):
            if val is None or val == "":
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                try:
                    return float(str(val).replace("%", "")) / 100.0 if isinstance(val, str) and val.endswith("%") else float(str(val).replace(",", ""))
                except Exception:
                    return default

        def _parse_contract(contract: Dict[str, Any], expiry_date: Optional[date], snapshot_date: Optional[date], cp: str):
            strike = _as_float(contract.get("strike") or contract.get("strikePrice"), default=math.nan)
            iv = _as_float(contract.get("impliedVolatility"), default=math.nan)
            bid = _as_float(contract.get("bid"))
            ask = _as_float(contract.get("ask"))
            last_price = _as_float(contract.get("lastPrice") or contract.get("price"))
            nbbo_mid = None
            if bid is not None and ask is not None and not math.isnan(bid) and not math.isnan(ask):
                nbbo_mid = (bid + ask) / 2.0
            elif last_price is not None and not math.isnan(last_price):
                nbbo_mid = last_price
            else:
                nbbo_mid = 0.0

            und_px = _as_float(
                contract.get("underlyingPrice")
                or contract.get("underlying_price")
                or underlying_price,
                default=0.0,
            )

            expiry_dt = pd.to_datetime(expiry_date or contract.get("expirationDate") or contract.get("expiration"), errors="coerce")
            if pd.isna(expiry_dt):
                expiry_dt = pd.NaT
            snapshot_dt = pd.to_datetime(
                snapshot_date
                or contract.get("lastTradeDate")
                or contract.get("exerciseDate")
                or contract.get("dataDate"),
                errors="coerce",
            )
            if pd.isna(snapshot_dt):
                snapshot_dt = pd.Timestamp.utcnow().normalize()

            days_to_expiry = 0.0
            if pd.notna(expiry_dt):
                diff = (expiry_dt - snapshot_dt).days
                days_to_expiry = max(diff, 0) / 365.0

            is_itm = bool(contract.get("inTheMoney") or contract.get("inTheMoneyFlag") or contract.get("inTheMoneyValue"))
            delta = _as_float(contract.get("delta"))
            gamma = _as_float(contract.get("gamma"))
            theta = _as_float(contract.get("theta"))
            vega = _as_float(contract.get("vega"))
            rho = _as_float(contract.get("rho"))

            row = {
                "date": snapshot_dt.date(),
                "underlier": symbol,
                "expiry": expiry_dt.date() if pd.notna(expiry_dt) else snapshot_dt.date(),
                "strike": strike,
                "cp_flag": cp,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
                "underlying_price": und_px if und_px is not None and not math.isnan(und_px) else 0.0,
                "risk_free_rate": _as_float(contract.get("riskFreeRate"), default=0.0) or 0.0,
                "dividend_yield": _as_float(contract.get("dividendYield"), default=None),
                "time_to_expiry": days_to_expiry,
                "nbbo_mid": nbbo_mid if nbbo_mid is not None and not math.isnan(nbbo_mid) else 0.0,
                "is_itm": bool(is_itm),
                "iv_valid": bool(iv is not None and not math.isnan(iv)),
                "is_crossed": bool(bid is not None and ask is not None and bid > ask),
            }
            return row

        if isinstance(data, list):
            for block in data:
                expiry_block = pd.to_datetime(block.get("expirationDate") or block.get("expiration"), errors="coerce")
                snapshot_block = pd.to_datetime(block.get("date") or block.get("recordDate"), errors="coerce")
                for cp_key, cp_flag in (("calls", "C"), ("puts", "P")):
                    contracts = block.get(cp_key) or []
                    for contract in contracts:
                        rows.append(_parse_contract(contract, expiry_block.date() if pd.notna(expiry_block) else None, snapshot_block.date() if pd.notna(snapshot_block) else None, cp_flag))
                # Some payloads flatten to "options" array with type indicator
                options_flat = block.get("options") or []
                for contract in options_flat:
                    cp_flag = str(contract.get("type") or contract.get("optionType") or "C").upper()
                    cp_flag = "P" if cp_flag.startswith("P") else "C"
                    rows.append(_parse_contract(contract, expiry_block.date() if pd.notna(expiry_block) else None, snapshot_block.date() if pd.notna(snapshot_block) else None, cp_flag))
        elif isinstance(data, dict):
            # Single expiry return style: {"calls": [...], "puts": [...]}
            expiry_block = pd.to_datetime(data.get("expirationDate") or data.get("expiration"), errors="coerce")
            for cp_key, cp_flag in (("calls", "C"), ("puts", "P")):
                for contract in data.get(cp_key, []) or []:
                    rows.append(_parse_contract(contract, expiry_block.date() if pd.notna(expiry_block) else None, None, cp_flag))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        expected_cols = [
            "date",
            "underlier",
            "expiry",
            "strike",
            "cp_flag",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "underlying_price",
            "risk_free_rate",
            "dividend_yield",
            "time_to_expiry",
            "nbbo_mid",
            "is_itm",
            "iv_valid",
            "is_crossed",
        ]
        defaults = {
            "iv": math.nan,
            "delta": math.nan,
            "gamma": math.nan,
            "theta": math.nan,
            "vega": math.nan,
            "rho": math.nan,
            "dividend_yield": None,
        }
        for col in expected_cols:
            if col not in df.columns:
                df[col] = defaults.get(col, 0.0 if col in {"underlying_price", "risk_free_rate", "time_to_expiry", "nbbo_mid"} else None)
        df = df[expected_cols]

        df = self._add_metadata(
            df,
            source_uri=f"alpha_vantage://HISTORICAL_OPTIONS/{symbol}{'/' + expiry if expiry else ''}",
        )
        return df

# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage corporate actions")
    parser.add_argument("--symbol", type=str, help="Symbol to fetch")
    parser.add_argument("--action", type=str, choices=["splits", "dividends", "all", "delisted"],
                        default="all", help="Action type")
    parser.add_argument("--output", type=str, default="data_layer/reference/corp_actions")

    args = parser.parse_args()

    connector = AlphaVantageConnector()

    if args.action == "delisted":
        # Fetch delisted symbols
        df = connector.fetch_listing_status(state="delisted")
        if not df.empty:
            output_path = f"{args.output}/delistings_av.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.DELISTINGS),
            )
            print(f"[OK] Fetched {len(df)} delisted symbols to {output_path}")
    elif args.symbol:
        # Fetch corporate actions for symbol
        if args.action == "splits":
            df = connector.fetch_splits(args.symbol)
        elif args.action == "dividends":
            df = connector.fetch_dividends(args.symbol)
        else:  # all
            df = connector.fetch_corporate_actions(args.symbol)

        if not df.empty:
            output_path = f"{args.output}/{args.symbol}_corp_actions.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.CORP_ACTIONS),
            )
            print(f"[OK] Fetched {len(df)} corporate actions for {args.symbol} to {output_path}")
        else:
            print(f"[WARN] No corporate actions found for {args.symbol}")
    else:
        print("Error: --symbol required for splits/dividends/all")
