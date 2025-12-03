"""
Alpaca Markets connector for equities bars and real-time data.

Free tier: Unlimited market data (paper trading account required)
API Docs: https://docs.alpaca.markets/docs/market-data

Supports:
- Historical bars (1Min, 5Min, 15Min, 1Hour, 1Day)
- Real-time streaming (development)
- Point-in-time safe (no forward-looking adjustments)
"""

import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import (
    StockBarsRequest,
    OptionBarsRequest,
    OptionTradesRequest,
    OptionChainRequest,
    CorporateActionsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


class AlpacaConnector(BaseConnector):
    """
    Connector for Alpaca Markets historical and real-time data.

    Free tier provides:
    - Historical bars (minute through daily)
    - Real-time quotes and trades (paper account)
    - Corporate actions feed
    """

    TIMEFRAME_MAP = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        rate_limit_per_sec: float = 2.83,  # ~170 rpm (85% of 200)
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

        # Initialize Alpaca client
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.use_sdk_bars = os.getenv("ALPACA_BARS_USE_SDK", "0").lower() in {"1", "true", "yes", "on"}
        # Initialize options client (for options bars/trades/chain when entitled)
        try:
            self.opt_client = OptionHistoricalDataClient(api_key, secret_key)
        except Exception as e:
            logger.warning(f"Alpaca options client unavailable: {e}")
            self.opt_client = None
        # Corporate actions client
        try:
            self.ca_client = CorporateActionsClient(api_key, secret_key)
        except Exception as e:
            logger.warning(f"Alpaca corporate actions client unavailable: {e}")
            self.ca_client = None
        self.secret_key = secret_key

        logger.info("Alpaca connector initialized")
        if not self.use_sdk_bars:
            logger.info("Alpaca bars will use REST path (ALPACA_BARS_USE_SDK=0)")
        else:
            logger.info("Alpaca bars will use SDK path (ALPACA_BARS_USE_SDK=1)")

    def _fetch_raw(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1Day",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch raw bars from Alpaca API.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')

        Returns:
            Raw API response dict
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ConnectorError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid: {list(self.TIMEFRAME_MAP.keys())}"
            )

        # Default to REST path because Alpaca SDK intermittently throws NameError
        if not self.use_sdk_bars:
            return self._fetch_raw_rest(symbols, start_date, end_date, timeframe)

        # Create request for SDK path when explicitly enabled
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.TIMEFRAME_MAP[timeframe],
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
            feed="iex",
        )

        try:
            # Fetch bars with timing logs to diagnose stalls in Alpaca SDK
            from time import time as _now

            t0 = _now()
            logger.debug(
                f"Alpaca get_stock_bars begin: tf={timeframe} symbols={len(symbols)} start={start_date} end={end_date}"
            )
            bars = self.client.get_stock_bars(request)
            dt = _now() - t0
            logger.debug(f"Alpaca get_stock_bars done in {dt:.2f}s: tf={timeframe}")
            return bars.dict()  # Convert to dict for processing
        except Exception as e:
            # Alpaca SDK occasionally throws NameError on internal json handling; fall back to REST
            emsg = str(e)
            if isinstance(e, NameError) or "local variable 'json'" in emsg:
                logger.error(f"Alpaca SDK NameError (json bug) detected; falling back to REST: {emsg}")
                return self._fetch_raw_rest(symbols, start_date, end_date, timeframe)
            logger.error(f"Alpaca API error: {e}")
            raise ConnectorError(f"Failed to fetch data from Alpaca: {e}")

    def _fetch_raw_rest(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        REST fallback when the official SDK misbehaves (e.g., NameError on json).
        Implements pagination and buckets bars by symbol to match SDK shape.
        """
        url = f"{self.base_url}/v2/stocks/bars"
        params = {
            "timeframe": timeframe,
            "start": f"{start_date.isoformat()}T00:00:00Z",
            "end": f"{end_date.isoformat()}T23:59:59Z",
            "feed": "iex",
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
                resp = self._get(url, params=p, headers=headers)  # type: ignore[attr-defined]
            except requests.exceptions.RequestException as rexc:
                raise ConnectorError(f"Failed to fetch data from Alpaca REST: {rexc}")
            data = resp.json()
            bars = data.get("bars", [])
            for bar in bars:
                sym = bar.get("S") or bar.get("symbol")
                if not sym:
                    continue
                bucket.setdefault(sym, []).append(
                    {
                        "timestamp": bar.get("t"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "vwap": bar.get("vw"),
                        "volume": bar.get("v"),
                        "trade_count": bar.get("n"),
                    }
                )
            page_token = data.get("next_page_token")
            if not page_token:
                break

        # Remove symbols with no data to mirror SDK dict output
        return {k: v for k, v in bucket.items() if v}

    def _transform(
        self,
        raw_data: Dict[str, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform Alpaca bars to our schema.

        Args:
            raw_data: Raw API response
            **kwargs: Additional context

        Returns:
            DataFrame conforming to EQUITY_BARS_SCHEMA
        """
        # Alpaca returns nested dict: {symbol: [bars]}
        rows = []

        for symbol, bars in raw_data.items():
            if not bars:
                continue

            for bar in bars:
                # Extract bar data
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
            # Respect per-connector pacing for Alpaca SDK calls
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
                # Add metadata
                source_uri = f"alpaca://{timeframe}/{start_date}/{end_date}"
                df = self._add_metadata(df, source_uri=source_uri)
                all_data.append(df)

        if not all_data:
            logger.warning("No data fetched from Alpaca")
            return pd.DataFrame()

        # Combine all batches
        result = pd.concat(all_data, ignore_index=True)

        logger.info(f"Fetched {len(result)} bars for {len(symbols)} symbols")
        return result

    # ---------------- Options (SDK) ----------------

    def fetch_option_bars(
        self,
        contracts: List[str],
        start_date: date,
        end_date: date,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch option bars for a list of contract symbols over a date range.

        Returns combined DataFrame with columns:
          symbol, timestamp, open, high, low, close, volume, trade_count, vwap
        """
        if not self.opt_client:
            raise ConnectorError("Alpaca options client not initialized (no entitlement?)")
        if timeframe not in self.TIMEFRAME_MAP:
            raise ConnectorError(f"Invalid timeframe for options: {timeframe}")
        if not contracts:
            return pd.DataFrame()
        req = OptionBarsRequest(
            symbol_or_symbols=contracts,
            timeframe=self.TIMEFRAME_MAP[timeframe],
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
        )
        try:
            # Respect per-connector pacing for Alpaca SDK calls
            try:
                self._rate_limit()
            except Exception:
                pass
            bars = self.opt_client.get_option_bars(req)
        except Exception as e:
            raise ConnectorError(f"Alpaca options bars failed: {e}")
        # Flatten BarSet -> records
        rows = []
        try:
            for sym, blist in bars.data.items():
                for b in blist:
                    rows.append(
                        {
                            "symbol": sym,
                            "timestamp": b.timestamp,
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                            "trade_count": b.trade_count,
                            "vwap": b.vwap,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to parse bars: {e}")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(rows)
        return df

    def fetch_option_trades(
        self,
        contracts: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch option trades for a list of contract symbols. Alpaca trades API typically covers recent week.

        Returns combined DataFrame with columns:
          symbol, timestamp, price, size, exchange, tape, id
        """
        if not self.opt_client:
            raise ConnectorError("Alpaca options client not initialized (no entitlement?)")
        if not contracts:
            return pd.DataFrame()
        req = OptionTradesRequest(
            symbol_or_symbols=contracts,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
        )
        try:
            # Respect per-connector pacing for Alpaca SDK calls
            try:
                self._rate_limit()
            except Exception:
                pass
            trades = self.opt_client.get_option_trades(req)
        except Exception as e:
            raise ConnectorError(f"Alpaca options trades failed: {e}")
        rows = []
        try:
            for sym, tlist in trades.data.items():
                for t in tlist:
                    rows.append(
                        {
                            "symbol": sym,
                            "timestamp": t.timestamp,
                            "price": t.price,
                            "size": t.size,
                            "exchange": getattr(t, "exchange", None),
                            "tape": getattr(t, "tape", None),
                            "id": getattr(t, "id", None),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to parse trades: {e}")
        return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

    def fetch_option_chain_symbols(self, underlying_symbol: str, feed: Optional[str] = None, limit: int = 200) -> List[str]:
        """Get available contract symbols for an underlier via OptionChainRequest.

        When options entitlements are not present, falls back to indicative feed if available.
        Returns a list of contract symbols (strings)."""
        if not self.opt_client:
            raise ConnectorError("Alpaca options client not initialized (no entitlement?)")
        # feed: 'opra' or 'indicative' or None
        from alpaca.data.enums import OptionsFeed
        feed_param = None
        try:
            if feed:
                feed_param = OptionsFeed(feed)
        except Exception:
            feed_param = None
        try:
            req = OptionChainRequest(underlying_symbol=underlying_symbol, feed=feed_param)
            # Respect per-connector pacing for Alpaca SDK calls
            try:
                self._rate_limit()
            except Exception:
                pass
            snapshots = self.opt_client.get_option_chain(req)
        except Exception as e:
            raise ConnectorError(f"Alpaca option chain failed: {e}")
        # snapshots is dict[contract_symbol] -> OptionsSnapshot
        symbols = list(snapshots.keys()) if isinstance(snapshots, dict) else []
        if limit and len(symbols) > limit:
            symbols = symbols[:limit]
        return symbols

    def fetch_option_chain_snapshot_df(self, underlying_symbol: str, feed: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch current option chain snapshots for an underlying and flatten to DataFrame.

        Columns include contract symbol, underlier, IV, greeks (delta,gamma,theta,vega,rho),
        last trade (price,size,timestamp), last quote (bid,ask,bid_size,ask_size,timestamp).
        """
        if not self.opt_client:
            raise ConnectorError("Alpaca options client not initialized (no entitlement?)")
        from alpaca.data.enums import OptionsFeed
        feed_param = None
        try:
            if feed:
                feed_param = OptionsFeed(feed)
        except Exception:
            feed_param = None
        try:
            req = OptionChainRequest(underlying_symbol=underlying_symbol, feed=feed_param)
            # Respect per-connector pacing for Alpaca SDK calls
            try:
                self._rate_limit()
            except Exception:
                pass
            snapshots = self.opt_client.get_option_chain(req)
        except Exception as e:
            raise ConnectorError(f"Alpaca option chain failed: {e}")
        rows = []
        if isinstance(snapshots, dict):
            items = list(snapshots.items())
            if limit and len(items) > limit:
                items = items[:limit]
            # Control handling of indicative quotes
            save_ind_qq = os.getenv("ALPACA_OPTIONS_SAVE_INDICATIVE_QUOTES", "false").lower() in ("1","true","yes","on")
            is_indicative = str(feed).lower() == "indicative"
            for sym, snap in items:
                try:
                    lt = getattr(snap, "latest_trade", None)
                    lq = getattr(snap, "latest_quote", None)
                    greeks = getattr(snap, "greeks", None)
                    if is_indicative and not save_ind_qq:
                        lq = None
                        greeks = None
                    rows.append(
                        {
                            "contract": sym,
                            "underlier": underlying_symbol,
                            "iv": getattr(snap, "implied_volatility", None) if not (is_indicative and not save_ind_qq) else None,
                            "delta": getattr(greeks, "delta", None) if greeks else None,
                            "gamma": getattr(greeks, "gamma", None) if greeks else None,
                            "theta": getattr(greeks, "theta", None) if greeks else None,
                            "vega": getattr(greeks, "vega", None) if greeks else None,
                            "rho": getattr(greeks, "rho", None) if greeks else None,
                            "last_trade_price": getattr(lt, "price", None) if lt else None,
                            "last_trade_size": getattr(lt, "size", None) if lt else None,
                            "last_trade_time": getattr(lt, "timestamp", None) if lt else None,
                            "bid": getattr(lq, "bid_price", None) if lq else None,
                            "ask": getattr(lq, "ask_price", None) if lq else None,
                            "bid_size": getattr(lq, "bid_size", None) if lq else None,
                            "ask_size": getattr(lq, "ask_size", None) if lq else None,
                            "quote_time": getattr(lq, "timestamp", None) if lq else None,
                            "feed": str(feed) if feed else None,
                            "indicative": is_indicative,
                            "quote_quality": ("indicative" if is_indicative else "opra"),
                        }
                    )
                except Exception:
                    continue
        return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

    def fetch_corporate_actions(self, start: date, end: date, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch corporate actions between dates (inclusive) and flatten to a DataFrame.

        Includes splits, dividends, mergers, spinoffs, name changes, etc.
        """
        if not self.ca_client:
            raise ConnectorError("Alpaca corporate actions client not initialized")
        try:
            from alpaca.data.enums import CorporateActionsType
        except Exception:
            CorporateActionsType = None  # Not required; types optional
        req = CorporateActionsRequest(
            symbols=symbols,
            start=start,
            end=end,
            # types=None (all)
        )
        try:
            # Respect per-connector pacing for Alpaca SDK calls
            try:
                self._rate_limit()
            except Exception:
                pass
            ca = self.ca_client.get_corporate_actions(req)
        except Exception as e:
            raise ConnectorError(f"Alpaca corporate actions failed: {e}")
        rows = []
        if getattr(ca, "data", None):
            for ca_type, arr in ca.data.items():
                for item in arr:
                    try:
                        d = item.__dict__.copy()
                        d["type"] = ca_type
                        rows.append(d)
                    except Exception:
                        continue
        return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

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
        # Note: This is a simplified implementation
        # In production, you'd want to:
        # 1. Fetch all active symbols from Alpaca
        # 2. Get historical bars
        # 3. Filter by price and ADV
        # 4. Store results in reference database

        logger.info(
            f"Building universe with min_price={min_price}, "
            f"min_volume={min_volume}"
        )

        # For now, return a starter universe (top US equities)
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
    from datetime import timedelta
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch Alpaca market data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1Day")
    parser.add_argument("--output", type=str, default="data_layer/raw/equities_bars")

    args = parser.parse_args()

    # Parse dates
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Initialize connector
    connector = AlpacaConnector()

    # Fetch data
    df = connector.fetch_bars(
        symbols=args.symbols,
        start_date=start,
        end_date=end,
        timeframe=args.timeframe,
    )

    if not df.empty:
        # Write to Parquet
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
