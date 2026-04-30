"""Alpaca market-data connector."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date as date_type
from itertools import islice
from typing import Iterator

import pandas as pd

from trademl.connectors.base import HTTPConnector


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


class AlpacaConnector(HTTPConnector):
    """Fetch daily equities bars from Alpaca."""

    vendor_name = "alpaca"

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        trading_base_url: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.secret_key = secret_key
        self.trading_base_url = (
            trading_base_url or kwargs.get("base_url") or ""
        ).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
        }

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized Alpaca bars."""
        if dataset == "equities_minute":
            return self._fetch_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Min",
                endpoint_key="equities_minute",
            )
        if dataset in {"stock_bars_boats", "stock_bars_otc"}:
            feed = "boats" if dataset == "stock_bars_boats" else "otc"
            return self._fetch_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Min",
                endpoint_key=dataset,
                feed=feed,
            )
        if dataset == "stock_trades":
            return self._fetch_market_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v2/stocks/trades",
                endpoint_key="stock_trades",
                root_key="trades",
                event_type="trade",
            )
        if dataset == "stock_quotes":
            return self._fetch_market_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v2/stocks/quotes",
                endpoint_key="stock_quotes",
                root_key="quotes",
                event_type="quote",
            )
        if dataset == "stock_snapshots":
            return self._fetch_stock_snapshots(symbols=symbols)
        if dataset == "news":
            return self._fetch_news(
                symbols=symbols, start_date=start_date, end_date=end_date
            )
        if dataset == "crypto_bars":
            return self._fetch_crypto_bars(
                symbols=symbols, start_date=start_date, end_date=end_date
            )
        if dataset == "crypto_trades":
            return self._fetch_crypto_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v1beta3/crypto/us/trades",
                endpoint_key="crypto_trades",
                root_key="trades",
                event_type="trade",
            )
        if dataset == "crypto_quotes":
            return self._fetch_crypto_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v1beta3/crypto/us/quotes",
                endpoint_key="crypto_quotes",
                root_key="quotes",
                event_type="quote",
            )
        if dataset == "crypto_snapshots":
            return self._fetch_crypto_snapshots(symbols=symbols)
        if dataset == "option_chain_reference":
            return self._fetch_option_chain_snapshots(symbols=symbols)
        if dataset == "option_snapshots":
            return self._fetch_option_snapshots(symbols=symbols)
        if dataset == "option_bars":
            return self._fetch_option_bars(
                symbols=symbols, start_date=start_date, end_date=end_date
            )
        if dataset == "assets":
            payload = self.request_json(
                base_url=self.trading_base_url,
                endpoint=self._assets_endpoint(),
                endpoint_key="assets",
                params={"status": "active", "asset_class": "us_equity"},
                task_kind="OTHER",
            )
            frame = pd.DataFrame(payload)
            if frame.empty:
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "name",
                        "exchange",
                        "status",
                        "tradable",
                        "asset_class",
                    ]
                )
            normalized = pd.DataFrame(
                {
                    "symbol": frame.get("symbol", pd.Series(dtype="string")),
                    "name": frame.get("name", pd.Series(dtype="string")),
                    "exchange": frame.get("exchange", pd.Series(dtype="string")),
                    "status": frame.get("status", pd.Series(dtype="string")),
                    "tradable": frame.get("tradable", pd.Series(dtype="bool")),
                    "asset_class": frame.get(
                        "class", frame.get("asset_class", pd.Series(dtype="string"))
                    ),
                }
            )
            return normalized.sort_values("symbol").reset_index(drop=True)
        if dataset == "corp_actions":
            frames: list[pd.DataFrame] = []
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
            for symbol in symbols:
                payload = self.request_json(
                    endpoint=f"/v2/stocks/{symbol}/corporate_actions/announcements",
                    endpoint_key="corp_actions",
                    params={"since": start, "until": end},
                    task_kind="OTHER",
                    logical_units=1,
                )
                rows = (
                    payload.get(
                        "announcements",
                        payload.get(
                            "results", payload if isinstance(payload, list) else []
                        ),
                    )
                    if isinstance(payload, dict)
                    else payload
                )
                frame = pd.DataFrame(rows)
                if frame.empty:
                    continue
                frame["symbol"] = symbol
                frame["event_type"] = (
                    frame.get(
                        "ca_type", frame.get("event_type", pd.Series(dtype="string"))
                    )
                    .astype("string")
                    .str.lower()
                )
                frame["ex_date"] = pd.to_datetime(
                    frame.get(
                        "ex_date", frame.get("execution_date", frame.get("date"))
                    ),
                    errors="coerce",
                ).dt.date
                frame["record_date"] = pd.to_datetime(
                    frame.get("record_date"), errors="coerce"
                ).dt.date
                frame["pay_date"] = pd.to_datetime(
                    frame.get("payable_date", frame.get("pay_date")), errors="coerce"
                ).dt.date
                frame["ratio"] = pd.to_numeric(frame.get("ratio"), errors="coerce")
                frame["amount"] = pd.to_numeric(
                    frame.get("cash", frame.get("amount")), errors="coerce"
                )
                frame["source"] = self.vendor_name
                frame["source_count"] = 1
                frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
                frames.append(
                    frame[
                        [
                            "symbol",
                            "event_type",
                            "ex_date",
                            "record_date",
                            "pay_date",
                            "ratio",
                            "amount",
                            "source",
                            "source_count",
                            "ingested_at",
                        ]
                    ]
                )
            if not frames:
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "event_type",
                        "ex_date",
                        "record_date",
                        "pay_date",
                        "ratio",
                        "amount",
                        "source",
                        "source_count",
                        "ingested_at",
                    ]
                )
            return (
                pd.concat(frames, ignore_index=True)
                .dropna(subset=["symbol", "ex_date"])
                .reset_index(drop=True)
            )
        if dataset != "equities_eod":
            raise ValueError(f"unsupported dataset for alpaca: {dataset}")

        return self._fetch_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1Day",
            endpoint_key="equities_eod",
        )

    def fetch_audit_sample(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch a bounded entitlement canary without draining a full archive lane."""
        if dataset in {"stock_trades", "stock_quotes"}:
            return self._fetch_market_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v2/stocks/trades" if dataset == "stock_trades" else "/v2/stocks/quotes",
                endpoint_key=dataset,
                root_key="trades" if dataset == "stock_trades" else "quotes",
                event_type="trade" if dataset == "stock_trades" else "quote",
                limit=100,
                max_pages=1,
            )
        if dataset in {"stock_bars_boats", "stock_bars_otc"}:
            return self._fetch_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Min",
                endpoint_key=dataset,
                feed="boats" if dataset == "stock_bars_boats" else "otc",
                limit=100,
                max_pages=1,
            )
        if dataset == "crypto_bars":
            return self._fetch_crypto_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                limit=100,
                max_pages=1,
            )
        if dataset in {"crypto_trades", "crypto_quotes"}:
            return self._fetch_crypto_events(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                endpoint="/v1beta3/crypto/us/trades" if dataset == "crypto_trades" else "/v1beta3/crypto/us/quotes",
                endpoint_key=dataset,
                root_key="trades" if dataset == "crypto_trades" else "quotes",
                event_type="trade" if dataset == "crypto_trades" else "quote",
                limit=100,
                max_pages=1,
            )
        if dataset == "option_bars":
            return self._fetch_option_bars(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                limit=100,
                max_pages=1,
            )
        if dataset == "option_chain_reference":
            return self._fetch_option_chain_snapshots(symbols=symbols, limit=100, max_pages=1)
        if dataset == "news":
            return self._fetch_news(symbols=symbols, start_date=start_date, end_date=end_date, limit=50, max_pages=1)
        return self.fetch(dataset, symbols, start_date, end_date)

    def _fetch_news(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        limit: int = 50,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        """Fetch Alpaca News and normalize it into the ticker_news schema."""
        frames: list[pd.DataFrame] = []
        page_token: str | None = None
        start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
        end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        pages = 0
        while True:
            pages += 1
            payload = self.request_json(
                endpoint="/v1beta1/news",
                endpoint_key="news",
                params={
                    "symbols": ",".join(str(symbol).upper() for symbol in symbols),
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "sort": "asc",
                    "include_content": "true",
                    **({"page_token": page_token} if page_token else {}),
                },
                task_kind="OTHER",
                logical_units=max(1, len(symbols)),
            )
            rows = payload.get("news", []) if isinstance(payload, dict) else payload
            frame = pd.DataFrame(rows)
            if not frame.empty:
                frames.append(frame)
            page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not page_token or (max_pages is not None and pages >= max_pages):
                break
        if not frames:
            return pd.DataFrame(columns=_news_columns())
        frame = pd.concat(frames, ignore_index=True)
        published = pd.to_datetime(
            frame.get("created_at", frame.get("updated_at")), errors="coerce", utc=True
        )
        raw_payload = frame.apply(lambda row: _json_dump(row.to_dict()), axis=1)
        normalized = pd.DataFrame(
            {
                "date": published.dt.date,
                "published_at": published,
                "crawled_at": pd.to_datetime(
                    frame.get("updated_at"), errors="coerce", utc=True
                ),
                "news_id": frame.get("id", pd.Series(dtype="string")).astype("string"),
                "headline": frame.get("headline", pd.Series(dtype="string")),
                "summary": frame.get("summary", pd.Series(dtype="string")),
                "content": frame.get(
                    "content",
                    frame.get("body", frame.get("text", pd.Series(dtype="string"))),
                ),
                "url": frame.get("url", pd.Series(dtype="string")),
                "image_url": frame.get("images", pd.Series(dtype="object")).map(
                    _first_image_url
                ),
                "category": pd.Series(["company_news"] * len(frame), dtype="string"),
                "source": frame.get("source", pd.Series(dtype="string")),
                "symbol": frame.get("symbols", pd.Series(dtype="object")).map(
                    _primary_symbol_from_value
                ),
                "related_symbols": frame.get("symbols", pd.Series(dtype="object")).map(
                    _symbols_from_value
                ),
                "tags": pd.Series([tuple()] * len(frame), dtype="object"),
                "source_name": self.vendor_name,
                "source_uri": "/v1beta1/news",
                "raw_payload": raw_payload,
                "raw_payload_hash": raw_payload.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()),
                "ingested_at": pd.Timestamp.now(tz=UTC),
            }
        )
        return normalized.dropna(subset=["published_at"]).reset_index(drop=True)

    def _fetch_bars(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        timeframe: str,
        endpoint_key: str,
        feed: str = "iex",
        limit: int = 10000,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if timeframe == "1Min":
            start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
            end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        else:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        for batch in _chunked(symbols, 100):
            page_token: str | None = None
            pages = 0
            while True:
                pages += 1
                payload = self.request_json(
                    endpoint="/v2/stocks/bars",
                    endpoint_key=endpoint_key,
                    params={
                        "symbols": ",".join(batch),
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": limit,
                        "adjustment": "raw",
                        "feed": feed,
                        "sort": "asc",
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="FORWARD",
                    logical_units=len(batch),
                )
                bars = payload.get("bars", {})
                for symbol, rows in bars.items():
                    if not rows:
                        continue
                    frame = pd.DataFrame(rows)
                    frame["symbol"] = symbol
                    frames.append(frame)
                page_token = payload.get("next_page_token")
                if not page_token or (max_pages is not None and pages >= max_pages):
                    break
        if not frames:
            self.budget_manager.record_empty_success(
                self.vendor_name, endpoint=endpoint_key
            )
            return pd.DataFrame(columns=self._columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"]).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/v2/stocks/bars"
        bars_frame["feed"] = feed
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], utc=True)
        renamed = bars_frame.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "vw": "vwap",
                "v": "volume",
                "n": "trade_count",
            }
        )
        return (
            renamed[self._columns()]
            .sort_values(["vendor_ts", "symbol"])
            .reset_index(drop=True)
        )

    def _assets_endpoint(self) -> str:
        """Return the assets endpoint relative to the configured trading base URL."""
        if self.trading_base_url.endswith("/v2"):
            return "/assets"
        return "/v2/assets"

    def _fetch_market_events(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        endpoint: str,
        endpoint_key: str,
        root_key: str,
        event_type: str,
        feed: str = "iex",
        limit: int = 10000,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
        end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        for batch in _chunked(symbols, 100):
            page_token: str | None = None
            pages = 0
            while True:
                pages += 1
                payload = self.request_json(
                    endpoint=endpoint,
                    endpoint_key=endpoint_key,
                    params={
                        "symbols": ",".join(batch),
                        "start": start,
                        "end": end,
                        "limit": limit,
                        **({"feed": feed} if feed else {}),
                        "sort": "asc",
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="OTHER",
                    logical_units=len(batch),
                )
                events = payload.get(root_key, {}) if isinstance(payload, dict) else {}
                for symbol, rows in events.items():
                    if not rows:
                        continue
                    frame = pd.DataFrame(rows)
                    frame["symbol"] = symbol
                    frames.append(frame)
                page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
                if not page_token or (max_pages is not None and pages >= max_pages):
                    break
        if not frames:
            self.budget_manager.record_empty_success(self.vendor_name, endpoint=endpoint_key)
            return pd.DataFrame(columns=_market_event_columns())
        frame = pd.concat(frames, ignore_index=True)
        return _normalize_market_events(
            frame=frame,
            source_name=self.vendor_name,
            source_uri=endpoint,
            event_type=event_type,
            feed=feed,
            asset_class="us_equity",
        )

    def _fetch_stock_snapshots(self, *, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for batch in _chunked(symbols, 100):
            payload = self.request_json(
                endpoint="/v2/stocks/snapshots",
                endpoint_key="stock_snapshots",
                params={"symbols": ",".join(batch), "feed": "iex"},
                task_kind="OTHER",
                logical_units=len(batch),
            )
            snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
            for symbol, row in dict(snapshots).items():
                if isinstance(row, dict):
                    frames.append(_flatten_snapshot(symbol=str(symbol), snapshot=row, asset_class="us_equity", feed="iex"))
        if not frames:
            return pd.DataFrame(columns=_snapshot_columns())
        return pd.DataFrame(frames).sort_values(["symbol"]).reset_index(drop=True)

    def _fetch_crypto_bars(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        limit: int = 10000,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
        end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        for batch in _chunked(symbols, 100):
            page_token: str | None = None
            pages = 0
            while True:
                pages += 1
                payload = self.request_json(
                    endpoint="/v1beta3/crypto/us/bars",
                    endpoint_key="crypto_bars",
                    params={
                        "symbols": ",".join(batch),
                        "timeframe": "1Min",
                        "start": start,
                        "end": end,
                        "limit": limit,
                        "sort": "asc",
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="OTHER",
                    logical_units=len(batch),
                )
                bars = payload.get("bars", {}) if isinstance(payload, dict) else {}
                for symbol, rows in bars.items():
                    if not rows:
                        continue
                    frame = pd.DataFrame(rows)
                    frame["symbol"] = symbol
                    frames.append(frame)
                page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
                if not page_token or (max_pages is not None and pages >= max_pages):
                    break
        if not frames:
            self.budget_manager.record_empty_success(self.vendor_name, endpoint="crypto_bars")
            return pd.DataFrame(columns=self._columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"]).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/v1beta3/crypto/us/bars"
        bars_frame["feed"] = "alpaca_crypto_us"
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], utc=True)
        renamed = bars_frame.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "vw": "vwap", "v": "volume", "n": "trade_count"}
        )
        return renamed[self._columns()].sort_values(["vendor_ts", "symbol"]).reset_index(drop=True)

    def _fetch_crypto_events(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        endpoint: str,
        endpoint_key: str,
        root_key: str,
        event_type: str,
        limit: int = 10000,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        frame = self._fetch_market_events(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            endpoint=endpoint,
            endpoint_key=endpoint_key,
            root_key=root_key,
            event_type=event_type,
            feed="",
            limit=limit,
            max_pages=max_pages,
        )
        if frame.empty:
            return frame
        frame["asset_class"] = "crypto"
        frame["feed"] = "alpaca_crypto_us"
        return frame

    def _fetch_crypto_snapshots(self, *, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for batch in _chunked(symbols, 100):
            payload = self.request_json(
                endpoint="/v1beta3/crypto/us/snapshots",
                endpoint_key="crypto_snapshots",
                params={"symbols": ",".join(batch)},
                task_kind="OTHER",
                logical_units=len(batch),
            )
            snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
            for symbol, row in dict(snapshots).items():
                if isinstance(row, dict):
                    frames.append(_flatten_snapshot(symbol=str(symbol), snapshot=row, asset_class="crypto", feed="alpaca_crypto_us"))
        if not frames:
            return pd.DataFrame(columns=_snapshot_columns())
        return pd.DataFrame(frames).sort_values(["symbol"]).reset_index(drop=True)

    def _fetch_option_chain_snapshots(self, *, symbols: list[str], limit: int = 1000, max_pages: int | None = None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for underlying in symbols:
            page_token: str | None = None
            pages = 0
            while True:
                pages += 1
                payload = self.request_json(
                    endpoint=f"/v1beta1/options/snapshots/{underlying}",
                    endpoint_key="option_chain_reference",
                    params={
                        "feed": "indicative",
                        "limit": limit,
                        "expiration_date_gte": pd.Timestamp.now(tz=UTC).date().isoformat(),
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="OTHER",
                    logical_units=1,
                )
                snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
                for contract_symbol, row in dict(snapshots).items():
                    if isinstance(row, dict):
                        flattened = _flatten_snapshot(
                            symbol=str(contract_symbol),
                            snapshot=row,
                            asset_class="option",
                            feed="indicative",
                        )
                        flattened["underlying_symbol"] = str(underlying).upper()
                        flattened["indicative"] = True
                        flattened["not_opra"] = True
                        flattened["not_live_trade_approved"] = True
                        frames.append(flattened)
                page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
                if not page_token or (max_pages is not None and pages >= max_pages):
                    break
        if not frames:
            return pd.DataFrame(columns=_option_snapshot_columns())
        return pd.DataFrame(frames).sort_values(["underlying_symbol", "symbol"]).reset_index(drop=True)

    def _fetch_option_snapshots(self, *, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for batch in _chunked(symbols, 100):
            payload = self.request_json(
                endpoint="/v1beta1/options/snapshots",
                endpoint_key="option_snapshots",
                params={"symbols": ",".join(batch), "feed": "indicative", "limit": 1000},
                task_kind="OTHER",
                logical_units=len(batch),
            )
            snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
            for symbol, row in dict(snapshots).items():
                if isinstance(row, dict):
                    flattened = _flatten_snapshot(symbol=str(symbol), snapshot=row, asset_class="option", feed="indicative")
                    flattened["indicative"] = True
                    flattened["not_opra"] = True
                    flattened["not_live_trade_approved"] = True
                    frames.append(flattened)
        if not frames:
            return pd.DataFrame(columns=_option_snapshot_columns())
        return pd.DataFrame(frames).sort_values(["symbol"]).reset_index(drop=True)

    def _fetch_option_bars(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        limit: int = 10000,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
        end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        for batch in _chunked(symbols, 100):
            page_token: str | None = None
            pages = 0
            while True:
                pages += 1
                payload = self.request_json(
                    endpoint="/v1beta1/options/bars",
                    endpoint_key="option_bars",
                    params={
                        "symbols": ",".join(batch),
                        "timeframe": "1Min",
                        "start": start,
                        "end": end,
                        "limit": limit,
                        "sort": "asc",
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="OTHER",
                    logical_units=len(batch),
                )
                bars = payload.get("bars", {}) if isinstance(payload, dict) else {}
                for symbol, rows in bars.items():
                    if rows:
                        frame = pd.DataFrame(rows)
                        frame["symbol"] = symbol
                        frames.append(frame)
                page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
                if not page_token or (max_pages is not None and pages >= max_pages):
                    break
        if not frames:
            return pd.DataFrame(columns=self._columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"]).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/v1beta1/options/bars"
        bars_frame["feed"] = "indicative"
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], utc=True)
        renamed = bars_frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "vw": "vwap", "v": "volume", "n": "trade_count"})
        return renamed[self._columns()].sort_values(["vendor_ts", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _columns() -> list[str]:
        return [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "trade_count",
            "feed",
            "ingested_at",
            "source_name",
            "source_uri",
            "vendor_ts",
        ]


def _symbols_from_value(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item).upper() for item in value if str(item).strip()}))
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(sorted({part.strip().upper() for part in text.split(",") if part.strip()}))


def _primary_symbol_from_value(value: object) -> str | None:
    symbols = _symbols_from_value(value)
    return symbols[0] if symbols else None


def _first_image_url(value: object) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("url") or first.get("source") or "") or None
        return str(first) or None
    if isinstance(value, dict):
        return str(value.get("url") or value.get("source") or "") or None
    return None


def _json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _event_id(*, symbol: object, vendor_ts: object, event_type: str, payload: object) -> str:
    return hashlib.sha256(f"{symbol}|{vendor_ts}|{event_type}|{_json_hash(payload)}".encode("utf-8")).hexdigest()


def _normalize_market_events(
    *,
    frame: pd.DataFrame,
    source_name: str,
    source_uri: str,
    event_type: str,
    feed: str,
    asset_class: str,
) -> pd.DataFrame:
    vendor_ts = pd.to_datetime(frame.get("t"), errors="coerce", utc=True)
    raw_payload = frame.apply(lambda row: _json_dump(row.to_dict()), axis=1)
    normalized = pd.DataFrame(
        {
            "date": vendor_ts.dt.date,
            "symbol": frame.get("symbol", pd.Series(dtype="string")).astype("string"),
            "event_type": event_type,
            "asset_class": asset_class,
            "price": pd.to_numeric(frame.get("p"), errors="coerce"),
            "size": pd.to_numeric(frame.get("s"), errors="coerce"),
            "exchange": frame.get("x", pd.Series(dtype="string")).astype("string"),
            "conditions": frame.get("c", pd.Series(dtype="object")).map(_symbols_from_value),
            "bid_price": pd.to_numeric(frame.get("bp"), errors="coerce"),
            "bid_size": pd.to_numeric(frame.get("bs"), errors="coerce"),
            "bid_exchange": frame.get("bx", pd.Series(dtype="string")).astype("string"),
            "ask_price": pd.to_numeric(frame.get("ap"), errors="coerce"),
            "ask_size": pd.to_numeric(frame.get("as"), errors="coerce"),
            "ask_exchange": frame.get("ax", pd.Series(dtype="string")).astype("string"),
            "feed": feed,
            "vendor_ts": vendor_ts,
            "source_name": source_name,
            "source_uri": source_uri,
            "raw_payload": raw_payload,
            "raw_payload_hash": raw_payload.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()),
            "ingested_at": pd.Timestamp.now(tz=UTC),
        }
    )
    normalized["event_id"] = [
        _event_id(symbol=symbol, vendor_ts=ts, event_type=event_type, payload=payload)
        for symbol, ts, payload in zip(normalized["symbol"], normalized["vendor_ts"], raw_payload, strict=False)
    ]
    return normalized.dropna(subset=["vendor_ts", "symbol"]).reset_index(drop=True)


def _flatten_snapshot(*, symbol: str, snapshot: dict[str, object], asset_class: str, feed: str) -> dict[str, object]:
    latest_trade = dict(snapshot.get("latestTrade") or snapshot.get("latest_trade") or {})
    latest_quote = dict(snapshot.get("latestQuote") or snapshot.get("latest_quote") or {})
    minute_bar = dict(snapshot.get("minuteBar") or snapshot.get("minute_bar") or {})
    daily_bar = dict(snapshot.get("dailyBar") or snapshot.get("daily_bar") or {})
    previous_bar = dict(snapshot.get("prevDailyBar") or snapshot.get("previous_daily_bar") or {})
    greeks = dict(snapshot.get("greeks") or {})
    ts = pd.to_datetime(
        latest_trade.get("t") or latest_quote.get("t") or minute_bar.get("t") or daily_bar.get("t"),
        errors="coerce",
        utc=True,
    )
    return {
        "date": ts.date() if not pd.isna(ts) else pd.Timestamp.now(tz=UTC).date(),
        "symbol": symbol.upper(),
        "asset_class": asset_class,
        "feed": feed,
        "vendor_ts": ts,
        "latest_trade_price": latest_trade.get("p"),
        "latest_trade_size": latest_trade.get("s"),
        "latest_quote_bid_price": latest_quote.get("bp"),
        "latest_quote_bid_size": latest_quote.get("bs"),
        "latest_quote_ask_price": latest_quote.get("ap"),
        "latest_quote_ask_size": latest_quote.get("as"),
        "minute_open": minute_bar.get("o"),
        "minute_high": minute_bar.get("h"),
        "minute_low": minute_bar.get("l"),
        "minute_close": minute_bar.get("c"),
        "minute_volume": minute_bar.get("v"),
        "daily_open": daily_bar.get("o"),
        "daily_high": daily_bar.get("h"),
        "daily_low": daily_bar.get("l"),
        "daily_close": daily_bar.get("c"),
        "daily_volume": daily_bar.get("v"),
        "previous_daily_close": previous_bar.get("c"),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "rho": greeks.get("rho"),
        "implied_volatility": snapshot.get("impliedVolatility") or snapshot.get("implied_volatility"),
        "source_name": "alpaca",
        "source_uri": "snapshot",
        "raw_payload": _json_dump(snapshot),
        "raw_payload_hash": _json_hash(snapshot),
        "ingested_at": pd.Timestamp.now(tz=UTC),
    }


def _news_columns() -> list[str]:
    return [
        "date",
        "published_at",
        "crawled_at",
        "news_id",
        "headline",
        "summary",
        "content",
        "url",
        "image_url",
        "category",
        "source",
        "symbol",
        "related_symbols",
        "tags",
        "source_name",
        "source_uri",
        "raw_payload",
        "raw_payload_hash",
        "ingested_at",
    ]


def _market_event_columns() -> list[str]:
    return [
        "date",
        "symbol",
        "event_type",
        "asset_class",
        "price",
        "size",
        "exchange",
        "conditions",
        "bid_price",
        "bid_size",
        "bid_exchange",
        "ask_price",
        "ask_size",
        "ask_exchange",
        "feed",
        "vendor_ts",
        "event_id",
        "source_name",
        "source_uri",
        "raw_payload",
        "raw_payload_hash",
        "ingested_at",
    ]


def _snapshot_columns() -> list[str]:
    return [
        "date",
        "symbol",
        "asset_class",
        "feed",
        "vendor_ts",
        "latest_trade_price",
        "latest_trade_size",
        "latest_quote_bid_price",
        "latest_quote_bid_size",
        "latest_quote_ask_price",
        "latest_quote_ask_size",
        "minute_open",
        "minute_high",
        "minute_low",
        "minute_close",
        "minute_volume",
        "daily_open",
        "daily_high",
        "daily_low",
        "daily_close",
        "daily_volume",
        "previous_daily_close",
        "source_name",
        "source_uri",
        "raw_payload",
        "raw_payload_hash",
        "ingested_at",
    ]


def _option_snapshot_columns() -> list[str]:
    return [
        *_snapshot_columns(),
        "underlying_symbol",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "implied_volatility",
        "indicative",
        "not_opra",
        "not_live_trade_approved",
    ]
