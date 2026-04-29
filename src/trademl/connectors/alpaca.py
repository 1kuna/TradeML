"""Alpaca market-data connector."""

from __future__ import annotations

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
        if dataset == "news":
            return self._fetch_news(
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

    def _fetch_news(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch Alpaca News and normalize it into the ticker_news schema."""
        frames: list[pd.DataFrame] = []
        page_token: str | None = None
        start = f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')}T00:00:00Z"
        end = f"{pd.Timestamp(end_date).strftime('%Y-%m-%d')}T23:59:59Z"
        while True:
            payload = self.request_json(
                endpoint="/v1beta1/news",
                endpoint_key="news",
                params={
                    "symbols": ",".join(str(symbol).upper() for symbol in symbols),
                    "start": start,
                    "end": end,
                    "limit": 50,
                    "sort": "asc",
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
            if not page_token:
                break
        if not frames:
            return pd.DataFrame(columns=_news_columns())
        frame = pd.concat(frames, ignore_index=True)
        published = pd.to_datetime(
            frame.get("created_at", frame.get("updated_at")), errors="coerce", utc=True
        )
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
            while True:
                payload = self.request_json(
                    endpoint="/v2/stocks/bars",
                    endpoint_key=endpoint_key,
                    params={
                        "symbols": ",".join(batch),
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": 10000,
                        "adjustment": "raw",
                        "feed": "iex",
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
                if not page_token:
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


def _news_columns() -> list[str]:
    return [
        "date",
        "published_at",
        "crawled_at",
        "news_id",
        "headline",
        "summary",
        "url",
        "image_url",
        "category",
        "source",
        "symbol",
        "related_symbols",
        "tags",
        "source_name",
        "source_uri",
        "ingested_at",
    ]
