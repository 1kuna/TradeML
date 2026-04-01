"""Alpaca market-data connector."""

from __future__ import annotations

from datetime import date as date_type
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

    def __init__(self, *, secret_key: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.secret_key = secret_key

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
        if dataset == "assets":
            payload = self.request_json(
                endpoint="/v2/assets",
                params={"status": "active", "asset_class": "us_equity"},
                task_kind="OTHER",
            )
            frame = pd.DataFrame(payload)
            if frame.empty:
                return pd.DataFrame(columns=["symbol", "name", "exchange", "status", "tradable", "asset_class"])
            normalized = pd.DataFrame(
                {
                    "symbol": frame.get("symbol", pd.Series(dtype="string")),
                    "name": frame.get("name", pd.Series(dtype="string")),
                    "exchange": frame.get("exchange", pd.Series(dtype="string")),
                    "status": frame.get("status", pd.Series(dtype="string")),
                    "tradable": frame.get("tradable", pd.Series(dtype="bool")),
                    "asset_class": frame.get("class", frame.get("asset_class", pd.Series(dtype="string"))),
                }
            )
            return normalized.sort_values("symbol").reset_index(drop=True)
        if dataset != "equities_eod":
            raise ValueError(f"unsupported dataset for alpaca: {dataset}")

        frames: list[pd.DataFrame] = []
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        for batch in _chunked(symbols, 100):
            page_token: str | None = None
            while True:
                payload = self.request_json(
                    endpoint="/v2/stocks/bars",
                    params={
                        "symbols": ",".join(batch),
                        "timeframe": "1Day",
                        "start": start,
                        "end": end,
                        "adjustment": "raw",
                        "feed": "iex",
                        **({"page_token": page_token} if page_token else {}),
                    },
                    task_kind="FORWARD",
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
            return pd.DataFrame(columns=self._columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"]).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.utcnow()
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/v2/stocks/bars"
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], utc=True)
        renamed = bars_frame.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "vw": "vwap", "v": "volume", "n": "trade_count"}
        )
        return renamed[self._columns()].sort_values(["date", "symbol"]).reset_index(drop=True)

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
