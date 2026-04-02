"""Finnhub connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class FinnhubConnector(HTTPConnector):
    """Fetch bars and event/reference datasets from Finnhub."""

    vendor_name = "finnhub"

    def _auth_params(self) -> dict[str, str]:
        return {"token": self.api_key or ""}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized Finnhub datasets."""
        if dataset == "equities_eod":
            return self._fetch_equities(symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "earnings_calendar":
            payload = self.request_json(
                endpoint="/api/v1/calendar/earnings",
                params={"from": pd.Timestamp(start_date).strftime("%Y-%m-%d"), "to": pd.Timestamp(end_date).strftime("%Y-%m-%d")},
            )
            return pd.DataFrame(payload.get("earningsCalendar", []))
        if dataset == "company_profile":
            frames = []
            for symbol in symbols:
                payload = self.request_json(endpoint="/api/v1/stock/profile2", params={"symbol": symbol})
                frames.append(pd.DataFrame([payload]))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raise ValueError(f"unsupported dataset for finnhub: {dataset}")

    def _fetch_equities(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        start = int(pd.Timestamp(start_date).timestamp())
        end = int(pd.Timestamp(end_date).timestamp())
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            payload = self.request_json(
                endpoint="/api/v1/stock/candle",
                params={"symbol": symbol, "resolution": "D", "from": start, "to": end},
            )
            if payload.get("s") != "ok" or not payload.get("t"):
                continue
            frame = pd.DataFrame(payload)
            frame["symbol"] = symbol
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=self._columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"], unit="s", utc=True).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.now(tz="UTC")
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/api/v1/stock/candle"
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], unit="s", utc=True)
        bars_frame["trade_count"] = pd.NA
        renamed = bars_frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        renamed["vwap"] = pd.NA
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
