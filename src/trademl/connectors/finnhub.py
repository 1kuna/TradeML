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
                endpoint_key="earnings_calendar",
                params={"from": pd.Timestamp(start_date).strftime("%Y-%m-%d"), "to": pd.Timestamp(end_date).strftime("%Y-%m-%d")},
                task_kind="OTHER",
            )
            return pd.DataFrame(payload.get("earningsCalendar", []))
        if dataset == "company_profile":
            frames = []
            for symbol in symbols:
                payload = self.request_json(endpoint="/api/v1/stock/profile2", endpoint_key="profile", params={"symbol": symbol}, logical_units=1)
                frames.append(pd.DataFrame([payload]))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if dataset == "company_news":
            frames = []
            for symbol in symbols:
                payload = self.request_json(
                    endpoint="/api/v1/company-news",
                    endpoint_key="company_news",
                    params={
                        "symbol": symbol,
                        "from": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                        "to": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                    },
                    task_kind="OTHER",
                    logical_units=1,
                )
                frame = pd.DataFrame(payload)
                if frame.empty:
                    continue
                published = pd.to_datetime(frame.get("datetime"), unit="s", errors="coerce", utc=True)
                normalized = pd.DataFrame(
                    {
                        "date": published.dt.date,
                        "published_at": published,
                        "crawled_at": pd.NaT,
                        "news_id": pd.to_numeric(frame.get("id"), errors="coerce"),
                        "headline": frame.get("headline", pd.Series(dtype="string")),
                        "summary": frame.get("summary", pd.Series(dtype="string")),
                        "url": frame.get("url", pd.Series(dtype="string")),
                        "image_url": frame.get("image", pd.Series(dtype="string")),
                        "category": frame.get("category", pd.Series(dtype="string")),
                        "source": frame.get("source", pd.Series(dtype="string")),
                        "symbol": symbol,
                        "related_symbols": frame.get("related").map(_symbols_from_value),
                        "tags": pd.Series([tuple()] * len(frame), dtype="object"),
                        "source_name": self.vendor_name,
                        "source_uri": "/api/v1/company-news",
                        "ingested_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
                frames.append(normalized.dropna(subset=["published_at"]))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_news_columns())
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
                endpoint_key="equities_eod",
                params={"symbol": symbol, "resolution": "D", "from": start, "to": end},
                task_kind="FORWARD",
                logical_units=1,
            )
            if payload.get("s") == "no_data":
                continue
            if payload.get("s") != "ok" or not payload.get("t"):
                continue
            frame = pd.DataFrame(payload)
            frame["symbol"] = symbol
            frames.append(frame)
        if not frames:
            self.budget_manager.record_empty_success(self.vendor_name, endpoint="equities_eod")
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


def _symbols_from_value(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(sorted({part.strip().upper() for part in text.split(",") if part.strip()}))


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
