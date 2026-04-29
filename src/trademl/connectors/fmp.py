"""Financial Modeling Prep connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class FMPConnector(HTTPConnector):
    """Fetch delistings and earnings data from FMP."""

    vendor_name = "fmp"

    def _auth_params(self) -> dict[str, str]:
        return {"apikey": self.api_key or ""}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized FMP datasets."""
        if dataset == "delistings":
            frames: list[pd.DataFrame] = []
            page = 0
            limit = 100
            while True:
                payload = self.request_json(
                    endpoint="/stable/delisted-companies",
                    endpoint_key="delistings",
                    params={"page": page, "limit": limit},
                )
                frame = pd.DataFrame(payload)
                if frame.empty:
                    break
                frames.append(frame)
                if len(frame) < limit:
                    break
                page += 1
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if dataset == "symbol_changes":
            payload = self.request_json(endpoint="/stable/symbol-change", endpoint_key="symbol_changes")
            return pd.DataFrame(payload)
        if dataset == "earnings_calendar":
            payload = self.request_json(
                endpoint="/stable/earnings-calendar",
                endpoint_key="earnings_calendar",
                params={"from": pd.Timestamp(start_date).strftime("%Y-%m-%d"), "to": pd.Timestamp(end_date).strftime("%Y-%m-%d")},
            )
            return pd.DataFrame(payload)
        if dataset in {"stock_news", "press_releases"}:
            return self._fetch_news(
                dataset=dataset, symbols=symbols, start_date=start_date, end_date=end_date
            )
        raise ValueError(f"unsupported dataset for fmp: {dataset}")

    def _fetch_news(
        self,
        *,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch FMP stock news or press releases into the ticker_news schema."""
        endpoint = (
            "/stable/news/press-releases"
            if dataset == "press_releases"
            else "/stable/news/stock"
        )
        payload = self.request_json(
            endpoint=endpoint,
            endpoint_key=dataset,
            params={
                "symbols": ",".join(str(symbol).upper() for symbol in symbols),
                "from": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "to": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "page": 0,
                "limit": 100,
            },
            logical_units=max(1, len(symbols)),
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=_news_columns())
        published = pd.to_datetime(
            frame.get("publishedDate", frame.get("date")), errors="coerce", utc=True
        )
        symbol_values = frame.get(
            "symbols", frame.get("symbol", frame.get("ticker", pd.Series(dtype="object")))
        )
        normalized = pd.DataFrame(
            {
                "date": published.dt.date,
                "published_at": published,
                "crawled_at": pd.NaT,
                "news_id": frame.get("url", pd.Series(dtype="string")).astype("string"),
                "headline": frame.get("title", pd.Series(dtype="string")),
                "summary": frame.get(
                    "text", frame.get("content", frame.get("summary", pd.Series(dtype="string")))
                ),
                "url": frame.get("url", pd.Series(dtype="string")),
                "image_url": frame.get("image", pd.Series(dtype="string")),
                "category": pd.Series([dataset] * len(frame), dtype="string"),
                "source": frame.get("site", frame.get("publisher", pd.Series(dtype="string"))),
                "symbol": symbol_values.map(_primary_symbol_from_value),
                "related_symbols": symbol_values.map(_symbols_from_value),
                "tags": pd.Series([tuple()] * len(frame), dtype="object"),
                "source_name": self.vendor_name,
                "source_uri": endpoint,
                "ingested_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        return normalized.dropna(subset=["published_at"]).reset_index(drop=True)


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
