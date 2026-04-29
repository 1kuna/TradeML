"""Alpha Vantage connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class AlphaVantageConnector(HTTPConnector):
    """Fetch listings and reference datasets from Alpha Vantage."""

    vendor_name = "alpha_vantage"

    def _auth_params(self) -> dict[str, str]:
        return {"apikey": self.api_key or ""}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized Alpha Vantage datasets."""
        if dataset == "listings":
            frames = [
                self.request_csv(
                    endpoint="/query",
                    endpoint_key="listings",
                    params={"function": "LISTING_STATUS", "date": pd.Timestamp(end_date).strftime("%Y-%m-%d"), "state": state},
                )
                for state in ["active", "delisted"]
            ]
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if combined.empty:
                return combined
            return combined.drop_duplicates().reset_index(drop=True)
        if dataset in {"corp_actions", "splits", "dividends"}:
            return self._fetch_corp_actions(dataset=dataset, symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "news_sentiment":
            return self._fetch_news_sentiment(
                symbols=symbols, start_date=start_date, end_date=end_date
            )
        raise ValueError(f"unsupported dataset for alpha_vantage: {dataset}")

    def _fetch_news_sentiment(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch Alpha Vantage NEWS_SENTIMENT into the ticker_news schema."""
        payload = self.request_json(
            endpoint="/query",
            endpoint_key="news_sentiment",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(str(symbol).upper() for symbol in symbols),
                "time_from": pd.Timestamp(start_date).strftime("%Y%m%dT0000"),
                "time_to": pd.Timestamp(end_date).strftime("%Y%m%dT2359"),
                "sort": "EARLIEST",
                "limit": 1000,
            },
            logical_units=max(1, len(symbols)),
        )
        rows = payload.get("feed", []) if isinstance(payload, dict) else payload
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=_news_columns())
        published = pd.to_datetime(
            frame.get("time_published"),
            format="%Y%m%dT%H%M%S",
            errors="coerce",
            utc=True,
        )
        ticker_values = frame.get("ticker_sentiment", pd.Series(dtype="object"))
        topic_values = frame.get("topics", pd.Series(dtype="object"))
        normalized = pd.DataFrame(
            {
                "date": published.dt.date,
                "published_at": published,
                "crawled_at": pd.NaT,
                "news_id": frame.get("url", pd.Series(dtype="string")).astype("string"),
                "headline": frame.get("title", pd.Series(dtype="string")),
                "summary": frame.get("summary", pd.Series(dtype="string")),
                "url": frame.get("url", pd.Series(dtype="string")),
                "image_url": frame.get("banner_image", pd.Series(dtype="string")),
                "category": pd.Series(["company_news"] * len(frame), dtype="string"),
                "source": frame.get("source", pd.Series(dtype="string")),
                "symbol": ticker_values.map(_primary_symbol_from_ticker_sentiment),
                "related_symbols": ticker_values.map(_symbols_from_ticker_sentiment),
                "tags": topic_values.map(_topics_from_value),
                "source_name": self.vendor_name,
                "source_uri": "/query?function=NEWS_SENTIMENT",
                "ingested_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        return normalized.dropna(subset=["published_at"]).reset_index(drop=True)

    def _fetch_corp_actions(
        self,
        *,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        for symbol in symbols:
            dividends_payload = self.request_json(
                endpoint="/query",
                endpoint_key="corp_actions",
                params={"function": "DIVIDENDS", "symbol": symbol},
                logical_units=1,
            )
            splits_payload = self.request_json(
                endpoint="/query",
                endpoint_key="corp_actions",
                params={"function": "SPLITS", "symbol": symbol},
                logical_units=1,
            )
            dividend_rows = self._normalize_actions(
                payload=dividends_payload,
                symbol=symbol,
                event_type="dividend",
                date_key="ex_dividend_date",
                value_key="amount",
                start_date=start_ts,
                end_date=end_ts,
            )
            split_rows = self._normalize_actions(
                payload=splits_payload,
                symbol=symbol,
                event_type="split",
                date_key="effective_date",
                value_key="split_factor",
                start_date=start_ts,
                end_date=end_ts,
            )
            if dataset == "dividends":
                frame = dividend_rows
            elif dataset == "splits":
                frame = split_rows
            else:
                non_empty = [item for item in [dividend_rows, split_rows] if not item.empty]
                frame = pd.DataFrame.from_records(
                    [record for item in non_empty for record in item.to_dict("records")]
                ) if non_empty else pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount", "source"])

    def _normalize_actions(
        self,
        *,
        payload: object,
        symbol: str,
        event_type: str,
        date_key: str,
        value_key: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = _action_rows(payload, event_type=event_type)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount", "source"])
        resolved_date_key = date_key if date_key in frame.columns else "date"
        frame["ex_date"] = pd.to_datetime(frame[resolved_date_key], errors="coerce")
        frame = frame.loc[frame["ex_date"].between(start_date.normalize(), end_date.normalize())].copy()
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount", "source"])
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        else:
            frame["symbol"] = frame["symbol"].fillna(symbol)
        frame["event_type"] = event_type
        raw_values = frame[value_key].astype(str)
        if event_type == "split":
            numeric_value = raw_values.map(_parse_split_factor)
        else:
            numeric_value = pd.to_numeric(frame[value_key], errors="coerce")
        frame["ratio"] = numeric_value if event_type == "split" else pd.NA
        frame["amount"] = numeric_value if event_type == "dividend" else pd.NA
        frame["source"] = "alpha_vantage"
        frame["ex_date"] = frame["ex_date"].dt.date
        return frame[["symbol", "event_type", "ex_date", "ratio", "amount", "source"]]


def _parse_split_factor(value: str) -> float | pd.NA:
    if ":" in value:
        left, right = value.split(":", 1)
        numerator = pd.to_numeric(left, errors="coerce")
        denominator = pd.to_numeric(right, errors="coerce")
        if pd.notna(numerator) and pd.notna(denominator) and float(denominator) != 0.0:
            return float(denominator) / float(numerator)
        return pd.NA
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else pd.NA


def _action_rows(payload: object, *, event_type: str) -> object:
    if not isinstance(payload, dict):
        return payload
    if event_type == "dividend":
        return payload.get("dividends", payload.get("data", []))
    if event_type == "split":
        return payload.get("splits", payload.get("data", []))
    return payload.get("data", [])


def _symbols_from_ticker_sentiment(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    symbols = {
        str(item.get("ticker", "")).upper()
        for item in value
        if isinstance(item, dict) and str(item.get("ticker", "")).strip()
    }
    return tuple(sorted(symbols))


def _primary_symbol_from_ticker_sentiment(value: object) -> str | None:
    symbols = _symbols_from_ticker_sentiment(value)
    return symbols[0] if symbols else None


def _topics_from_value(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    topics = {
        str(item.get("topic", "")).upper()
        for item in value
        if isinstance(item, dict) and str(item.get("topic", "")).strip()
    }
    return tuple(sorted(topics))


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
