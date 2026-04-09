"""Massive / Polygon.io connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class MassiveConnector(HTTPConnector):
    """Fetch bars and reference datasets from Massive (Polygon.io)."""

    vendor_name = "massive"

    def _auth_params(self) -> dict[str, str]:
        return {"apiKey": self.api_key or ""}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized Massive datasets."""
        if dataset == "equities_eod":
            return self._fetch_equities_bars(symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "reference_splits":
            return self._fetch_reference(symbols=symbols, endpoint_template="/v3/reference/splits")
        if dataset == "reference_dividends":
            return self._fetch_reference(symbols=symbols, endpoint_template="/v3/reference/dividends")
        if dataset == "reference_tickers":
            payload = self.request_json(endpoint="/v3/reference/tickers", endpoint_key="reference_tickers", params={"active": "true", "limit": 1000})
            return pd.DataFrame(payload.get("results", []))
        raise ValueError(f"unsupported dataset for massive: {dataset}")

    def _fetch_equities_bars(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        for symbol in symbols:
            payload = self.request_json(
                endpoint=f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                endpoint_key="equities_eod",
                params={"adjusted": "false", "sort": "asc", "limit": 50000},
                logical_units=1,
            )
            rows = payload.get("results", [])
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["symbol"] = symbol
            frames.append(frame)
        if not frames:
            self.budget_manager.record_empty_success(self.vendor_name, endpoint="equities_eod")
            return pd.DataFrame(columns=self._bar_columns())
        bars_frame = pd.concat(frames, ignore_index=True)
        bars_frame["date"] = pd.to_datetime(bars_frame["t"], unit="ms", utc=True).dt.date
        bars_frame["ingested_at"] = pd.Timestamp.now(tz="UTC")
        bars_frame["source_name"] = self.vendor_name
        bars_frame["source_uri"] = "/v2/aggs/ticker"
        bars_frame["vendor_ts"] = pd.to_datetime(bars_frame["t"], unit="ms", utc=True)
        renamed = bars_frame.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "vw": "vwap", "v": "volume", "n": "trade_count"}
        )
        if "trade_count" not in renamed:
            renamed["trade_count"] = pd.NA
        return renamed[self._bar_columns()].sort_values(["date", "symbol"]).reset_index(drop=True)

    def _fetch_reference(self, *, symbols: list[str], endpoint_template: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            payload = self.request_json(
                endpoint=endpoint_template,
                endpoint_key="reference_splits" if "splits" in endpoint_template else "reference_dividends",
                params={"ticker": symbol, "limit": 1000},
                logical_units=1,
            )
            frame = pd.DataFrame(payload.get("results", []))
            if not frame.empty:
                frame["symbol"] = symbol
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _bar_columns() -> list[str]:
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
