"""Twelve Data connector."""

from __future__ import annotations

from datetime import UTC, date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class TwelveDataConnector(HTTPConnector):
    """Fetch Twelve Data bars, corporate actions, earnings, statements, and symbol metadata."""

    vendor_name = "twelve_data"

    def _auth_params(self) -> dict[str, str]:
        return {"apikey": self.api_key or ""}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch a normalized Twelve Data dataset."""
        if dataset == "equities_eod":
            return self._fetch_equities(symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset in {"dividends", "splits"}:
            return self._fetch_corporate_actions(dataset=dataset, symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "earnings_calendar":
            payload = self.request_json(
                endpoint="/earnings",
                params={"start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"), "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d")},
                task_kind="OTHER",
            )
            rows = payload.get("earnings", payload.get("data", payload if isinstance(payload, list) else [])) if isinstance(payload, dict) else payload
            return pd.DataFrame(rows)
        if dataset == "financial_statements":
            return self._fetch_financial_statements(symbols=symbols)
        if dataset == "stocks":
            payload = self.request_json(endpoint="/stocks", task_kind="OTHER")
            rows = payload.get("data", payload if isinstance(payload, list) else []) if isinstance(payload, dict) else payload
            return pd.DataFrame(rows)
        if dataset == "price_target":
            frames = []
            for symbol in symbols:
                payload = self.request_json(endpoint="/price_target", params={"symbol": symbol}, task_kind="OTHER")
                frames.append(pd.DataFrame([payload.get("data", payload)] if isinstance(payload, dict) else payload))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if dataset == "insider_transactions":
            frames = []
            for symbol in symbols:
                payload = self.request_json(endpoint="/insider_transactions", params={"symbol": symbol}, task_kind="OTHER")
                rows = payload.get("data", payload if isinstance(payload, list) else []) if isinstance(payload, dict) else payload
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame["symbol"] = symbol
                    frames.append(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        raise ValueError(f"unsupported dataset for twelve_data: {dataset}")

    def _fetch_equities(
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
                endpoint="/time_series",
                params={"symbol": symbol, "interval": "1day", "start_date": start, "end_date": end, "order": "ASC", "format": "JSON"},
                task_kind="FORWARD",
            )
            values = payload.get("values", []) if isinstance(payload, dict) else payload
            frame = pd.DataFrame(values)
            if frame.empty:
                continue
            frame["symbol"] = payload.get("meta", {}).get("symbol", symbol) if isinstance(payload, dict) else symbol
            frame["vendor_ts"] = pd.to_datetime(frame.get("datetime", frame.get("date")), errors="coerce", utc=True)
            frame["date"] = frame["vendor_ts"].dt.date
            frame["open"] = pd.to_numeric(frame.get("open"), errors="coerce")
            frame["high"] = pd.to_numeric(frame.get("high"), errors="coerce")
            frame["low"] = pd.to_numeric(frame.get("low"), errors="coerce")
            frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
            frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
            frame["vwap"] = pd.NA
            frame["trade_count"] = pd.NA
            frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
            frame["source_name"] = self.vendor_name
            frame["source_uri"] = "/time_series"
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=self._equity_columns())
        return pd.concat(frames, ignore_index=True)[self._equity_columns()].sort_values(["date", "symbol"]).reset_index(drop=True)

    def _fetch_corporate_actions(
        self,
        *,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        endpoint = f"/{dataset}"
        for symbol in symbols:
            payload = self.request_json(
                endpoint=endpoint,
                params={"symbol": symbol, "start_date": start, "end_date": end},
                task_kind="OTHER",
            )
            rows = payload.get(dataset, payload.get("data", payload.get("values", []))) if isinstance(payload, dict) else payload
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            frame["symbol"] = payload.get("meta", {}).get("symbol", symbol) if isinstance(payload, dict) else symbol
            frame["event_type"] = "dividend" if dataset == "dividends" else "split"
            frame["ex_date"] = pd.to_datetime(frame.get("ex_date", frame.get("date")), errors="coerce").dt.date
            frame["record_date"] = _normalize_optional_date(frame, "record_date", "recordDate").dt.date
            frame["pay_date"] = _normalize_optional_date(frame, "payment_date", "pay_date", "payDate").dt.date
            if dataset == "dividends":
                frame["amount"] = pd.to_numeric(frame.get("amount", frame.get("cash_amount")), errors="coerce")
                frame["ratio"] = pd.NA
            else:
                frame["ratio"] = frame.get("ratio", pd.Series(dtype="string")).map(_parse_adjustment_ratio)
                frame["amount"] = pd.NA
            frame["source"] = self.vendor_name
            frame["source_count"] = 1
            frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
            frames.append(frame[self._action_columns()])
        if not frames:
            return pd.DataFrame(columns=self._action_columns())
        return pd.concat(frames, ignore_index=True).dropna(subset=["symbol", "ex_date"]).reset_index(drop=True)

    def _fetch_financial_statements(self, *, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        endpoints = {
            "income_statement": "/income_statement",
            "balance_sheet": "/balance_sheet",
            "cash_flow": "/cash_flow",
        }
        for symbol in symbols:
            for statement_type, endpoint in endpoints.items():
                payload = self.request_json(endpoint=endpoint, params={"symbol": symbol}, task_kind="OTHER")
                rows = payload.get(statement_type, payload.get("data", payload if isinstance(payload, list) else [])) if isinstance(payload, dict) else payload
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    fiscal_date = pd.to_datetime(row.get("fiscal_date", row.get("date")), errors="coerce")
                    frames.append(
                        {
                            "symbol": payload.get("meta", {}).get("symbol", symbol) if isinstance(payload, dict) else symbol,
                            "statement_type": statement_type,
                            "date": fiscal_date.normalize() if pd.notna(fiscal_date) else pd.NaT,
                            "as_of_date": fiscal_date.normalize() if pd.notna(fiscal_date) else pd.NaT,
                            "period": row.get("period"),
                            "data": row,
                        }
                    )
        if not frames:
            return pd.DataFrame(columns=["symbol", "statement_type", "date", "as_of_date", "period", "data"])
        return pd.DataFrame(frames).sort_values(["symbol", "statement_type", "date"]).reset_index(drop=True)

    @staticmethod
    def _equity_columns() -> list[str]:
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

    @staticmethod
    def _action_columns() -> list[str]:
        return ["symbol", "event_type", "ex_date", "record_date", "pay_date", "ratio", "amount", "source", "source_count", "ingested_at"]


def _parse_adjustment_ratio(value: object) -> float | pd.NA:
    text = str(value or "").strip()
    if not text:
        return pd.NA
    if ":" in text:
        left, right = text.split(":", 1)
        numerator = pd.to_numeric(left, errors="coerce")
        denominator = pd.to_numeric(right, errors="coerce")
        if pd.notna(numerator) and pd.notna(denominator) and float(denominator) != 0.0:
            return float(denominator) / float(numerator)
        return pd.NA
    numeric = pd.to_numeric(text, errors="coerce")
    return float(numeric) if pd.notna(numeric) else pd.NA


def _normalize_optional_date(frame: pd.DataFrame, *column_names: str) -> pd.Series:
    for column_name in column_names:
        if column_name in frame.columns:
            return pd.to_datetime(frame[column_name], errors="coerce")
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
