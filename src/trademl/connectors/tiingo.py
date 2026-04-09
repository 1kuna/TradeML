"""Tiingo connector."""

from __future__ import annotations

from datetime import UTC, date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class TiingoConnector(HTTPConnector):
    """Fetch Tiingo bars, corporate actions, fundamentals, and ticker metadata."""

    vendor_name = "tiingo"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key or ''}"}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch a normalized Tiingo dataset."""
        if dataset == "equities_eod":
            return self._fetch_equities(symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "corp_actions_dividends":
            return self._fetch_corporate_actions(
                endpoint="/tiingo/corporate-actions/distributions",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                event_type="dividend",
            )
        if dataset == "corp_actions_splits":
            return self._fetch_corporate_actions(
                endpoint="/tiingo/corporate-actions/splits",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                event_type="split",
            )
        if dataset == "fundamentals":
            return self._fetch_fundamentals(symbols=symbols, start_date=start_date, end_date=end_date)
        if dataset == "supported_tickers":
            payload = self.request_json(endpoint="/tiingo/daily", endpoint_key="supported_tickers", task_kind="OTHER")
            frame = pd.DataFrame(payload)
            if frame.empty:
                return pd.DataFrame(columns=["symbol", "name", "exchange", "asset_type", "start_date", "end_date"])
            normalized = pd.DataFrame(
                {
                    "symbol": frame.get("ticker", frame.get("symbol", pd.Series(dtype="string"))).astype("string").str.upper(),
                    "name": frame.get("name", pd.Series(dtype="string")).astype("string"),
                    "exchange": frame.get("exchangeCode", frame.get("exchange", pd.Series(dtype="string"))).astype("string").str.upper(),
                    "asset_type": frame.get("assetType", pd.Series(dtype="string")).astype("string"),
                    "start_date": pd.to_datetime(frame.get("startDate"), errors="coerce").dt.normalize(),
                    "end_date": pd.to_datetime(frame.get("endDate"), errors="coerce").dt.normalize(),
                }
            )
            return normalized.dropna(subset=["symbol"]).drop_duplicates().sort_values("symbol").reset_index(drop=True)
        raise ValueError(f"unsupported dataset for tiingo: {dataset}")

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
                endpoint=f"/tiingo/daily/{symbol}/prices",
                endpoint_key="equities_eod",
                params={"startDate": start, "endDate": end, "resampleFreq": "daily"},
                task_kind="FORWARD",
                logical_units=1,
            )
            frame = pd.DataFrame(payload)
            if frame.empty:
                continue
            frame["symbol"] = symbol
            frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
            frame["source_name"] = self.vendor_name
            frame["source_uri"] = f"/tiingo/daily/{symbol}/prices"
            frame["vendor_ts"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True)
            frame["date"] = frame["vendor_ts"].dt.date
            for column in ["open", "high", "low", "close", "volume"]:
                frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
            frame["adjClose"] = pd.to_numeric(frame.get("adjClose"), errors="coerce")
            frame["adjVolume"] = pd.to_numeric(frame.get("adjVolume"), errors="coerce")
            frame["divCash"] = pd.to_numeric(frame.get("divCash"), errors="coerce")
            frame["splitFactor"] = pd.to_numeric(frame.get("splitFactor"), errors="coerce")
            frame["vwap"] = pd.NA
            frame["trade_count"] = pd.NA
            normalized = frame.rename(
                columns={
                    "adjClose": "adj_close",
                    "adjVolume": "adj_volume",
                    "divCash": "div_cash",
                    "splitFactor": "split_factor",
                }
            )
            frames.append(normalized)
        if not frames:
            self.budget_manager.record_empty_success(self.vendor_name, endpoint="equities_eod")
            return pd.DataFrame(columns=self._equity_columns())
        return pd.concat(frames, ignore_index=True)[self._equity_columns()].sort_values(["date", "symbol"]).reset_index(drop=True)

    def _fetch_corporate_actions(
        self,
        *,
        endpoint: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
        event_type: str,
    ) -> pd.DataFrame:
        params = {
            "startDate": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "endDate": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        }
        if symbols:
            params["tickers"] = ",".join(symbols)
        payload = self.request_json(
            endpoint=endpoint,
            endpoint_key="corp_actions_dividends" if event_type == "dividend" else "corp_actions_splits",
            params=params,
            task_kind="OTHER",
            logical_units=max(1, len(symbols)),
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=self._action_columns())
        frame["symbol"] = frame.get("ticker", frame.get("symbol", pd.Series(dtype="string"))).astype("string").str.upper()
        frame["event_type"] = event_type
        frame["ex_date"] = pd.to_datetime(frame.get("exDate", frame.get("ex_date", frame.get("date"))), errors="coerce").dt.date
        frame["record_date"] = _normalize_optional_date(frame, "recordDate", "record_date")
        frame["pay_date"] = pd.to_datetime(
            _normalize_optional_date(frame, "payDate", "paymentDate", "payment_date"),
            errors="coerce",
        ).dt.date
        if event_type == "split":
            frame["ratio"] = self._normalize_split_ratio(frame)
            frame["amount"] = pd.NA
        else:
            frame["amount"] = pd.to_numeric(
                frame.get("distribution", frame.get("amount", frame.get("cash", frame.get("divCash")))),
                errors="coerce",
            )
            frame["ratio"] = pd.NA
        frame["source"] = self.vendor_name
        frame["source_count"] = 1
        frame["ingested_at"] = pd.Timestamp.now(tz=UTC)
        normalized = frame[self._action_columns()].dropna(subset=["symbol", "ex_date"]).reset_index(drop=True)
        return normalized

    def _fetch_fundamentals(
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
            daily_payload = self.request_json(
                endpoint=f"/tiingo/fundamentals/{symbol}/daily",
                endpoint_key="fundamentals_daily",
                params={"startDate": start, "endDate": end},
                task_kind="OTHER",
                logical_units=1,
            )
            frames.extend(self._normalize_fundamental_payload(symbol=symbol, payload=daily_payload, statement_type="daily"))
            statements_payload = self.request_json(
                endpoint=f"/tiingo/fundamentals/{symbol}/statements",
                endpoint_key="fundamentals_statements",
                params={"startDate": start, "endDate": end},
                task_kind="OTHER",
                logical_units=1,
            )
            frames.extend(self._normalize_fundamental_payload(symbol=symbol, payload=statements_payload, statement_type="statements"))
        if not frames:
            return pd.DataFrame(columns=["symbol", "statement_type", "date", "as_of_date", "data"])
        frame = pd.DataFrame(frames)
        return frame.sort_values(["symbol", "statement_type", "date"]).reset_index(drop=True)

    @staticmethod
    def _normalize_fundamental_payload(
        *,
        symbol: str,
        payload: object,
        statement_type: str,
    ) -> list[dict[str, object]]:
        rows = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
        normalized: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            as_of_date = pd.to_datetime(row.get("date", row.get("asOfDate", row.get("reportDate"))), errors="coerce")
            normalized.append(
                {
                    "symbol": symbol,
                    "statement_type": statement_type,
                    "date": as_of_date.normalize() if pd.notna(as_of_date) else pd.NaT,
                    "as_of_date": as_of_date.normalize() if pd.notna(as_of_date) else pd.NaT,
                    "data": row,
                }
            )
        return normalized

    @staticmethod
    def _normalize_split_ratio(frame: pd.DataFrame) -> pd.Series:
        factor = pd.to_numeric(frame.get("splitFactor"), errors="coerce")
        if not factor.isna().all():
            return factor
        split_from = pd.to_numeric(frame.get("splitFrom", frame.get("split_from")), errors="coerce")
        split_to = pd.to_numeric(frame.get("splitTo", frame.get("split_to")), errors="coerce")
        return split_from / split_to

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
            "adj_close",
            "adj_volume",
            "div_cash",
            "split_factor",
        ]

    @staticmethod
    def _action_columns() -> list[str]:
        return ["symbol", "event_type", "ex_date", "record_date", "pay_date", "ratio", "amount", "source", "source_count", "ingested_at"]


def _normalize_optional_date(frame: pd.DataFrame, *column_names: str) -> pd.Series:
    for column_name in column_names:
        if column_name in frame.columns:
            return pd.to_datetime(frame[column_name], errors="coerce")
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
