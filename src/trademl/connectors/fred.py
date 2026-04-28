"""FRED / ALFRED connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector

FRED_PAGE_LIMIT = 10000


class FredConnector(HTTPConnector):
    """Fetch macro series and vintage dates from FRED."""

    vendor_name = "fred"

    def _auth_params(self) -> dict[str, str]:
        return {"api_key": self.api_key or "", "file_type": "json"}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch normalized FRED datasets."""
        if dataset == "macros_treasury":
            return self._fetch_observations(
                symbols=symbols, start_date=start_date, end_date=end_date
            )
        if dataset == "vintagedates":
            frames = []
            for series_id in symbols:
                vintage_dates = []
                offset = 0
                while True:
                    payload = self.request_json(
                        endpoint="/fred/series/vintagedates",
                        endpoint_key="vintagedates",
                        params={
                            "series_id": series_id,
                            "limit": FRED_PAGE_LIMIT,
                            "offset": offset,
                        },
                        task_kind="OTHER",
                        logical_units=1,
                    )
                    page = list(payload.get("vintage_dates", []))
                    vintage_dates.extend(page)
                    if len(page) < FRED_PAGE_LIMIT:
                        break
                    offset += FRED_PAGE_LIMIT
                frames.append(
                    pd.DataFrame(
                        {
                            "series_id": series_id,
                            "vintage_date": vintage_dates,
                        }
                    )
                )
            if not frames:
                self.budget_manager.record_empty_success(
                    self.vendor_name, endpoint="vintagedates"
                )
                return pd.DataFrame()
            result = pd.concat(frames, ignore_index=True)
            if result.empty:
                self.budget_manager.record_empty_success(
                    self.vendor_name, endpoint="vintagedates"
                )
            return result
        raise ValueError(f"unsupported dataset for fred: {dataset}")

    def _fetch_observations(
        self,
        *,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        frames = []
        for series_id in symbols:
            observations = []
            offset = 0
            while True:
                payload = self.request_json(
                    endpoint="/fred/series/observations",
                    endpoint_key="macros_treasury",
                    params={
                        "series_id": series_id,
                        "observation_start": pd.Timestamp(start_date).strftime(
                            "%Y-%m-%d"
                        ),
                        "observation_end": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                        "limit": FRED_PAGE_LIMIT,
                        "offset": offset,
                        "sort_order": "asc",
                    },
                    task_kind="OTHER",
                    logical_units=1,
                )
                page = list(payload.get("observations", []))
                observations.extend(page)
                if len(page) < FRED_PAGE_LIMIT:
                    break
                offset += FRED_PAGE_LIMIT
            frame = pd.DataFrame(observations)
            if frame.empty:
                continue
            frame["series_id"] = series_id
            frames.append(frame)
        if not frames:
            self.budget_manager.record_empty_success(
                self.vendor_name, endpoint="macros_treasury"
            )
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "observation_date",
                    "value",
                    "vintage_date",
                    "ingested_at",
                ]
            )
        observations = pd.concat(frames, ignore_index=True)
        observations["observation_date"] = pd.to_datetime(observations["date"]).dt.date
        observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
        observations["vintage_date"] = pd.to_datetime(
            observations.get("realtime_start"), errors="coerce"
        ).dt.date
        observations["ingested_at"] = pd.Timestamp.now(tz="UTC")
        return observations[
            ["series_id", "observation_date", "value", "vintage_date", "ingested_at"]
        ]
