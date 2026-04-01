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
            payload = self.request_json(endpoint="/stable/delisted-companies")
            return pd.DataFrame(payload)
        if dataset == "earnings_calendar":
            payload = self.request_json(
                endpoint="/stable/earnings-calendar",
                params={"from": pd.Timestamp(start_date).strftime("%Y-%m-%d"), "to": pd.Timestamp(end_date).strftime("%Y-%m-%d")},
            )
            return pd.DataFrame(payload)
        raise ValueError(f"unsupported dataset for fmp: {dataset}")
