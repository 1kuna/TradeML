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
            return self.request_csv(
                endpoint="/query",
                params={"function": "LISTING_STATUS", "date": pd.Timestamp(end_date).strftime("%Y-%m-%d"), "state": "active"},
            )
        raise ValueError(f"unsupported dataset for alpha_vantage: {dataset}")
