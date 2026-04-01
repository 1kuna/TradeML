"""SEC EDGAR connector."""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd

from trademl.connectors.base import HTTPConnector


class SecEdgarConnector(HTTPConnector):
    """Fetch filing history from SEC EDGAR submissions API."""

    vendor_name = "sec_edgar"

    def __init__(self, *, user_agent: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.user_agent = user_agent

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch filing index rows for supplied CIKs."""
        if dataset != "filing_index":
            raise ValueError(f"unsupported dataset for sec_edgar: {dataset}")

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        frames = []
        for cik in symbols:
            normalized_cik = str(cik).zfill(10)
            payload = self.request_json(endpoint=f"/submissions/CIK{normalized_cik}.json")
            recent = pd.DataFrame(payload.get("filings", {}).get("recent", {}))
            if recent.empty:
                continue
            recent["cik"] = normalized_cik
            recent["filingDate"] = pd.to_datetime(recent["filingDate"])
            filtered = recent.loc[
                recent["filingDate"].between(start_ts.normalize(), end_ts.normalize())
                & recent["form"].isin(["8-K", "10-K", "10-Q"])
            ]
            frames.append(filtered)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
