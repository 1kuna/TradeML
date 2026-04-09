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
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch filing index rows for supplied CIKs."""
        if dataset == "company_tickers":
            payload = self.request_json(
                base_url="https://www.sec.gov",
                endpoint="/files/company_tickers.json",
                endpoint_key="company_tickers",
            )
            rows = payload.values() if isinstance(payload, dict) else payload
            return pd.DataFrame(rows)
        if dataset == "companyfacts":
            frames = []
            for cik in symbols:
                normalized_cik = str(cik).zfill(10)
                payload = self.request_json(endpoint=f"/api/xbrl/companyfacts/CIK{normalized_cik}.json", endpoint_key="companyfacts", logical_units=1)
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "cik": normalized_cik,
                                "entityName": payload.get("entityName"),
                                "facts": payload.get("facts", {}),
                            }
                        ]
                    )
                )
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["cik", "entityName", "facts"])
        if dataset != "filing_index":
            raise ValueError(f"unsupported dataset for sec_edgar: {dataset}")

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        frames = []
        for cik in symbols:
            normalized_cik = str(cik).zfill(10)
            payload = self.request_json(endpoint=f"/submissions/CIK{normalized_cik}.json", endpoint_key="filing_index", logical_units=1)
            recent = self._normalize_filing_rows(payload.get("filings", {}).get("recent", {}), cik=normalized_cik)
            if not recent.empty:
                frames.append(self._filter_filings(recent, start_ts=start_ts, end_ts=end_ts))
            for metadata in payload.get("filings", {}).get("files", []) or []:
                if not self._submission_segment_overlaps(metadata, start_ts=start_ts, end_ts=end_ts):
                    continue
                name = str(metadata.get("name") or "").strip()
                if not name:
                    continue
                archive_payload = self.request_json(endpoint=f"/submissions/{name}", endpoint_key="filing_index", logical_units=1)
                archive_rows = self._normalize_filing_rows(archive_payload, cik=normalized_cik)
                if archive_rows.empty:
                    continue
                frames.append(self._filter_filings(archive_rows, start_ts=start_ts, end_ts=end_ts))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _normalize_filing_rows(payload: object, *, cik: str) -> pd.DataFrame:
        """Normalize a SEC recent-filings payload into a dataframe."""
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame()
        frame["cik"] = cik
        frame["filingDate"] = pd.to_datetime(frame.get("filingDate"), errors="coerce")
        return frame

    @staticmethod
    def _filter_filings(frame: pd.DataFrame, *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Filter filing rows down to the forms and date window we care about."""
        if frame.empty:
            return frame
        return frame.loc[
            frame["filingDate"].between(start_ts.normalize(), end_ts.normalize())
            & frame["form"].isin(["8-K", "10-K", "10-Q"])
        ].copy()

    @staticmethod
    def _submission_segment_overlaps(metadata: object, *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Return whether an archived submissions segment overlaps the requested window."""
        if not isinstance(metadata, dict):
            return False
        filing_from = pd.to_datetime(metadata.get("filingFrom"), errors="coerce")
        filing_to = pd.to_datetime(metadata.get("filingTo"), errors="coerce")
        if pd.notna(filing_from) and pd.notna(filing_to):
            return filing_to.normalize() >= start_ts.normalize() and filing_from.normalize() <= end_ts.normalize()
        date_range = str(metadata.get("dateRange") or "").strip()
        if " to " in date_range:
            left, right = date_range.split(" to ", 1)
            range_start = pd.to_datetime(left, errors="coerce")
            range_end = pd.to_datetime(right, errors="coerce")
            if pd.notna(range_start) and pd.notna(range_end):
                return range_end.normalize() >= start_ts.normalize() and range_start.normalize() <= end_ts.normalize()
        return True
