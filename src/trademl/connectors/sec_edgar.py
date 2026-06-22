"""SEC EDGAR connector."""

from __future__ import annotations

import contextlib
from datetime import date as date_type
from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path
import re
import time
from urllib.parse import quote
import uuid

import pandas as pd
import requests

from trademl.connectors.base import (
    HTTPConnector,
    BudgetBlockedConnectorError,
    PermanentConnectorError,
    RemoteRateLimitConnectorError,
    TemporaryConnectorError,
)
from trademl.events.form4 import (
    Form4ManifestRow,
    Form4RetrievalMetadata,
    Form4RetrievalResult,
    extract_primary_ownership_xml_from_complete_txt,
    parse_form4_index_manifest,
    sgml_tag,
)
from trademl.events.sec8k import parse_sec8k_index_manifest


class MissingCompanyfactsError(PermanentConnectorError):
    """SEC has no companyfacts object for this CIK."""

    def __init__(self, *, cik: str, message: str) -> None:
        super().__init__(message)
        self.cik = cik


class SecEdgarConnector(HTTPConnector):
    """Fetch filing history from SEC EDGAR submissions API."""

    vendor_name = "sec_edgar"

    def __init__(
        self,
        *,
        user_agent: str,
        max_complete_text_bytes: int = 256 * 1024 * 1024,
        max_complete_text_seconds: float = 120.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.user_agent = user_agent
        self.max_complete_text_bytes = int(max_complete_text_bytes)
        self.max_complete_text_seconds = float(max_complete_text_seconds)

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    @staticmethod
    def parse_form4_index_manifest(
        text: str,
        *,
        index_year: int,
        index_quarter: int,
        index_crawled_at: str,
        discovery_source: str = "sec_full_index",
    ) -> list[Form4ManifestRow]:
        """Parse SEC full-index text into Form 4 manifest rows."""
        return parse_form4_index_manifest(
            text,
            index_year=index_year,
            index_quarter=index_quarter,
            index_crawled_at=index_crawled_at,
            discovery_source=discovery_source,
        )

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        """Retrieve authoritative ownership XML with complete-text fallback."""
        quality_flags: list[str] = []
        primary_url = (
            raw_primary_xml_url(
                manifest.archive_cik, manifest.accession, primary_document
            )
            if primary_document
            else None
        )
        primary_status: int | None = None
        primary_hash: str | None = None
        ownership_xml: str | None = None
        if primary_url is not None:
            primary_status, primary_text = self._request_text_status(
                primary_url, endpoint_key="form4_primary_xml"
            )
            if primary_status < 400 and "<ownershipDocument" in primary_text:
                ownership_xml = primary_text
                primary_hash = _sha256_text(primary_text)
            else:
                if primary_status == 404:
                    quality_flags.append("raw_primary_404")
                else:
                    quality_flags.append("raw_primary_unusable")

        complete_url = complete_txt_url_from_index_filename(manifest.index_filename)
        complete_status: int | None = None
        complete_hash: str | None = None
        complete_txt: str | None = None
        if ownership_xml is None:
            complete_status, complete_txt = self._request_text_status(
                complete_url, endpoint_key="form4_complete_txt"
            )
            if complete_status < 400:
                complete_hash = _sha256_text(complete_txt)
                extracted = extract_primary_ownership_xml_from_complete_txt(
                    complete_txt
                )
                if extracted is not None:
                    ownership_xml = extracted
                    quality_flags.append("used_complete_txt_fallback")
                else:
                    quality_flags.append("no_primary_xml")

        accepted_raw, accepted_source = _accepted_from_sources(
            submissions_metadata=submissions_metadata,
            complete_txt=complete_txt,
        )
        if accepted_raw is None and ownership_xml is not None:
            accepted_raw, accepted_source = self._retrieve_form4_accepted_fallbacks(
                manifest=manifest, quality_flags=quality_flags
            )
        xml_source = "failed"
        if ownership_xml is not None and primary_hash is not None:
            xml_source = "raw_primary"
        elif ownership_xml is not None:
            xml_source = "complete_txt_extracted"
        metadata = Form4RetrievalMetadata.from_accepted_raw(
            primary_xml_url=primary_url,
            primary_xml_http_status=primary_status,
            primary_xml_sha256=primary_hash,
            complete_txt_url=complete_url,
            complete_txt_http_status=complete_status,
            complete_txt_sha256=complete_hash,
            xml_source=xml_source,
            accepted_at_raw=accepted_raw,
            accepted_at_source=accepted_source,
            quality_flags=quality_flags,
        )
        return Form4RetrievalResult(
            metadata=metadata, ownership_xml=ownership_xml, complete_txt=complete_txt
        )

    def _retrieve_form4_accepted_fallbacks(
        self, *, manifest: Form4ManifestRow, quality_flags: list[str]
    ) -> tuple[str | None, str | None]:
        """Fetch SGML/index accepted timestamps when XML retrieval lacks one."""
        header_status, header_text = self._request_text_status(
            sgml_header_url(manifest.archive_cik, manifest.accession),
            endpoint_key="form4_sgml_header",
        )
        if header_status < 400:
            raw = sgml_tag(header_text, "ACCEPTANCE-DATETIME")
            if raw:
                quality_flags.append("used_sgml_header_accepted_at")
                return raw, "sgml_header"
        elif header_status == 404:
            quality_flags.append("sgml_header_404")
        else:
            quality_flags.append("sgml_header_unusable")

        index_status, index_text = self._request_text_status(
            accession_index_url(manifest.archive_cik, manifest.accession),
            endpoint_key="form4_accession_index",
        )
        if index_status < 400:
            raw = _accepted_from_accession_index_html(index_text)
            if raw:
                quality_flags.append("used_accession_index_accepted_at")
                return raw, "accession_index"
        elif index_status == 404:
            quality_flags.append("accession_index_404")
        else:
            quality_flags.append("accession_index_unusable")
        return None, None

    def _request_text_status(self, url: str, *, endpoint_key: str) -> tuple[int, str]:
        """Fetch text while returning non-retryable HTTP statuses for fallbacks."""
        telemetry_key = endpoint_key
        for attempt in range(1, self.retry_config.max_attempts + 1):
            decision = self.budget_manager.budget_decision(
                self.vendor_name, task_kind="OTHER", units=1
            )
            if not decision.allowed:
                self.budget_manager.record_local_budget_block(
                    self.vendor_name, endpoint=telemetry_key
                )
                raise BudgetBlockedConnectorError(self.vendor_name, decision)
            try:
                response = self.session.request(
                    method="GET",
                    url=url,
                    params=self._auth_params(),
                    headers=self._headers(),
                    timeout=30,
                    stream=endpoint_key == "sec8k_complete_txt",
                )
            except requests.RequestException as exc:
                self.budget_manager.record_spend(
                    self.vendor_name,
                    task_kind="OTHER",
                    units=1,
                    endpoint=telemetry_key,
                    logical_units=1,
                )
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(self._sleep_duration(attempt))
                    continue
                raise TemporaryConnectorError(
                    f"{self.vendor_name} request failed: {exc}"
                ) from exc
            self.budget_manager.record_spend(
                self.vendor_name,
                task_kind="OTHER",
                units=1,
                endpoint=telemetry_key,
                logical_units=1,
            )
            response_text = (
                self._bounded_complete_text(response=response, endpoint_key=endpoint_key)
                if endpoint_key == "sec8k_complete_txt"
                else response.text
            )
            if response.status_code == 429:
                self.budget_manager.record_remote_rate_limit(
                    self.vendor_name, endpoint=telemetry_key
                )
                retry_after_seconds = self._retry_after_seconds(response, endpoint_key)
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(
                        retry_after_seconds or self._sleep_duration(attempt)
                    )
                    continue
                raise RemoteRateLimitConnectorError(
                    self.vendor_name, retry_after_seconds=retry_after_seconds
                )
            if response.status_code in self._retryable_statuses(endpoint_key):
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(
                        self._retry_after_seconds(response, endpoint_key)
                        or self._sleep_duration(attempt)
                    )
                    continue
                raise TemporaryConnectorError(
                    f"{self.vendor_name} request failed: "
                    f"{response.status_code} {response_text[:512]}"
                )
            return response.status_code, response_text
        raise TemporaryConnectorError(f"{self.vendor_name} request failed after retries")

    def _bounded_complete_text(
        self, *, response: requests.Response, endpoint_key: str
    ) -> str:
        """Read SEC complete text with total byte/time guards."""
        max_bytes = max(0, int(self.max_complete_text_bytes))
        max_seconds = max(0.0, float(self.max_complete_text_seconds))
        content_length = response.headers.get("Content-Length")
        if max_bytes and content_length:
            with contextlib.suppress(ValueError):
                if int(content_length) > max_bytes:
                    response.close()
                    raise PermanentConnectorError(
                        f"{self.vendor_name} {endpoint_key} exceeded max bytes from header: "
                        f"bytes={content_length} max_bytes={max_bytes}"
                    )
        start = time.perf_counter()
        total = 0
        chunks: list[bytes] = []
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if max_seconds and (time.perf_counter() - start) > max_seconds:
                    raise TemporaryConnectorError(
                        f"{self.vendor_name} {endpoint_key} exceeded download deadline: "
                        f"elapsed_seconds={time.perf_counter() - start:.2f} "
                        f"max_seconds={max_seconds:.2f}"
                    )
                if not chunk:
                    continue
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    raise PermanentConnectorError(
                        f"{self.vendor_name} {endpoint_key} exceeded max bytes: "
                        f"bytes={total} max_bytes={max_bytes}"
                    )
                chunks.append(chunk)
        finally:
            response.close()
        encoding = getattr(response, "encoding", None) or "utf-8"
        return b"".join(chunks).decode(encoding, errors="replace")

    def stream_companyfacts_to_gzip(self, *, cik: str, output: Path) -> dict[str, object]:
        """Stream a SEC companyfacts payload to gzip without building a JSON object."""
        normalized_cik = str(cik).zfill(10)
        endpoint = f"/api/xbrl/companyfacts/CIK{normalized_cik}.json"
        telemetry_key = "companyfacts"
        for attempt in range(1, self.retry_config.max_attempts + 1):
            if not self.budget_manager.can_spend(
                self.vendor_name, task_kind="OTHER", units=1
            ):
                self.budget_manager.record_local_budget_block(
                    self.vendor_name, endpoint=telemetry_key
                )
                raise TemporaryConnectorError(
                    f"budget exhausted for vendor={self.vendor_name}"
                )
            start = time.perf_counter()
            try:
                response = self.session.request(
                    method="GET",
                    url=f"{self.base_url}{endpoint}",
                    params=self._auth_params(),
                    headers=self._headers(),
                    timeout=30,
                    stream=True,
                )
            except requests.RequestException as exc:
                self.budget_manager.record_spend(
                    self.vendor_name,
                    task_kind="OTHER",
                    units=1,
                    endpoint=telemetry_key,
                    logical_units=1,
                )
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(self._sleep_duration(attempt))
                    continue
                raise TemporaryConnectorError(
                    f"{self.vendor_name} request failed: {exc}"
                ) from exc
            self.budget_manager.record_spend(
                self.vendor_name,
                task_kind="OTHER",
                units=1,
                endpoint=telemetry_key,
                logical_units=1,
            )
            if response.status_code < 400:
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
                raw_bytes = 0
                try:
                    with gzip.open(tmp_path, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            raw_bytes += len(chunk)
                            handle.write(chunk)
                    os.replace(tmp_path, output)
                finally:
                    with contextlib.suppress(Exception):
                        response.close()
                self._log_request(
                    endpoint=endpoint,
                    symbols=[normalized_cik],
                    rows=1,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
                return {
                    "cik": normalized_cik,
                    "facts_path": str(output),
                    "raw_bytes": raw_bytes,
                }
            response_text = response.text[:512] if response.text else ""
            if response.status_code == 429:
                self.budget_manager.record_remote_rate_limit(
                    self.vendor_name, endpoint=telemetry_key
                )
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(
                        self._retry_after_seconds(response, telemetry_key)
                        or self._sleep_duration(attempt)
                    )
                    continue
                raise self._rate_limit_error()
            if response.status_code in self._retryable_statuses(telemetry_key):
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(
                        self._retry_after_seconds(response, telemetry_key)
                        or self._sleep_duration(attempt)
                    )
                    continue
                raise TemporaryConnectorError(
                    f"{self.vendor_name} request failed: {response.status_code} {response_text}"
                )
            if response.status_code == 404 and "NoSuchKey" in response_text:
                raise MissingCompanyfactsError(
                    cik=normalized_cik,
                    message=f"SEC companyfacts missing for CIK{normalized_cik}",
                )
            self.budget_manager.record_permanent_failure(
                self.vendor_name, endpoint=telemetry_key
            )
            raise PermanentConnectorError(
                f"{self.vendor_name} request failed: {response.status_code} {response_text}"
            )
        raise TemporaryConnectorError(f"{self.vendor_name} request failed after retries")

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
        if dataset == "form4_ownership":
            return self._fetch_form4_manifest(start_date=start_date, end_date=end_date)
        if dataset == "sec8k_index":
            return self._fetch_sec8k_manifest(start_date=start_date, end_date=end_date)
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

    def _fetch_form4_manifest(
        self,
        *,
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch SEC quarterly full-index manifests for Form 4 discovery."""
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        crawled_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        rows: list[Form4ManifestRow] = []
        for year, quarter in _quarters_between(start_ts, end_ts):
            url = (
                "https://www.sec.gov/Archives/edgar/full-index/"
                f"{year}/QTR{quarter}/form.idx"
            )
            response = self._request(
                method="GET",
                endpoint="",
                endpoint_key="form4_ownership",
                absolute_url=url,
                task_kind="OTHER",
                logical_units=1,
            )
            rows.extend(
                parse_form4_index_manifest(
                    response.text,
                    index_year=year,
                    index_quarter=quarter,
                    index_crawled_at=crawled_at,
                )
            )
        frame = pd.DataFrame([row.to_dict() for row in rows])
        if frame.empty:
            return frame
        frame["filed_date_ts"] = pd.to_datetime(frame["filed_date"], errors="coerce")
        frame = frame.loc[
            frame["filed_date_ts"].between(start_ts, end_ts)
        ].drop(columns=["filed_date_ts"])
        return frame.reset_index(drop=True)

    def _fetch_sec8k_manifest(
        self,
        *,
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch SEC quarterly full-index manifests for exact 8-K discovery."""
        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        crawled_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        rows = []
        for year, quarter in _quarters_between(start_ts, end_ts):
            url = (
                "https://www.sec.gov/Archives/edgar/full-index/"
                f"{year}/QTR{quarter}/form.idx"
            )
            response = self._request(
                method="GET",
                endpoint="",
                endpoint_key="sec8k_index",
                absolute_url=url,
                task_kind="OTHER",
                logical_units=1,
            )
            rows.extend(
                parse_sec8k_index_manifest(
                    response.text,
                    index_year=year,
                    index_quarter=quarter,
                    index_crawled_at=crawled_at,
                )
            )
        frame = pd.DataFrame([row.to_dict() for row in rows])
        if frame.empty:
            return frame
        frame["filed_date_ts"] = pd.to_datetime(frame["filed_date"], errors="coerce")
        frame = frame.loc[
            frame["filed_date_ts"].between(start_ts, end_ts)
        ].drop(columns=["filed_date_ts"])
        return frame.reset_index(drop=True)

    def retrieve_complete_submission_text(
        self, *, index_filename: str, endpoint_key: str = "sec_complete_txt"
    ) -> tuple[int, str, str]:
        """Retrieve complete SEC submission text from an index filename."""
        url = complete_txt_url_from_index_filename(index_filename)
        status, text = self._request_text_status(url, endpoint_key=endpoint_key)
        return status, text, url

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


def sec_archive_dir(archive_cik: str | int, accession: str) -> str:
    """Return the SEC accession directory URL from archive CIK and accession."""
    cik_no_leading_zeros = str(int(str(archive_cik)))
    accession_no_dashes = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/"
    )


def raw_primary_xml_url(
    archive_cik: str | int, accession: str, primary_document: str
) -> str:
    """Return the raw primary ownership XML URL."""
    return sec_archive_dir(archive_cik, accession) + quote(primary_document)


def complete_txt_url_from_index_filename(index_filename: str) -> str:
    """Return the complete submission text URL from SEC index filename."""
    normalized = index_filename.lstrip("/")
    match = re.fullmatch(r"edgar/data/(\d+)/([^/]+)\.txt", normalized)
    if match is not None:
        archive_cik, accession = match.groups()
        normalized = (
            f"edgar/data/{archive_cik}/{accession.replace('-', '')}/{accession}.txt"
        )
    return "https://www.sec.gov/Archives/" + normalized


def complete_txt_url_from_parts(archive_cik: str | int, accession: str) -> str:
    """Return the complete submission text URL from archive CIK and accession."""
    cik_no_leading_zeros = str(int(str(archive_cik)))
    accession_no_dashes = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/{accession}.txt"
    )


def accession_index_url(archive_cik: str | int, accession: str) -> str:
    """Return the SEC accession HTML index URL."""
    return sec_archive_dir(archive_cik, accession) + f"{accession}-index.htm"


def accession_directory_index_json_url(archive_cik: str | int, accession: str) -> str:
    """Return the SEC accession directory JSON index URL."""
    return sec_archive_dir(archive_cik, accession) + "index.json"


def sgml_header_url(archive_cik: str | int, accession: str) -> str:
    """Return the SEC accession SGML header URL."""
    return sec_archive_dir(archive_cik, accession) + f"{accession}.hdr.sgml"


def _quarters_between(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> list[tuple[int, int]]:
    quarters: list[tuple[int, int]] = []
    current = pd.Timestamp(year=start_ts.year, month=((start_ts.month - 1) // 3) * 3 + 1, day=1)
    while current <= end_ts:
        quarter = ((current.month - 1) // 3) + 1
        quarters.append((int(current.year), int(quarter)))
        current = current + pd.DateOffset(months=3)
    return quarters


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _accepted_from_sources(
    *,
    submissions_metadata: dict[str, object] | None,
    complete_txt: str | None,
) -> tuple[str | None, str | None]:
    if submissions_metadata:
        raw = submissions_metadata.get("acceptanceDateTime")
        if raw:
            return str(raw), "submissions"
    if complete_txt:
        raw = sgml_tag(complete_txt, "ACCEPTANCE-DATETIME")
        if raw:
            return raw, "sgml_header"
    return None, None


def _accepted_from_accession_index_html(text: str) -> str | None:
    match = re.search(
        r"Accepted\s*(?:</[^>]+>\s*)*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1)
