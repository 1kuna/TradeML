"""Bounded SEC Form 4 historical manifest retrieval and parsing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable, Protocol

import pandas as pd

from trademl.connectors.base import (
    BudgetBlockedConnectorError,
    RemoteRateLimitConnectorError,
)
from trademl.events.form4 import (
    Form4ManifestRow,
    Form4ParseResult,
    Form4RetrievalMetadata,
    Form4RetrievalResult,
    parse_form4_ownership_xml,
    write_form4_parse_results,
    write_form4_retrieval_artifacts,
)
from trademl.events.form4_candidates import (
    build_form4_candidate_events_from_curated,
    write_form4_candidate_events,
)
from trademl.events.form4_fixture_gate import build_sec_form4_connector_from_env


FORM4_INGEST_VERSION = "form4_ingest_v1"


class Form4ManifestRetrievalClient(Protocol):
    """Small SEC connector surface needed by bounded Form 4 ingestion."""

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch a Form 4 manifest dataframe."""

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        """Retrieve ownership XML and source metadata for one accession."""


def run_form4_ingest_from_env(
    *,
    data_root: Path,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    user_agent: str | None = None,
    max_retrieval_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    use_cache: bool = True,
) -> dict[str, object]:
    """Run a bounded Form 4 ingest using SEC settings from environment."""
    connector = build_sec_form4_connector_from_env(user_agent=user_agent)
    return run_form4_ingest(
        data_root=data_root,
        connector=connector,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        max_retrieval_attempts=max_retrieval_attempts,
        rate_limit_pause_seconds=rate_limit_pause_seconds,
        use_cache=use_cache,
    )


def run_form4_ingest(
    *,
    data_root: Path,
    connector: Form4ManifestRetrievalClient,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    max_retrieval_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    use_cache: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Fetch a SEC index manifest, retrieve filings, parse rows, and write artifacts."""
    root = Path(data_root).expanduser()
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest_frame = connector.fetch("form4_ownership", [], start_date, end_date)
    manifest_rows = [_manifest_from_record(record) for record in manifest_frame.to_dict("records")]
    manifest_rows = [row for row in manifest_rows if row is not None]
    if limit is not None:
        manifest_rows = manifest_rows[: max(0, int(limit))]
    manifest_paths = write_form4_manifest(root=root, rows=manifest_rows)

    parse_results: list[Form4ParseResult] = []
    accessions: list[dict[str, object]] = []
    for manifest in manifest_rows:
        item: dict[str, object] = {
            "accession": manifest.accession,
            "archive_cik": manifest.archive_cik,
            "filed_date": manifest.filed_date,
            "status": "FAILED",
            "errors": [],
        }
        errors: list[str] = []
        try:
            cached = (
                _read_cached_retrieval(root=root, manifest=manifest) if use_cache else None
            )
            if cached is None:
                retrieval, retry_events = _retrieve_with_retries(
                    connector=connector,
                    manifest=manifest,
                    max_attempts=max_retrieval_attempts,
                    rate_limit_pause_seconds=rate_limit_pause_seconds,
                    sleep_fn=sleep_fn,
                )
                raw_paths = write_form4_retrieval_artifacts(
                    root=root,
                    manifest=manifest,
                    retrieval=retrieval,
                )
            else:
                retrieval, raw_paths = cached
                retry_events = []
            if retry_events:
                item["retry_events"] = retry_events
            item["raw_artifacts"] = [str(path) for path in raw_paths]
            item["retrieval"] = retrieval.metadata.to_dict()
            if retrieval.ownership_xml is None:
                errors.append("missing_ownership_xml")
            else:
                raw_xml_path, complete_txt_path = _artifact_paths(raw_paths)
                parse_result = parse_form4_ownership_xml(
                    retrieval.ownership_xml,
                    manifest=manifest,
                    retrieval=retrieval.metadata,
                    first_seen_at_utc=datetime.now(timezone.utc),
                    raw_xml_path=raw_xml_path,
                    complete_txt_path=complete_txt_path,
                )
                parse_results.append(parse_result)
                item["issuer_cik"] = parse_result.issuer_cik
                item["issuer_symbol"] = parse_result.issuer_trading_symbol_raw
                item["document_type"] = parse_result.document_type
                item["nonderivative_transaction_count"] = len(
                    parse_result.nonderivative_transactions
                )
                item["derivative_transaction_count"] = parse_result.derivative_transaction_count
                item["source_quality_flags"] = parse_result.source_quality_flags
        except Exception as exc:  # pragma: no cover - live operational guard
            errors.append(f"exception:{type(exc).__name__}:{exc}")
        item["errors"] = errors
        item["status"] = "PARSED" if not errors else "FAILED"
        accessions.append(item)

    parse_artifacts = (
        [str(path) for path in write_form4_parse_results(root=root, results=parse_results)]
        if parse_results
        else []
    )
    candidate_payload: dict[str, object] | None = None
    if parse_results:
        events = build_form4_candidate_events_from_curated(root=root)
        candidate_payload = write_form4_candidate_events(root=root, events=events)

    failed = sum(1 for item in accessions if item["status"] != "PARSED")
    payload: dict[str, object] = {
        "version": FORM4_INGEST_VERSION,
        "checked_at": checked_at,
        "verdict": "PASS" if failed == 0 else "PARTIAL",
        "data_root": str(root),
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "max_retrieval_attempts": max_retrieval_attempts,
        "rate_limit_pause_seconds": rate_limit_pause_seconds,
        "cache_enabled": use_cache,
        "manifest_count": len(manifest_rows),
        "parsed_count": len(parse_results),
        "failed_count": failed,
        "manifest_artifacts": [str(path) for path in manifest_paths],
        "parse_artifacts": parse_artifacts,
        "candidate_artifacts": candidate_payload,
        "accessions": accessions,
    }
    _write_ingest_artifact(root=root, payload=payload)
    return payload


def _retrieve_with_retries(
    *,
    connector: Form4ManifestRetrievalClient,
    manifest: Form4ManifestRow,
    max_attempts: int,
    rate_limit_pause_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[Form4RetrievalResult, list[dict[str, object]]]:
    """Retrieve one Form 4 document, pausing on SEC/local rate-limit pressure."""
    retry_events: list[dict[str, object]] = []
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return connector.retrieve_form4_ownership_xml(manifest), retry_events
        except BudgetBlockedConnectorError as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = _budget_block_sleep_seconds(
                exc, default_seconds=rate_limit_pause_seconds
            )
            retry_events.append(
                {
                    "attempt": attempt,
                    "reason": "local_budget_block",
                    "sleep_seconds": sleep_seconds,
                }
            )
            sleep_fn(sleep_seconds)
        except RemoteRateLimitConnectorError as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = float(exc.retry_after_seconds or rate_limit_pause_seconds)
            retry_events.append(
                {
                    "attempt": attempt,
                    "reason": "remote_rate_limit",
                    "sleep_seconds": sleep_seconds,
                }
            )
            sleep_fn(sleep_seconds)
    raise RuntimeError("unreachable Form 4 retrieval retry state")


def _budget_block_sleep_seconds(
    exc: BudgetBlockedConnectorError, *, default_seconds: float
) -> float:
    """Return a positive sleep duration from a local budget block decision."""
    next_eligible = exc.decision.next_eligible_at
    if next_eligible is None:
        return float(default_seconds)
    now = datetime.now(timezone.utc)
    if next_eligible.tzinfo is None:
        next_eligible = next_eligible.replace(tzinfo=timezone.utc)
    return max(0.1, (next_eligible - now).total_seconds())


def _read_cached_retrieval(
    *, root: Path, manifest: Form4ManifestRow
) -> tuple[Form4RetrievalResult, list[Path]] | None:
    """Read an existing raw Form 4 retrieval artifact for resumable backfills."""
    target = _raw_retrieval_dir(root=root, manifest=manifest)
    metadata_path = target / "metadata.json"
    primary_path = target / "primary.xml"
    complete_path = target / "complete.txt"
    if not metadata_path.exists() or not primary_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_metadata = payload.get("retrieval")
    if not isinstance(raw_metadata, dict):
        return None
    try:
        ownership_xml = primary_path.read_text(encoding="utf-8")
        complete_txt = (
            complete_path.read_text(encoding="utf-8") if complete_path.exists() else None
        )
    except OSError:
        return None
    quality_flags = [
        str(item) for item in raw_metadata.get("quality_flags", []) if str(item)
    ]
    if "used_cached_raw_artifact" not in quality_flags:
        quality_flags.append("used_cached_raw_artifact")
    metadata = Form4RetrievalMetadata.from_accepted_raw(
        primary_xml_url=_optional_str(raw_metadata.get("primary_xml_url")),
        primary_xml_http_status=_optional_int(raw_metadata.get("primary_xml_http_status")),
        primary_xml_sha256=_optional_str(raw_metadata.get("primary_xml_sha256")),
        complete_txt_url=_optional_str(raw_metadata.get("complete_txt_url")),
        complete_txt_http_status=_optional_int(raw_metadata.get("complete_txt_http_status")),
        complete_txt_sha256=_optional_str(raw_metadata.get("complete_txt_sha256")),
        xml_source=_optional_str(raw_metadata.get("xml_source")) or "cached_raw",
        accepted_at_raw=_optional_str(raw_metadata.get("accepted_at_raw")),
        accepted_at_source=_optional_str(raw_metadata.get("accepted_at_source")),
        quality_flags=quality_flags,
    )
    paths = [primary_path]
    if complete_path.exists():
        paths.append(complete_path)
    paths.append(metadata_path)
    return Form4RetrievalResult(
        metadata=metadata,
        ownership_xml=ownership_xml,
        complete_txt=complete_txt,
    ), paths


def _raw_retrieval_dir(*, root: Path, manifest: Form4ManifestRow) -> Path:
    return (
        Path(root)
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / f"archive_cik={manifest.archive_cik}"
        / f"accession={manifest.accession_no_dashes}"
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_form4_manifest(*, root: Path, rows: list[Form4ManifestRow]) -> list[Path]:
    """Write SEC-index Form 4 manifest rows under the documented archive layout."""
    if not rows:
        return []
    written: list[Path] = []
    frame = pd.DataFrame([row.to_dict() for row in rows]).drop_duplicates(
        subset=["accession", "archive_cik", "index_filename"]
    )
    manifest_root = Path(root) / "data" / "raw" / "sec" / "form4_manifest"
    for (year, quarter), group in frame.groupby(["index_year", "index_quarter"]):
        path = manifest_root / f"year={int(year)}" / f"qtr={int(quarter)}" / "manifest.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            group = pd.concat([existing, group], ignore_index=True).drop_duplicates(
                subset=["accession", "archive_cik", "index_filename"]
            )
        group.sort_values(["filed_date", "accession"]).to_parquet(path, index=False)
        written.append(path)
    curated = Path(root) / "data" / "curated" / "sec" / "form4" / "manifest" / "data.parquet"
    curated.parent.mkdir(parents=True, exist_ok=True)
    if curated.exists():
        frame = pd.concat([pd.read_parquet(curated), frame], ignore_index=True)
    frame.drop_duplicates(subset=["accession", "archive_cik", "index_filename"]).sort_values(
        ["filed_date", "accession"]
    ).to_parquet(curated, index=False)
    written.append(curated)
    return written


def _manifest_from_record(record: dict[str, object]) -> Form4ManifestRow | None:
    accession = str(record.get("accession") or "").strip()
    archive_cik = str(record.get("archive_cik") or "").strip()
    index_filename = str(record.get("index_filename") or "").strip()
    if not accession or not archive_cik or not index_filename:
        return None
    return Form4ManifestRow(
        archive_cik=archive_cik,
        form=str(record.get("form") or "4"),
        filed_date=str(record.get("filed_date") or ""),
        index_filename=index_filename,
        accession=accession,
        accession_no_dashes=str(record.get("accession_no_dashes") or accession.replace("-", "")),
        discovery_source=str(record.get("discovery_source") or "sec_full_index"),
        index_year=int(record.get("index_year") or 0),
        index_quarter=int(record.get("index_quarter") or 0),
        index_file_hash=str(record.get("index_file_hash") or ""),
        index_crawled_at=str(record.get("index_crawled_at") or ""),
    )


def _artifact_paths(paths: list[Path]) -> tuple[str | None, str | None]:
    raw_xml_path = None
    complete_txt_path = None
    for path in paths:
        if path.name == "primary.xml":
            raw_xml_path = str(path)
        elif path.name == "complete.txt":
            complete_txt_path = str(path)
    return raw_xml_path, complete_txt_path


def _write_ingest_artifact(*, root: Path, payload: dict[str, object]) -> None:
    target = Path(root) / "control" / "cluster" / "state" / "research" / "form4_ingest"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(payload["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
