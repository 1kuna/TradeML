"""Bounded SEC 8-K full-index ingest and candidate curation."""

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
from trademl.events.form4_ingest import _budget_block_sleep_seconds
from trademl.events.form4_fixture_gate import build_sec_form4_connector_from_env
from trademl.events.sec8k import (
    ITEM_FAMILIES,
    Sec8KManifestRow,
    build_sec8k_item_candidates,
    parse_sec8k_complete_text,
    write_sec8k_item_candidates,
)


SEC8K_INGEST_VERSION = "sec8k_ingest_v1"


class Sec8KIngestClient(Protocol):
    """Small SEC connector surface needed by bounded 8-K ingestion."""

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch a SEC dataframe."""

    def retrieve_complete_submission_text(
        self, *, index_filename: str, endpoint_key: str = "sec8k_complete_txt"
    ) -> tuple[int, str, str]:
        """Fetch complete SEC submission text from an index filename."""


def run_sec8k_ingest_from_env(
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
    """Run a bounded 8-K ingest using SEC settings from environment."""
    connector = build_sec_form4_connector_from_env(user_agent=user_agent)
    return run_sec8k_ingest(
        data_root=data_root,
        connector=connector,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        max_retrieval_attempts=max_retrieval_attempts,
        rate_limit_pause_seconds=rate_limit_pause_seconds,
        use_cache=use_cache,
    )


def run_sec8k_ingest(
    *,
    data_root: Path,
    connector: Sec8KIngestClient,
    start_date: str,
    end_date: str,
    limit: int | None = None,
    max_retrieval_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    use_cache: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Fetch SEC 8-K full-index rows, retrieve complete text, and write candidates."""
    root = Path(data_root).expanduser()
    checked_at = _now_iso()
    manifest_frame = connector.fetch("sec8k_index", [], start_date, end_date)
    manifest_rows = [_manifest_from_record(record) for record in manifest_frame.to_dict("records")]
    manifest_rows = [row for row in manifest_rows if row is not None]
    if limit is not None:
        manifest_rows = manifest_rows[: max(0, int(limit))]
    manifest_paths = write_sec8k_manifest(root=root, rows=manifest_rows)
    ticker_map_payload = _fetch_and_write_company_tickers(
        root=root,
        connector=connector,
        start_date=start_date,
        end_date=end_date,
    )
    ticker_by_cik = ticker_map_payload["ticker_by_cik"]

    filing_rows: list[dict[str, object]] = []
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
                _read_cached_complete_text(root=root, manifest=manifest)
                if use_cache
                else None
            )
            if cached is None:
                status, text, url, retry_events = _retrieve_with_retries(
                    connector=connector,
                    manifest=manifest,
                    max_attempts=max_retrieval_attempts,
                    rate_limit_pause_seconds=rate_limit_pause_seconds,
                    sleep_fn=sleep_fn,
                )
                complete_path, metadata_path = write_sec8k_raw_artifacts(
                    root=root,
                    manifest=manifest,
                    complete_text=text,
                    complete_txt_url=url,
                    complete_txt_http_status=status,
                )
            else:
                text, complete_path, metadata_path = cached
                status = 200
                url = None
                retry_events = []
            if retry_events:
                item["retry_events"] = retry_events
            parsed = parse_sec8k_complete_text(text)
            filing_rows.append(
                _filing_row(
                    manifest=manifest,
                    parsed=parsed,
                    ticker_by_cik=ticker_by_cik,
                    complete_path=complete_path,
                )
            )
            item["complete_txt_http_status"] = status
            item["complete_txt_url"] = url
            item["raw_artifacts"] = [str(complete_path), str(metadata_path)]
            item["item_numbers"] = list(parsed.item_numbers)
            item["exhibit_count"] = sum(1 for doc in parsed.documents if doc.is_exhibit)
            item["status"] = "PARSED"
        except Exception as exc:  # pragma: no cover - live operational guard
            errors.append(f"exception:{type(exc).__name__}:{exc}")
            item["status"] = "FAILED"
        item["errors"] = errors
        accessions.append(item)

    filing_artifacts = write_sec8k_filing_index(root=root, rows=filing_rows)
    candidates = build_sec8k_item_candidates(
        filings=pd.DataFrame(filing_rows),
        source_path=Path(filing_artifacts["reference_path"]),
    )
    candidate_payload = write_sec8k_item_candidates(root=root, candidates=candidates)
    failed = sum(1 for item in accessions if item["status"] != "PARSED")
    payload: dict[str, object] = {
        "version": SEC8K_INGEST_VERSION,
        "checked_at": checked_at,
        "verdict": "PASS" if failed == 0 else "PARTIAL",
        "data_root": str(root),
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "cache_enabled": use_cache,
        "manifest_count": len(manifest_rows),
        "parsed_count": len(filing_rows),
        "failed_count": failed,
        "manifest_artifacts": [str(path) for path in manifest_paths],
        "filing_artifacts": filing_artifacts,
        "company_tickers_artifact": ticker_map_payload["artifact"],
        "candidate_artifacts": candidate_payload,
        "accessions": accessions,
    }
    _write_ingest_artifact(root=root, payload=payload)
    return payload


def write_sec8k_manifest(*, root: Path, rows: list[Sec8KManifestRow]) -> list[Path]:
    """Write SEC-index 8-K manifest rows under the NAS-style raw layout."""
    if not rows:
        return []
    frame = pd.DataFrame([row.to_dict() for row in rows]).drop_duplicates(
        subset=["accession", "archive_cik", "index_filename"]
    )
    written: list[Path] = []
    manifest_root = Path(root) / "data" / "raw" / "sec" / "sec8k_manifest"
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
    curated = Path(root) / "data" / "curated" / "sec" / "sec8k" / "manifest" / "data.parquet"
    curated.parent.mkdir(parents=True, exist_ok=True)
    if curated.exists():
        frame = pd.concat([pd.read_parquet(curated), frame], ignore_index=True)
    frame.drop_duplicates(subset=["accession", "archive_cik", "index_filename"]).sort_values(
        ["filed_date", "accession"]
    ).to_parquet(curated, index=False)
    written.append(curated)
    return written


def write_sec8k_raw_artifacts(
    *,
    root: Path,
    manifest: Sec8KManifestRow,
    complete_text: str,
    complete_txt_url: str | None,
    complete_txt_http_status: int | None,
) -> tuple[Path, Path]:
    """Write raw SEC 8-K complete text and retrieval metadata."""
    target = _raw_dir(root=root, manifest=manifest)
    target.mkdir(parents=True, exist_ok=True)
    complete = target / "complete.txt"
    complete.write_text(complete_text, encoding="utf-8")
    metadata = target / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "source_family": "sec8k",
                "manifest": manifest.to_dict(),
                "retrieval": {
                    "complete_txt_url": complete_txt_url,
                    "complete_txt_http_status": complete_txt_http_status,
                    "complete_txt_sha256": _sha256_text(complete_text),
                    "xml_source": "complete_txt",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return complete, metadata


def write_sec8k_filing_index(
    *, root: Path, rows: list[dict[str, object]]
) -> dict[str, str]:
    """Write NAS-visible SEC filing-index artifacts for 8-K candidates."""
    frame = pd.DataFrame(rows)
    reference = Path(root) / "data" / "reference" / "sec_filing_index.parquet"
    curated = Path(root) / "data" / "curated" / "sec" / "sec8k" / "filing_index" / "data.parquet"
    for path in (reference, curated):
        path.parent.mkdir(parents=True, exist_ok=True)
        output = frame
        if path.exists() and not frame.empty:
            output = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        if not output.empty:
            output = output.drop_duplicates(
                subset=["accessionNumber"], keep="last"
            ).sort_values(["filingDate", "accessionNumber"])
        output.to_parquet(path, index=False)
    return {"reference_path": str(reference), "curated_path": str(curated)}


def _retrieve_with_retries(
    *,
    connector: Sec8KIngestClient,
    manifest: Sec8KManifestRow,
    max_attempts: int,
    rate_limit_pause_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[int, str, str, list[dict[str, object]]]:
    retry_events: list[dict[str, object]] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            status, text, url = connector.retrieve_complete_submission_text(
                index_filename=manifest.index_filename,
                endpoint_key="sec8k_complete_txt",
            )
            return status, text, url, retry_events
        except BudgetBlockedConnectorError as exc:
            if attempt >= max_attempts:
                raise
            sleep_seconds = _budget_block_sleep_seconds(
                exc, default_seconds=rate_limit_pause_seconds
            )
            retry_events.append(
                {"attempt": attempt, "reason": "local_budget_block", "sleep_seconds": sleep_seconds}
            )
            sleep_fn(sleep_seconds)
        except RemoteRateLimitConnectorError as exc:
            if attempt >= max_attempts:
                raise
            sleep_seconds = float(exc.retry_after_seconds or rate_limit_pause_seconds)
            retry_events.append(
                {"attempt": attempt, "reason": "remote_rate_limit", "sleep_seconds": sleep_seconds}
            )
            sleep_fn(sleep_seconds)
    raise RuntimeError("unreachable SEC 8-K retrieval retry state")


def _fetch_and_write_company_tickers(
    *,
    root: Path,
    connector: Sec8KIngestClient,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    frame = connector.fetch("company_tickers", [], start_date, end_date)
    target = Path(root) / "data" / "reference" / "sec_company_tickers.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    ticker_by_cik: dict[str, str] = {}
    ticker_rank_by_cik: dict[str, tuple[int, str]] = {}
    for row in frame.to_dict("records"):
        cik = str(row.get("cik_str") or row.get("cik") or "").strip()
        ticker = str(row.get("ticker") or "").strip().upper()
        if not cik or not ticker:
            continue
        cik_key = str(int(cik)) if cik.isdigit() else cik
        rank = (_ticker_preference_rank(ticker), ticker)
        if cik_key not in ticker_rank_by_cik or rank < ticker_rank_by_cik[cik_key]:
            ticker_rank_by_cik[cik_key] = rank
            ticker_by_cik[cik_key] = ticker
    return {"artifact": str(target), "ticker_by_cik": ticker_by_cik}


def _filing_row(
    *,
    manifest: Sec8KManifestRow,
    parsed,
    ticker_by_cik: dict[str, str],
    complete_path: Path,
) -> dict[str, object]:
    items = [item for item in parsed.item_numbers if item in ITEM_FAMILIES]
    cik_key = str(int(manifest.archive_cik)) if str(manifest.archive_cik).isdigit() else manifest.archive_cik
    return {
        "form": "8-K",
        "accessionNumber": manifest.accession,
        "accession": manifest.accession,
        "accession_no_dashes": manifest.accession_no_dashes,
        "archive_cik": manifest.archive_cik,
        "cik": str(manifest.archive_cik).zfill(10),
        "ticker": ticker_by_cik.get(cik_key, ""),
        "filingDate": manifest.filed_date,
        "filed_date": manifest.filed_date,
        "acceptanceDateTime": parsed.accepted_at_raw,
        "accepted_at_raw": parsed.accepted_at_raw,
        "items": items,
        "sec_items": items,
        "exhibit_count": sum(1 for doc in parsed.documents if doc.is_exhibit),
        "document_count": len(parsed.documents),
        "source_hash": parsed.source_hash,
        "complete_txt_path": str(complete_path),
        "index_filename": manifest.index_filename,
        "index_year": manifest.index_year,
        "index_quarter": manifest.index_quarter,
        "parser_version": SEC8K_INGEST_VERSION,
    }


def _read_cached_complete_text(
    *, root: Path, manifest: Sec8KManifestRow
) -> tuple[str, Path, Path] | None:
    target = _raw_dir(root=root, manifest=manifest)
    complete = target / "complete.txt"
    metadata = target / "metadata.json"
    if not complete.exists() or not metadata.exists():
        return None
    try:
        return complete.read_text(encoding="utf-8"), complete, metadata
    except OSError:
        return None


def _ticker_preference_rank(ticker: str) -> int:
    """Prefer primary common-like symbols when SEC lists multiple securities per CIK."""
    normalized = ticker.strip().upper()
    rank = 0
    if "-P" in normalized or ".P" in normalized:
        rank += 100
    if any(fragment in normalized for fragment in ("-WS", ".WS", "-WT", ".WT")):
        rank += 100
    if any(separator in normalized for separator in ("-", ".")):
        rank += 10
    return rank


def _manifest_from_record(record: dict[str, object]) -> Sec8KManifestRow | None:
    accession = str(record.get("accession") or "").strip()
    archive_cik = str(record.get("archive_cik") or "").strip()
    index_filename = str(record.get("index_filename") or "").strip()
    if not accession or not archive_cik or not index_filename:
        return None
    return Sec8KManifestRow(
        archive_cik=archive_cik,
        form=str(record.get("form") or "8-K"),
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


def _raw_dir(*, root: Path, manifest: Sec8KManifestRow) -> Path:
    return (
        Path(root)
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / f"archive_cik={manifest.archive_cik}"
        / f"accession={manifest.accession_no_dashes}"
    )


def _write_ingest_artifact(*, root: Path, payload: dict[str, object]) -> None:
    target = Path(root) / "control" / "cluster" / "state" / "research" / "sec8k_ingest"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(payload["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
