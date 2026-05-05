"""Shared data-quality audit artifacts for collection and modeling sources."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from trademl.data_node.db import DataNodeDB
from trademl.modeling.factory import SOURCE_REQUIRED_COLUMNS


DATASET_PATH_DEFAULTS = {
    "equities_minute": ("data/raw/equities_minute",),
    "ticker_news": ("data/raw/ticker_news",),
    "stock_trades": ("data/raw/alpaca_market_events", "data/raw/stock_trades"),
    "stock_quotes": ("data/raw/alpaca_market_events", "data/raw/stock_quotes"),
    "sec_filings": ("data/reference/sec_filings.parquet", "data/reference/sec_filing_index.parquet"),
    "sec_companyfacts": ("data/reference/sec_companyfacts.parquet",),
    "fundamentals_daily": ("data/reference/fundamentals_daily.parquet",),
    "equities_ohlcv_adj": ("data/curated/equities_ohlcv_adj", "data/raw/equities_bars"),
    "macros_fred": ("data/raw/macros_fred",),
}

QUALITY_REQUIRED_COLUMNS = {
    **SOURCE_REQUIRED_COLUMNS,
    "stock_trades": ("symbol", "timestamp_or_vendor_ts", "price_or_px"),
    "stock_quotes": ("symbol", "timestamp_or_vendor_ts", "bid_or_bid_price", "ask_or_ask_price"),
    "macros_fred": ("series_id", "observation_date", "value"),
}

DUPLICATE_KEYS = {
    "equities_minute": ("source_name", "symbol", "vendor_ts"),
    "ticker_news": ("source_name", "symbol", "vendor_ts"),
    "stock_trades": ("source_name", "symbol", "vendor_ts"),
    "stock_quotes": ("source_name", "symbol", "vendor_ts"),
    "fundamentals_daily": ("symbol", "metric_date", "metric_name"),
    "equities_ohlcv_adj": ("date", "symbol"),
}


def run_data_quality_audit(
    *,
    data_root: Path,
    db: DataNodeDB | None = None,
    datasets: list[str] | tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run bounded data-quality checks and persist the latest shared artifact."""
    current = now or datetime.now(tz=UTC)
    selected = list(datasets or DATASET_PATH_DEFAULTS)
    source_availability = _read_json(
        data_root / "control" / "cluster" / "state" / "data" / "source_availability" / "latest.json"
    )
    rows = [
        _quality_row(
            data_root=data_root,
            dataset=dataset,
            source_availability=dict((source_availability.get("datasets") or {}).get(dataset) or {}),
        )
        for dataset in selected
    ]
    payload = {
        "version": "data_quality_v1",
        "generated_at": current.isoformat(),
        "rows": rows,
        "summary": {
            "ok": sum(1 for row in rows if row["verdict"] == "OK"),
            "warning": sum(1 for row in rows if row["verdict"] == "WARNING"),
            "critical": sum(1 for row in rows if row["verdict"] == "CRITICAL"),
            "info": sum(1 for row in rows if row["verdict"] == "INFO"),
        },
    }
    path = data_root / "control" / "cluster" / "state" / "data" / "quality" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, payload)
    payload["path"] = str(path)
    if db is not None:
        for row in rows:
            db.record_data_quality_check(
                dataset=str(row["dataset"]),
                check_name="source_quality",
                verdict=str(row["verdict"]),
                status=str(row["status"]),
                source_path=str(row.get("source_path") or ""),
                rows_checked=int(row.get("rows_checked") or 0),
                partitions_checked=int(row.get("partitions_checked") or 0),
                required_columns_missing=list(row.get("missing_required_columns") or []),
                duplicate_key_count=int(row.get("duplicate_key_count") or 0),
                null_rate_max=row.get("null_rate_max"),
                timestamp_violation_count=int(row.get("timestamp_violation_count") or 0),
                pit_violation_count=int(row.get("pit_violation_count") or 0),
                readable=bool(row.get("readable", True)),
                reason=str(row.get("reason") or ""),
                metadata=row,
                created_at=current.isoformat(),
            )
    return payload


def _quality_row(*, data_root: Path, dataset: str, source_availability: dict[str, Any]) -> dict[str, Any]:
    if bool(source_availability.get("known_unavailable")):
        return {
            "dataset": dataset,
            "verdict": "INFO",
            "status": "known_unavailable",
            "state": source_availability.get("state"),
            "reason": source_availability.get("reason") or "known unavailable source",
            "rows_checked": 0,
            "partitions_checked": 0,
            "readable": True,
            "missing_required_columns": [],
        }
    roots = _dataset_roots(data_root=data_root, dataset=dataset, source_availability=source_availability)
    files = []
    for root in roots:
        files.extend(_source_files(root))
    files = sorted(dict.fromkeys(files))[-5:]
    if not files:
        return {
            "dataset": dataset,
            "verdict": "WARNING",
            "status": "source_unavailable",
            "reason": "no readable parquet files found at expected source paths",
            "source_path": str(roots[0]) if roots else None,
            "rows_checked": 0,
            "partitions_checked": 0,
            "readable": False,
            "missing_required_columns": list(QUALITY_REQUIRED_COLUMNS.get(dataset, ())),
        }
    columns: set[str] = set()
    rows_total = 0
    unreadable: list[str] = []
    for path in files:
        try:
            metadata = pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001
            unreadable.append(f"{path}: {exc}")
            continue
        columns.update(str(column) for column in metadata.schema.names)
        rows_total += int(metadata.metadata.num_rows)
    missing = _missing_required_columns(dataset=dataset, columns=columns)
    latest_frame = _read_latest_sample(files)
    duplicate_count = _duplicate_count(dataset=dataset, frame=latest_frame)
    null_rate_max = _max_null_rate(dataset=dataset, frame=latest_frame)
    timestamp_violations = _timestamp_violations(latest_frame)
    pit_violations = _pit_violations(latest_frame)
    verdict = "OK"
    status = "ok"
    reason = ""
    if unreadable:
        verdict = "CRITICAL"
        status = "unreadable"
        reason = unreadable[0]
    elif rows_total <= 0:
        verdict = "WARNING"
        status = "zero_rows"
        reason = "source exists but recent files have zero rows"
    elif missing:
        verdict = "CRITICAL"
        status = "schema_mismatch"
        reason = f"missing required columns: {missing}"
    elif timestamp_violations or pit_violations:
        verdict = "CRITICAL"
        status = "timestamp_violation"
        reason = "timestamp/PIT sanity check failed"
    elif duplicate_count:
        verdict = "WARNING"
        status = "duplicate_keys"
        reason = "duplicate keys detected in latest sample"
    return {
        "dataset": dataset,
        "verdict": verdict,
        "status": status,
        "reason": reason,
        "source_path": str(files[-1]),
        "source_roots": [str(root) for root in roots],
        "rows_checked": rows_total,
        "partitions_checked": len(files),
        "columns": sorted(columns),
        "missing_required_columns": missing,
        "duplicate_key_count": duplicate_count,
        "null_rate_max": null_rate_max,
        "timestamp_violation_count": timestamp_violations,
        "pit_violation_count": pit_violations,
        "readable": not unreadable,
        "latest_partition": _partition_date_value(files[-1]),
    }


def _dataset_roots(*, data_root: Path, dataset: str, source_availability: dict[str, Any]) -> list[Path]:
    raw_paths = list(source_availability.get("existing_paths") or source_availability.get("paths") or [])
    if not raw_paths:
        raw_paths = list(DATASET_PATH_DEFAULTS.get(dataset, ()))
    roots = []
    candidate_roots = _candidate_data_roots(data_root)
    for raw in raw_paths:
        path = Path(str(raw))
        if path.is_absolute():
            roots.append(path)
        else:
            roots.extend(root / path for root in candidate_roots)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _candidate_data_roots(data_root: Path) -> list[Path]:
    """Return plausible NAS aliases for one mounted data root."""
    roots = [data_root]
    parent = data_root.parent
    if parent.exists():
        for sibling in sorted(parent.iterdir()):
            if sibling == data_root or not sibling.is_dir():
                continue
            if "nas" in sibling.name.lower():
                roots.append(sibling)
    return roots


def _source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files = sorted(root.glob("date=*/data.parquet"))
    if files:
        return files
    files = sorted(root.glob("series=*/data.parquet"))
    if files:
        return files
    return sorted(root.glob("*.parquet"))


def _read_latest_sample(files: list[Path]) -> pd.DataFrame:
    for path in reversed(files):
        with contextlib.suppress(Exception):
            return pd.read_parquet(path)
    return pd.DataFrame()


def _missing_required_columns(*, dataset: str, columns: set[str]) -> list[str]:
    if not columns:
        return list(QUALITY_REQUIRED_COLUMNS.get(dataset, ()))
    if dataset in {"equities_minute", "stock_trades", "stock_quotes"}:
        missing = sorted({"symbol"}.difference(columns))
        if dataset == "equities_minute":
            missing.extend(sorted({"open", "high", "low", "close"}.difference(columns)))
        if not ({"timestamp", "vendor_ts"} & columns):
            missing.append("timestamp_or_vendor_ts")
        if dataset == "stock_trades" and not ({"price", "px"} & columns):
            missing.append("price_or_px")
        if dataset == "stock_quotes":
            if not ({"bid_price", "bid"} & columns):
                missing.append("bid_or_bid_price")
            if not ({"ask_price", "ask"} & columns):
                missing.append("ask_or_ask_price")
        return missing
    if dataset == "ticker_news":
        missing = []
        if not ({"symbol", "symbols"} & columns):
            missing.append("symbol_or_symbols")
        if not ({"published_at", "vendor_ts"} & columns):
            missing.append("published_at_or_vendor_ts")
        return missing
    if dataset == "sec_filings":
        missing = []
        if "form" not in columns:
            missing.append("form")
        if not ({"symbol", "cik", "cik_str"} & columns):
            missing.append("symbol_or_cik")
        if not ({"accepted_at", "acceptanceDateTime", "filing_date", "filingDate"} & columns):
            missing.append("accepted_at_or_acceptanceDateTime_or_filingDate")
        return missing
    if dataset == "sec_companyfacts":
        missing = []
        if "cik" not in columns:
            missing.append("cik")
        if not ({"facts_path", "facts_relative_path"} & columns):
            missing.append("facts_path_or_relative_path")
        return missing
    return sorted(set(QUALITY_REQUIRED_COLUMNS.get(dataset, ())).difference(columns))


def _duplicate_count(*, dataset: str, frame: pd.DataFrame) -> int:
    keys = [key for key in DUPLICATE_KEYS.get(dataset, ()) if key in frame.columns]
    if not keys or frame.empty:
        return 0
    return int(frame.duplicated(keys, keep=False).sum())


def _max_null_rate(*, dataset: str, frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    concrete = [
        column
        for column in QUALITY_REQUIRED_COLUMNS.get(dataset, ())
        if "_or_" not in column and column in frame.columns
    ]
    if not concrete:
        return None
    return float(frame[concrete].isna().mean().max())


def _timestamp_violations(frame: pd.DataFrame) -> int:
    groups = (
        ("timestamp", "vendor_ts"),
        ("published_at", "vendor_ts"),
        ("accepted_at", "acceptanceDateTime"),
    )
    violations = 0
    for group in groups:
        columns = [column for column in group if column in frame.columns]
        if not columns:
            continue
        valid = pd.Series(False, index=frame.index)
        for column in columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            valid = valid | parsed.notna()
        violations += int((~valid).sum())
    return violations


def _pit_violations(frame: pd.DataFrame) -> int:
    if "feature_available_at" not in frame.columns or "date" not in frame.columns:
        return 0
    available = pd.to_datetime(frame["feature_available_at"], errors="coerce", utc=True)
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return int((available > dates + pd.Timedelta(days=1)).sum())


def _partition_date_value(path: Path) -> str | None:
    return path.parent.name.partition("=")[2] if path.parent.name.startswith("date=") else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with contextlib.suppress(Exception):
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
