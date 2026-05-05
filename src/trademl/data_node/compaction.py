"""Bounded raw/archive parquet compaction helpers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd


def compact_archive_partitions(
    *,
    data_root: Path,
    datasets: Iterable[str] | None = None,
    max_partitions: int = 10,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compact bounded raw/archive parquet partitions and write telemetry."""
    current = now or datetime.now(tz=UTC)
    raw_root = data_root / "data" / "raw"
    requested = sorted({str(dataset) for dataset in datasets or [] if str(dataset).strip()})
    dataset_roots = [raw_root / dataset for dataset in requested] if requested else sorted(path for path in raw_root.glob("*") if path.is_dir())
    rows: list[dict[str, Any]] = []
    remaining = max(1, int(max_partitions))
    for dataset_root in dataset_roots:
        if remaining <= 0:
            break
        if not dataset_root.exists():
            rows.append(
                {
                    "dataset": dataset_root.name,
                    "status": "missing",
                    "input_files": 0,
                    "output_files": 0,
                    "rows_in": 0,
                    "rows_out": 0,
                    "duplicates_dropped": 0,
                    "duration_ms": 0.0,
                    "error": "dataset raw root missing",
                }
            )
            continue
        for partition in sorted(dataset_root.glob("date=*")):
            if remaining <= 0:
                break
            files = sorted(path for path in partition.glob("*.parquet") if path.name != ".lock")
            extra_files = [path for path in files if path.name != "data.parquet"]
            if len(files) <= 1 and not extra_files:
                continue
            rows.append(_compact_partition(dataset=dataset_root.name, partition=partition, files=files, dry_run=dry_run))
            remaining -= 1
    payload = {
        "version": "archive_compaction_v1",
        "generated_at": current.isoformat(),
        "dry_run": bool(dry_run),
        "max_partitions": max(1, int(max_partitions)),
        "rows": rows,
        "summary": {
            "attempted_partitions": len(rows),
            "successes": sum(1 for row in rows if row.get("status") in {"success", "dry_run"}),
            "failures": sum(1 for row in rows if row.get("status") == "failed"),
            "rows_in": sum(int(row.get("rows_in") or 0) for row in rows),
            "rows_out": sum(int(row.get("rows_out") or 0) for row in rows),
            "duplicates_dropped": sum(int(row.get("duplicates_dropped") or 0) for row in rows),
        },
    }
    _write_compaction_telemetry(data_root=data_root, payload=payload)
    return payload


def _compact_partition(*, dataset: str, partition: Path, files: list[Path], dry_run: bool) -> dict[str, Any]:
    started = perf_counter()
    try:
        frames = [pd.read_parquet(path) for path in files]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows_in = int(len(combined))
        before = int(len(combined))
        if not combined.empty:
            combined = combined.drop_duplicates().reset_index(drop=True)
            combined = _deterministic_sort(combined)
        rows_out = int(len(combined))
        output = partition / "data.parquet"
        payload_hash = _payload_hash(combined)
        if not dry_run:
            tmp = output.with_name(f"{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
            combined.to_parquet(tmp, index=False)
            tmp.replace(output)
            for path in files:
                if path != output:
                    path.unlink(missing_ok=True)
        return {
            "dataset": dataset,
            "partition": partition.name,
            "partition_path": str(partition),
            "status": "dry_run" if dry_run else "success",
            "input_files": len(files),
            "output_files": 1 if not combined.empty or files else 0,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_dropped": max(0, before - rows_out),
            "schema_version": _schema_version(combined),
            "payload_hash": payload_hash,
            "duration_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "dataset": dataset,
            "partition": partition.name,
            "partition_path": str(partition),
            "status": "failed",
            "input_files": len(files),
            "output_files": 0,
            "rows_in": 0,
            "rows_out": 0,
            "duplicates_dropped": 0,
            "schema_version": None,
            "payload_hash": None,
            "duration_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": str(exc),
        }


def _deterministic_sort(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("date", "symbol", "vendor_ts", "timestamp", "source_name") if column in frame.columns]
    if sort_columns:
        try:
            return frame.sort_values(sort_columns).reset_index(drop=True)
        except TypeError:
            pass
    ordering = frame.astype(str).agg("|".join, axis=1)
    return frame.assign(_compact_sort_key=ordering).sort_values("_compact_sort_key").drop(columns=["_compact_sort_key"]).reset_index(drop=True)


def _schema_version(frame: pd.DataFrame) -> str:
    pieces = [f"{column}:{dtype}" for column, dtype in zip(frame.columns, frame.dtypes, strict=False)]
    return "compact_schema_" + hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()[:12]


def _payload_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha1(b"empty").hexdigest()
    normalized = frame.sort_index(axis=1).reset_index(drop=True).astype(str)
    values = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return hashlib.sha1(values).hexdigest()


def _write_compaction_telemetry(*, data_root: Path, payload: dict[str, Any]) -> Path:
    path = data_root / "control" / "cluster" / "state" / "data" / "compaction" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
