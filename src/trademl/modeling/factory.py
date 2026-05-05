"""Point-in-time modeling feature and label artifact factory."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trademl.features.equities import build_features
from trademl.labels.returns import build_labels

DEFAULT_FEATURE_SET = "daily_price_liquidity_v1"
DEFAULT_FEATURE_VERSION = "price_liquidity_v1"
DEFAULT_LABEL_DEFINITION = "universe_relative_forward_return"
DEFAULT_LABEL_VERSION = "universe_relative_forward_return_v1"
DEFAULT_LABEL_HORIZONS = (1, 5, 20)
FEATURE_SOURCE_CONTRACT_VERSION = "feature_source_contract_v1"
SOURCE_REQUIRED_COLUMNS = {
    "ticker_news": ("symbol_or_symbols", "published_at_or_vendor_ts"),
    "equities_minute": ("symbol", "timestamp_or_vendor_ts", "open", "high", "low", "close"),
    "sec_filings": ("form", "symbol_or_cik", "accepted_at_or_acceptanceDateTime_or_filingDate"),
    "fundamentals_tiingo": ("symbol",),
    "equities_ohlcv_adj": ("date", "symbol", "close"),
}
SOURCE_ROOT_DATASET_PATHS = {
    "sec_filings": ("data/reference/sec_filings.parquet", "data/reference/sec_filing_index.parquet"),
    "fundamentals_tiingo": ("data/reference/fundamentals_tiingo.parquet", "data/reference/fundamentals_daily.parquet"),
    "ticker_news": ("data/raw/ticker_news",),
    "equities_minute": ("data/raw/equities_minute",),
}
SOURCE_STATE_AVAILABLE = "AVAILABLE"
SOURCE_STATE_STALE = "STALE"
SOURCE_STATE_ZERO_COVERAGE = "ZERO_COVERAGE"
SOURCE_STATE_ENTITLEMENT_UNAVAILABLE = "ENTITLEMENT_UNAVAILABLE"
SOURCE_STATE_SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
SOURCE_STATE_DISABLED_BY_POLICY = "DISABLED_BY_POLICY"
SOURCE_STATE_SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
SOURCE_STATE_PRIORITY = {
    SOURCE_STATE_SCHEMA_MISMATCH: 6,
    SOURCE_STATE_SOURCE_UNAVAILABLE: 5,
    SOURCE_STATE_ZERO_COVERAGE: 4,
    SOURCE_STATE_STALE: 3,
    SOURCE_STATE_ENTITLEMENT_UNAVAILABLE: 2,
    SOURCE_STATE_DISABLED_BY_POLICY: 1,
    SOURCE_STATE_AVAILABLE: 0,
}
KNOWN_UNAVAILABLE_SOURCES = {
    "fundamentals_tiingo": {
        "state": SOURCE_STATE_ENTITLEMENT_UNAVAILABLE,
        "reason": "fundamentals_tiingo currently has zero usable rows under the free-plan/source entitlement",
    }
}


def build_modeling_artifacts(
    *,
    data_root: Path,
    feature_config: dict[str, Any],
    feature_set: str = DEFAULT_FEATURE_SET,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    label_definition: str = DEFAULT_LABEL_DEFINITION,
    label_version: str = DEFAULT_LABEL_VERSION,
    label_horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Build deterministic partitioned modeling feature and label artifacts."""
    panel = _load_curated_panel(data_root=data_root, report_date=report_date)
    feature_frame = _modeling_features(
        panel=panel,
        data_root=data_root,
        feature_config=feature_config,
        feature_set=feature_set,
        feature_version=feature_version,
    )
    label_frame = _modeling_labels(
        panel=panel,
        label_definition=label_definition,
        label_version=label_version,
        label_horizons=tuple(int(horizon) for horizon in label_horizons),
    )
    data_revision = _data_revision(data_root=data_root)
    feature_frame["data_revision"] = data_revision
    label_frame["data_revision"] = data_revision
    feature_root = _features_root(data_root=data_root, feature_version=feature_version)
    label_root = _labels_root(data_root=data_root, label_version=label_version)
    feature_partitions = _write_partitioned(frame=feature_frame, root=feature_root)
    label_partitions = _write_partitioned(frame=label_frame, root=label_root)
    feature_groups = _feature_groups_for(feature_set=feature_set, feature_version=feature_version)
    group_metadata = _feature_group_metadata(
        frame=feature_frame,
        data_root=data_root,
        feature_set=feature_set,
        feature_version=feature_version,
    )
    readiness = _feature_readiness(
        frame=feature_frame,
        group_metadata=group_metadata,
        feature_set=feature_set,
        feature_version=feature_version,
    )
    source_contract = _write_feature_source_contract(data_root=data_root)
    payload = {
        "feature_set": feature_set,
        "feature_version": feature_version,
        "label_definition": label_definition,
        "label_version": label_version,
        "label_horizons": sorted({int(horizon) for horizon in label_horizons}),
        "data_revision": data_revision,
        "built_at": datetime.now(tz=UTC).isoformat(),
        "feature_rows": int(len(feature_frame)),
        "label_rows": int(len(label_frame)),
        "feature_partitions": feature_partitions,
        "label_partitions": label_partitions,
        "feature_root": str(feature_root),
        "label_root": str(label_root),
        "feature_groups": feature_groups,
        "feature_group_metadata": group_metadata,
        "feature_readiness": readiness,
        "source_contract": source_contract,
    }
    _write_registry(data_root=data_root, payload=payload)
    return payload


def load_modeling_dataset(
    *,
    data_root: Path,
    feature_version: str,
    label_version: str,
    label_horizon: int,
    report_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a modeling-ready frame from versioned feature and label artifacts."""
    label_col = f"label_{int(label_horizon)}d"
    raw_col = f"raw_forward_return_{int(label_horizon)}d"
    features = _load_partitioned(root=_features_root(data_root=data_root, feature_version=feature_version), report_date=report_date)
    labels = _load_partitioned(root=_labels_root(data_root=data_root, label_version=label_version), report_date=report_date)
    required = {"date", "symbol", label_col, raw_col}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"label artifact {label_version!r} missing columns: {missing}")
    feature_required = {"date", "symbol", "feature_available_at", "feature_version", "data_revision"}
    feature_missing = sorted(feature_required.difference(features.columns))
    if feature_missing:
        raise ValueError(f"feature artifact {feature_version!r} missing columns: {feature_missing}")
    target_date_col = f"target_date_{int(label_horizon)}d"
    label_columns = ["date", "symbol", raw_col, label_col, "label_version", "label_definition", "data_revision"]
    if target_date_col in labels.columns:
        label_columns.append(target_date_col)
        if report_date:
            immature = pd.to_datetime(labels[target_date_col]) > pd.Timestamp(report_date)
            labels.loc[immature, [raw_col, label_col]] = pd.NA
    merged = features.merge(labels[label_columns], on=["date", "symbol"])
    if "data_revision_x" in merged.columns or "data_revision_y" in merged.columns:
        merged["data_revision"] = merged.get("data_revision_x", merged.get("data_revision_y"))
        merged = merged.drop(columns=[column for column in ("data_revision_x", "data_revision_y") if column in merged.columns])
    merged["date"] = pd.to_datetime(merged["date"])
    merged["feature_available_at"] = pd.to_datetime(merged["feature_available_at"], utc=True)
    if (merged["feature_available_at"].dt.tz_convert(None) > merged["date"]).any():
        raise ValueError("feature_available_at cannot be after the modeling date")
    metadata = {
        **modeling_artifact_metadata(data_root=data_root),
        "feature_version": feature_version,
        "label_version": label_version,
        "label_horizon": int(label_horizon),
        "label_col": label_col,
        "raw_label_col": raw_col,
        "label_target_date_col": target_date_col,
    }
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True), metadata


def feature_label_preflight(
    *,
    data_root: Path,
    feature_version: str,
    label_version: str,
    label_horizon: int,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Return whether modeling feature/label artifacts are usable."""
    try:
        frame, metadata = load_modeling_dataset(
            data_root=data_root,
            feature_version=feature_version,
            label_version=label_version,
            label_horizon=label_horizon,
            report_date=report_date,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": str(exc),
            "feature_version": feature_version,
            "label_version": label_version,
            "label_horizon": int(label_horizon),
        }
    readiness = _metadata_feature_readiness(metadata=metadata, feature_version=feature_version)
    if not readiness.get("ok", True):
        return {
            "ok": False,
            "reason": str(readiness.get("reason") or "feature readiness failed"),
            "feature_version": feature_version,
            "label_version": label_version,
            "label_horizon": int(label_horizon),
            "rows": int(len(frame)),
            "feature_readiness": readiness,
        }
    label_col = f"label_{int(label_horizon)}d"
    sample = frame.dropna(subset=[label_col]).head(1)
    if sample.empty:
        return {
            "ok": False,
            "reason": f"no mature labels for horizon {label_horizon}",
            "feature_version": feature_version,
            "label_version": label_version,
            "label_horizon": int(label_horizon),
            "rows": int(len(frame)),
        }
    return {
        "ok": True,
        "feature_version": feature_version,
        "label_version": label_version,
        "label_horizon": int(label_horizon),
        "rows": int(len(frame)),
        "sample_date": pd.Timestamp(sample["date"].iloc[0]).date().isoformat(),
        "data_revision": metadata.get("data_revision"),
        "feature_readiness": readiness,
    }


def modeling_artifact_metadata(*, data_root: Path) -> dict[str, Any]:
    """Return the latest modeling artifact registry payload when present."""
    path = _registry_path(data_root=data_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _modeling_features(
    *,
    panel: pd.DataFrame,
    data_root: Path,
    feature_config: dict[str, Any],
    feature_set: str,
    feature_version: str,
) -> pd.DataFrame:
    features = build_features(panel, feature_config)
    features["date"] = pd.to_datetime(features["date"])
    features = _attach_optional_feature_groups(
        features=features,
        panel=panel,
        data_root=data_root,
        feature_set=feature_set,
        feature_version=feature_version,
    )
    features["feature_available_at"] = features["date"].dt.tz_localize("UTC")
    features["feature_set"] = feature_set
    features["feature_version"] = feature_version
    features["universe_member"] = True
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)


def _feature_groups_for(*, feature_set: str, feature_version: str) -> list[str]:
    name = f"{feature_set} {feature_version}".lower()
    groups = ["price_liquidity"]
    if "multi_source" in name:
        return [*groups, "fundamentals_sec", "news_events", "minute_daily"]
    if "fundamental" in name or "sec" in name or "filing" in name:
        groups.append("fundamentals_sec")
    if "news" in name:
        groups.append("news_events")
    if "minute" in name or "intraday" in name:
        groups.append("minute_daily")
    return groups


def _feature_group_metadata(*, frame: pd.DataFrame, data_root: Path, feature_set: str, feature_version: str) -> dict[str, Any]:
    groups = _feature_groups_for(feature_set=feature_set, feature_version=feature_version)
    metadata: dict[str, Any] = {}
    for group in groups:
        columns = _feature_group_columns(group)
        present = [column for column in columns if column in frame.columns]
        if present:
            coverage = float(frame[present].notna().any(axis=1).mean())
        else:
            coverage = 0.0
        metadata[group] = {
            "columns": present,
            "row_coverage": coverage,
            "readiness_status": _feature_group_readiness_status(group=group, columns=present, row_coverage=coverage),
            "sources": _source_readiness_for_group(data_root=data_root, group=group),
            **_feature_group_policy(group),
        }
    return metadata


def _feature_group_columns(group: str) -> list[str]:
    if group == "fundamentals_sec":
        return [
            "fundamental_metric_count",
            "fundamental_numeric_mean",
            "fundamental_days_since_latest_metric",
            "sec_filings_30d",
            "sec_filings_90d",
            "sec_8k_30d",
            "sec_10q_90d",
            "sec_10k_90d",
            "sec_days_since_last_filing",
        ]
    if group == "news_events":
        return [
            "news_count_1d",
            "news_count_7d",
            "news_source_count_7d",
            "news_vendor_count_7d",
            "news_abnormal_volume_7d",
            "news_days_since_last",
            "news_novelty_proxy_7d",
        ]
    if group == "minute_daily":
        return [
            "minute_intraday_return",
            "minute_intraday_range",
            "minute_realized_vol",
            "minute_first_30m_return",
            "minute_last_30m_return",
            "minute_volume_sum",
            "minute_volume_ratio",
            "minute_close_imbalance_proxy",
        ]
    return []


def _feature_group_policy(group: str) -> dict[str, Any]:
    if group == "fundamentals_sec":
        return {
            "source_datasets": ["sec_filings", "fundamentals_tiingo"],
            "feature_available_at_policy": "SEC accepted_at or reference last_verified must be <= modeling date",
            "safety_delay": "none",
        }
    if group == "news_events":
        return {
            "source_datasets": ["ticker_news"],
            "feature_available_at_policy": "published/vendor timestamp plus one calendar day safety delay must be <= modeling date",
            "safety_delay": "1d",
        }
    if group == "minute_daily":
        return {
            "source_datasets": ["equities_minute"],
            "feature_available_at_policy": "prior-session minute aggregates are shifted to the next modeling date",
            "safety_delay": "next_trading_day",
        }
    return {
        "source_datasets": ["equities_ohlcv_adj"],
        "feature_available_at_policy": "daily adjusted OHLCV available by modeling date",
        "safety_delay": "none",
    }


def _attach_optional_feature_groups(
    *,
    features: pd.DataFrame,
    panel: pd.DataFrame,
    data_root: Path,
    feature_set: str,
    feature_version: str,
) -> pd.DataFrame:
    groups = set(_feature_groups_for(feature_set=feature_set, feature_version=feature_version))
    enriched = features.copy()
    if "fundamentals_sec" in groups:
        enriched = _merge_feature_group(enriched, _sec_filing_features(data_root=data_root, dates=enriched["date"]))
        enriched = _merge_feature_group(enriched, _fundamental_features(data_root=data_root, dates=enriched["date"]))
    if "news_events" in groups:
        enriched = _merge_feature_group(enriched, _news_event_features(data_root=data_root, dates=enriched["date"]))
    if "minute_daily" in groups:
        enriched = _merge_feature_group(enriched, _minute_daily_features(data_root=data_root, panel=panel))
    return enriched


def _merge_feature_group(features: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return features
    merged = features.merge(group, on=["date", "symbol"], how="left")
    added = [column for column in group.columns if column not in {"date", "symbol"}]
    for column in added:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def write_feature_source_contract(
    *,
    data_root: Path,
    source_root: Path | None = None,
    dataset_paths: dict[str, Iterable[Path | str]] | None = None,
) -> dict[str, Any]:
    """Write an explicit feature source contract for modeling source adapters."""
    existing = _read_configured_feature_source_paths(data_root=data_root)
    overrides: dict[str, list[Path]] = {dataset: list(paths) for dataset, paths in existing.items()}
    if source_root is not None:
        root = source_root.expanduser()
        for dataset, rel_paths in SOURCE_ROOT_DATASET_PATHS.items():
            overrides[dataset] = [root / rel_path for rel_path in rel_paths]
    for dataset, paths in dict(dataset_paths or {}).items():
        overrides[str(dataset)] = [
            path.expanduser() if isinstance(path, Path) else Path(str(path)).expanduser()
            for path in paths
        ]
    return _write_feature_source_contract(data_root=data_root, configured_paths=overrides)


def _read_configured_feature_source_paths(*, data_root: Path) -> dict[str, list[Path]]:
    contract_path = _source_contract_path(data_root=data_root)
    configured: dict[str, list[Path]] = {}
    if contract_path.exists():
        with contextlib.suppress(Exception):
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            for dataset, paths in dict(payload.get("datasets") or {}).items():
                raw_paths = paths.get("paths") if isinstance(paths, dict) else paths
                if isinstance(raw_paths, list):
                    configured[str(dataset)] = [Path(str(path)) if Path(str(path)).is_absolute() else data_root / str(path) for path in raw_paths]
    return configured


def _feature_source_contract(data_root: Path, configured_paths: dict[str, list[Path]] | None = None) -> dict[str, list[Path]]:
    configured = configured_paths if configured_paths is not None else _read_configured_feature_source_paths(data_root=data_root)
    defaults = {
        "sec_filings": [
            data_root / "data" / "reference" / "sec_filings.parquet",
            data_root / "data" / "reference" / "sec_filing_index.parquet",
        ],
        "fundamentals_tiingo": [
            data_root / "data" / "reference" / "fundamentals_tiingo.parquet",
            data_root / "data" / "reference" / "fundamentals_daily.parquet",
        ],
        "ticker_news": [data_root / "data" / "raw" / "ticker_news"],
        "equities_minute": [data_root / "data" / "raw" / "equities_minute"],
        "equities_ohlcv_adj": [data_root / "data" / "curated" / "equities_ohlcv_adj"],
    }
    merged: dict[str, list[Path]] = {}
    for dataset, paths in defaults.items():
        seen: set[str] = set()
        merged[dataset] = []
        for path in [*configured.get(dataset, []), *paths]:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            merged[dataset].append(path)
    return merged


def _configured_feature_source_paths(
    data_root: Path,
    configured_paths: dict[str, list[Path]] | None = None,
) -> dict[str, set[str]]:
    if configured_paths is not None:
        return {dataset: {str(path) for path in paths} for dataset, paths in configured_paths.items()}
    contract_path = _source_contract_path(data_root=data_root)
    configured: dict[str, set[str]] = {}
    if not contract_path.exists():
        return configured
    with contextlib.suppress(Exception):
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        for dataset, paths in dict(payload.get("datasets") or {}).items():
            raw_paths = paths.get("paths") if isinstance(paths, dict) else paths
            if not isinstance(raw_paths, list):
                continue
            configured[str(dataset)] = {
                str(Path(str(path)) if Path(str(path)).is_absolute() else data_root / str(path))
                for path in raw_paths
            }
    return configured


def _load_feature_source_frame(
    *,
    data_root: Path,
    dataset: str,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sources = []
    frames = []
    requested_columns = [str(column) for column in columns or []]
    for root in _feature_source_contract(data_root).get(dataset, []):
        files = _source_files(root, start_date=start_date, end_date=end_date)
        rows = 0
        columns: list[str] = []
        latest_partition: str | None = None
        for path in files:
            try:
                read_columns = _readable_parquet_columns(path=path, requested=requested_columns)
                frame = pd.read_parquet(path, columns=read_columns or None)
            except Exception:  # noqa: BLE001
                continue
            if frame.empty:
                rows += 0
            else:
                frames.append(frame)
                rows += int(len(frame))
                columns = sorted(set(columns).union(str(column) for column in frame.columns))
            partition = _partition_date_value(path)
            latest_partition = max([value for value in (latest_partition, partition) if value], default=latest_partition)
        sources.append(
            {
                "dataset": dataset,
                "path": str(root),
                "exists": root.exists(),
                "file_count": len(files),
                "rows": rows,
                "columns": columns,
                "latest_partition": latest_partition,
            }
        )
    if not frames:
        return pd.DataFrame(), sources
    return pd.concat(frames, ignore_index=True), sources


def _source_files(
    root: Path,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d") if start_date is not None else None
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d") if end_date is not None else None
    files = sorted(root.glob("date=*/data.parquet"))
    if start or end:
        files = [
            path
            for path in files
            if (
                (partition := _partition_date_value(path)) is not None
                and (start is None or partition >= start)
                and (end is None or partition <= end)
            )
        ]
    if files:
        return files
    return sorted(root.glob("*.parquet"))


def _partition_date_value(path: Path) -> str | None:
    return path.parent.name.partition("=")[2] if path.parent.name.startswith("date=") else None


def _readable_parquet_columns(*, path: Path, requested: list[str]) -> list[str]:
    if not requested:
        return []
    try:
        names = set(pq.ParquetFile(path).schema.names)
    except Exception:  # noqa: BLE001
        return []
    return [column for column in requested if column in names]


def _source_readiness_for_group(*, data_root: Path, group: str) -> dict[str, Any]:
    return {
        dataset: _source_summary(data_root=data_root, dataset=dataset)
        for dataset in _feature_group_policy(group)["source_datasets"]
    }


def _source_summary(
    *,
    data_root: Path,
    dataset: str,
    configured_paths: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    sources = []
    configured = _configured_feature_source_paths(data_root, configured_paths=configured_paths).get(dataset, set())
    for root in _feature_source_contract(data_root, configured_paths=configured_paths).get(dataset, []):
        files = _source_files(root)
        rows = 0
        columns: set[str] = set()
        latest_partition: str | None = None
        for path in files:
            file_summary = _parquet_file_summary(path)
            rows += int(file_summary.get("rows") or 0)
            columns.update(str(column) for column in list(file_summary.get("columns") or []))
            partition = path.parent.name.partition("=")[2] if path.parent.name.startswith("date=") else None
            latest_partition = max([value for value in (latest_partition, partition) if value], default=latest_partition)
        missing_required = _missing_required_source_columns(dataset=dataset, columns=columns)
        status = "missing"
        if root.exists() and rows <= 0:
            status = "empty"
        elif rows > 0 and missing_required:
            status = "invalid_schema"
        elif rows > 0:
            status = "available"
        lifecycle = _source_lifecycle_state(
            dataset=dataset,
            status=status,
            rows=rows,
            missing_required=missing_required,
            exists=root.exists(),
        )
        sources.append(
            {
                "dataset": dataset,
                "path": str(root),
                "exists": root.exists(),
                "source_status": "contract" if str(root) in configured else "default_unverified",
                "file_count": len(files),
                "rows": rows,
                "columns": sorted(columns),
                "required_columns": list(SOURCE_REQUIRED_COLUMNS.get(dataset, ())),
                "missing_required_columns": missing_required,
                "status": status,
                "source_state": lifecycle["state"],
                "lifecycle_state": lifecycle["state"],
                "actionable": lifecycle["actionable"],
                "known_unavailable": lifecycle["known_unavailable"],
                "reason": lifecycle["reason"],
                "latest_partition": latest_partition,
            }
        )
    rows = sum(int(source.get("rows") or 0) for source in sources)
    existing = [source for source in sources if source.get("exists")]
    latest_values = [str(source.get("latest_partition")) for source in sources if source.get("latest_partition")]
    available = [source for source in sources if source.get("status") == "available"]
    invalid = [source for source in sources if source.get("status") == "invalid_schema"]
    source_status = "available" if available else "invalid_schema" if invalid else "empty" if existing else "missing"
    source_state = _aggregate_source_state(sources)
    reasons = [str(source.get("reason")) for source in sources if source.get("reason")]
    return {
        "dataset": dataset,
        "status": source_status,
        "source_state": source_state,
        "lifecycle_state": source_state,
        "actionable": False if source_state in {SOURCE_STATE_ENTITLEMENT_UNAVAILABLE, SOURCE_STATE_DISABLED_BY_POLICY} else any(bool(source.get("actionable")) for source in sources),
        "known_unavailable": any(bool(source.get("known_unavailable")) for source in sources),
        "reason": "; ".join(dict.fromkeys(reasons)),
        "rows": rows,
        "paths": [str(source.get("path")) for source in sources],
        "existing_paths": [str(source.get("path")) for source in existing],
        "columns": sorted({column for source in sources for column in list(source.get("columns") or [])}),
        "required_columns": list(SOURCE_REQUIRED_COLUMNS.get(dataset, ())),
        "missing_required_columns": sorted({column for source in sources for column in list(source.get("missing_required_columns") or [])}) if not available else [],
        "latest_partition": max(latest_values) if latest_values else None,
        "sources": sources,
    }


def _source_lifecycle_state(
    *,
    dataset: str,
    status: str,
    rows: int,
    missing_required: list[str],
    exists: bool,
) -> dict[str, Any]:
    """Map a source adapter status to the auditable lifecycle state contract."""
    known = dict(KNOWN_UNAVAILABLE_SOURCES.get(dataset) or {})
    if status == "available":
        return {
            "state": SOURCE_STATE_AVAILABLE,
            "actionable": False,
            "known_unavailable": False,
            "reason": "",
        }
    if not exists:
        return {
            "state": SOURCE_STATE_SOURCE_UNAVAILABLE,
            "actionable": True,
            "known_unavailable": False,
            "reason": "source path is missing",
        }
    if status == "invalid_schema" or missing_required:
        return {
            "state": SOURCE_STATE_SCHEMA_MISMATCH,
            "actionable": True,
            "known_unavailable": False,
            "reason": f"missing required columns: {missing_required}",
        }
    if known and rows <= 0:
        return {
            "state": str(known.get("state") or SOURCE_STATE_ENTITLEMENT_UNAVAILABLE),
            "actionable": False,
            "known_unavailable": True,
            "reason": str(known.get("reason") or "source is unavailable by current entitlement/policy"),
        }
    return {
        "state": SOURCE_STATE_ZERO_COVERAGE,
        "actionable": True,
        "known_unavailable": False,
        "reason": "source exists but has zero usable rows",
    }


def _aggregate_source_state(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return SOURCE_STATE_SOURCE_UNAVAILABLE
    if any(str(source.get("source_state")) == SOURCE_STATE_AVAILABLE for source in sources):
        return SOURCE_STATE_AVAILABLE
    if any(str(source.get("source_state")) == SOURCE_STATE_SCHEMA_MISMATCH for source in sources):
        return SOURCE_STATE_SCHEMA_MISMATCH
    if any(bool(source.get("known_unavailable")) for source in sources):
        states = [str(source.get("source_state")) for source in sources if bool(source.get("known_unavailable"))]
        return states[0] if states else SOURCE_STATE_ENTITLEMENT_UNAVAILABLE
    return max(
        (str(source.get("source_state") or SOURCE_STATE_SOURCE_UNAVAILABLE) for source in sources),
        key=lambda state: SOURCE_STATE_PRIORITY.get(state, 0),
    )


def _missing_required_source_columns(*, dataset: str, columns: set[str]) -> list[str]:
    if not columns:
        return list(SOURCE_REQUIRED_COLUMNS.get(dataset, ()))
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
    if dataset == "equities_minute":
        missing = sorted({"symbol", "open", "high", "low", "close"}.difference(columns))
        if not ({"timestamp", "vendor_ts"} & columns):
            missing.append("timestamp_or_vendor_ts")
        return missing
    return sorted(set(SOURCE_REQUIRED_COLUMNS.get(dataset, ())).difference(columns))


def _parquet_file_summary(path: Path) -> dict[str, Any]:
    try:
        metadata = pq.ParquetFile(path)
    except Exception:  # noqa: BLE001
        return {"rows": 0, "columns": []}
    return {
        "rows": int(metadata.metadata.num_rows),
        "columns": list(metadata.schema.names),
    }


def _feature_group_readiness_status(*, group: str, columns: list[str], row_coverage: float) -> str:
    if group == "price_liquidity":
        return "READY"
    if not columns:
        return "BLOCKED"
    if row_coverage <= 0.0:
        return "BLOCKED"
    return "READY"


def _feature_readiness(
    *,
    frame: pd.DataFrame,
    group_metadata: dict[str, Any],
    feature_set: str,
    feature_version: str,
) -> dict[str, Any]:
    groups = _feature_groups_for(feature_set=feature_set, feature_version=feature_version)
    optional_groups = [group for group in groups if group != "price_liquidity"]
    blockers: list[str] = []
    for group in optional_groups:
        metadata = dict(group_metadata.get(group) or {})
        if metadata.get("readiness_status") != "READY":
            blockers.append(f"{group} has no usable source-backed feature coverage")
    missing_cols = sorted({"date", "symbol", "feature_available_at", "feature_version", "data_revision"}.difference(frame.columns))
    if missing_cols:
        blockers.append(f"feature artifact missing columns: {missing_cols}")
    if "feature_available_at" in frame.columns and "date" in frame.columns:
        available = pd.to_datetime(frame["feature_available_at"], utc=True, errors="coerce")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if available.isna().any():
            blockers.append("feature_available_at contains unparseable values")
        elif (available.dt.tz_convert(None) > dates).any():
            blockers.append("feature_available_at cannot be after the modeling date")
    return {
        "ok": not blockers,
        "status": "READY" if not blockers else "BLOCKED",
        "reason": "; ".join(blockers),
        "required_groups": optional_groups,
        "feature_groups": groups,
        "group_status": {group: dict(group_metadata.get(group) or {}).get("readiness_status") for group in groups},
    }


def _metadata_feature_readiness(*, metadata: dict[str, Any], feature_version: str) -> dict[str, Any]:
    readiness = dict(metadata.get("feature_readiness") or {})
    if readiness and str(metadata.get("feature_version") or "") == feature_version:
        return readiness
    return {"ok": True, "status": "UNKNOWN", "reason": "feature readiness metadata is unavailable"}


def _write_feature_source_contract(*, data_root: Path, configured_paths: dict[str, list[Path]] | None = None) -> dict[str, Any]:
    datasets = {
        dataset: {
            "paths": [str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path) for path in paths],
            "summary": _source_summary(data_root=data_root, dataset=dataset, configured_paths=configured_paths),
        }
        for dataset, paths in _feature_source_contract(data_root, configured_paths=configured_paths).items()
    }
    payload = {
        "version": FEATURE_SOURCE_CONTRACT_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "datasets": datasets,
    }
    path = _source_contract_path(data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_source_availability(data_root=data_root, source_contract=payload)
    return payload


def _write_source_availability(*, data_root: Path, source_contract: dict[str, Any]) -> dict[str, Any]:
    datasets = {}
    state_counts: dict[str, int] = {}
    for dataset, entry in dict(source_contract.get("datasets") or {}).items():
        summary = dict(entry.get("summary") or {})
        state = str(summary.get("source_state") or summary.get("lifecycle_state") or SOURCE_STATE_SOURCE_UNAVAILABLE)
        state_counts[state] = state_counts.get(state, 0) + 1
        datasets[str(dataset)] = {
            "dataset": str(dataset),
            "state": state,
            "status": summary.get("status"),
            "actionable": bool(summary.get("actionable")),
            "known_unavailable": bool(summary.get("known_unavailable")),
            "rows": int(summary.get("rows") or 0),
            "latest_partition": summary.get("latest_partition"),
            "reason": summary.get("reason"),
            "paths": list(summary.get("paths") or []),
            "existing_paths": list(summary.get("existing_paths") or []),
            "missing_required_columns": list(summary.get("missing_required_columns") or []),
            "sources": list(summary.get("sources") or []),
        }
    payload = {
        "version": "source_availability_v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "state_counts": state_counts,
        "datasets": datasets,
    }
    path = _source_availability_path(data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _normalize_fundamental_source(frame: pd.DataFrame) -> pd.DataFrame:
    if {"symbol", "metric_date", "metric_name", "metric_value"}.issubset(frame.columns):
        return frame.copy()
    if not {"symbol", "date", "data"}.issubset(frame.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        raw_payload = record.get("data")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        if isinstance(raw_payload, str):
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(raw_payload)
                payload = decoded if isinstance(decoded, dict) else {}
        for key, value in payload.items():
            rows.append(
                {
                    "symbol": str(record.get("symbol")),
                    "metric_date": record.get("date"),
                    "metric_name": str(key),
                    "metric_value": value,
                    "last_verified": record.get("as_of_date") or record.get("date"),
                }
            )
    return pd.DataFrame(rows)


def _normalize_sec_filing_source(*, data_root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "symbol" not in working.columns:
        cik_col = "cik" if "cik" in working.columns else "cik_str" if "cik_str" in working.columns else None
        if cik_col is None:
            return pd.DataFrame()
        mapping = _cik_symbol_mapping(data_root=data_root)
        if mapping.empty:
            return pd.DataFrame()
        working["_cik_key"] = pd.to_numeric(working[cik_col], errors="coerce").astype("Int64").astype(str)
        working = working.merge(mapping, on="_cik_key", how="left")
    if "form" not in working.columns or "symbol" not in working.columns:
        return pd.DataFrame()
    accepted_col = "accepted_at" if "accepted_at" in working.columns else "acceptanceDateTime" if "acceptanceDateTime" in working.columns else "filing_date" if "filing_date" in working.columns else "filingDate" if "filingDate" in working.columns else None
    if accepted_col is None:
        return pd.DataFrame()
    working["available_at"] = pd.to_datetime(working[accepted_col], errors="coerce", utc=True).dt.tz_convert(None)
    working["symbol"] = working["symbol"].astype(str).str.upper()
    return working.dropna(subset=["symbol", "available_at"])


def _cik_symbol_mapping(*, data_root: Path) -> pd.DataFrame:
    frames = []
    for path in [
        data_root / "data" / "reference" / "sec_company_tickers.parquet",
        data_root / "data" / "reference" / "universe.parquet",
    ]:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        cik_col = "cik_str" if "cik_str" in frame.columns else "cik" if "cik" in frame.columns else None
        symbol_col = "ticker" if "ticker" in frame.columns else "symbol" if "symbol" in frame.columns else None
        if cik_col is None or symbol_col is None:
            continue
        mapping = frame[[cik_col, symbol_col]].copy()
        mapping["_cik_key"] = pd.to_numeric(mapping[cik_col], errors="coerce").astype("Int64").astype(str)
        mapping["symbol"] = mapping[symbol_col].astype(str).str.upper()
        frames.append(mapping[["_cik_key", "symbol"]])
    if not frames:
        return pd.DataFrame(columns=["_cik_key", "symbol"])
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates("_cik_key")


def _fundamental_features(*, data_root: Path, dates: pd.Series) -> pd.DataFrame:
    frame, _sources = _load_feature_source_frame(data_root=data_root, dataset="fundamentals_tiingo")
    if frame.empty:
        return pd.DataFrame()
    frame = _normalize_fundamental_source(frame)
    if frame.empty:
        return pd.DataFrame()
    modeling_dates = pd.DataFrame({"date": sorted(pd.to_datetime(dates).dropna().unique())})
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["metric_date"] = pd.to_datetime(frame["metric_date"], errors="coerce")
    available = pd.to_datetime(frame.get("last_verified", frame["metric_date"]), errors="coerce")
    frame["available_at"] = available.fillna(frame["metric_date"])
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date_value in modeling_dates["date"]:
        eligible = frame[(frame["available_at"] <= date_value) & (frame["metric_date"] <= date_value)]
        if eligible.empty:
            continue
        latest = eligible.sort_values(["symbol", "metric_name", "available_at"]).groupby(["symbol", "metric_name"], as_index=False).tail(1)
        grouped = latest.groupby("symbol")
        for symbol, group in grouped:
            numeric = group["metric_value"].dropna()
            age = (date_value - group["metric_date"].max()).days if group["metric_date"].notna().any() else 0
            rows.append(
                {
                    "date": date_value,
                    "symbol": str(symbol),
                    "fundamental_metric_count": float(len(group)),
                    "fundamental_numeric_mean": float(numeric.mean()) if not numeric.empty else 0.0,
                    "fundamental_days_since_latest_metric": float(max(age, 0)),
                }
            )
    return pd.DataFrame(rows)


def _sec_filing_features(*, data_root: Path, dates: pd.Series) -> pd.DataFrame:
    frame, _sources = _load_feature_source_frame(data_root=data_root, dataset="sec_filings")
    if frame.empty:
        return pd.DataFrame()
    frame = _normalize_sec_filing_source(data_root=data_root, frame=frame)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce")
    frame = frame.dropna(subset=["available_at"])
    modeling_dates = np.array(sorted(pd.to_datetime(dates).dropna().unique()), dtype="datetime64[ns]")
    if len(modeling_dates) == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.sort_values("available_at").groupby("symbol", sort=True):
        times = np.array(pd.to_datetime(group["available_at"]).to_numpy(dtype="datetime64[ns]"))
        if len(times) == 0:
            continue
        forms = group["form"].astype(str).str.upper().to_numpy()
        form_8k = forms == "8-K"
        form_10q = forms == "10-Q"
        form_10k = forms == "10-K"
        for current in modeling_dates:
            end = int(np.searchsorted(times, current, side="right"))
            if end <= 0:
                continue
            start_30 = int(np.searchsorted(times, current - np.timedelta64(30, "D"), side="left"))
            start_90 = int(np.searchsorted(times, current - np.timedelta64(90, "D"), side="left"))
            latest = pd.Timestamp(times[end - 1])
            rows.append(
                {
                    "date": pd.Timestamp(current),
                    "symbol": str(symbol),
                    "sec_filings_30d": float(end - start_30),
                    "sec_filings_90d": float(end - start_90),
                    "sec_8k_30d": float(form_8k[start_30:end].sum()),
                    "sec_10q_90d": float(form_10q[start_90:end].sum()),
                    "sec_10k_90d": float(form_10k[start_90:end].sum()),
                    "sec_days_since_last_filing": float(max((pd.Timestamp(current) - latest).days, 0)),
                }
            )
    return pd.DataFrame(rows)


def _news_event_features(*, data_root: Path, dates: pd.Series) -> pd.DataFrame:
    modeling_dates = sorted(pd.to_datetime(dates).dropna().unique())
    if not modeling_dates:
        return pd.DataFrame()
    frame, _sources = _load_feature_source_frame(
        data_root=data_root,
        dataset="ticker_news",
        end_date=pd.Timestamp(modeling_dates[-1]),
        columns=["symbol", "symbols", "published_at", "vendor_ts", "source", "vendor", "source_name", "headline", "title"],
    )
    if frame.empty:
        return pd.DataFrame()
    ts_col = "published_at" if "published_at" in frame.columns else "vendor_ts" if "vendor_ts" in frame.columns else None
    if ts_col is None:
        return pd.DataFrame()
    frame = frame.copy()
    frame["available_at"] = pd.to_datetime(frame[ts_col], errors="coerce", utc=True).dt.tz_convert(None) + pd.Timedelta(days=1)
    if "symbol" not in frame.columns and "symbols" in frame.columns:
        frame = frame.explode("symbols").rename(columns={"symbols": "symbol"})
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.dropna(subset=["available_at"])
    frame = frame.sort_values(["symbol", "available_at"]).reset_index(drop=True)
    source_col = "source" if "source" in frame.columns else "vendor" if "vendor" in frame.columns else None
    vendor_col = "vendor" if "vendor" in frame.columns else "source_name" if "source_name" in frame.columns else source_col
    headline_col = "headline" if "headline" in frame.columns else "title" if "title" in frame.columns else None
    rows: list[dict[str, Any]] = []
    date_values = np.array([np.datetime64(pd.Timestamp(date_value), "ns") for date_value in modeling_dates])
    one_day = np.timedelta64(1, "D")
    seven_days = np.timedelta64(7, "D")
    thirty_seven_days = np.timedelta64(37, "D")
    for symbol, group in frame.groupby("symbol", sort=True):
        times = np.array(pd.to_datetime(group["available_at"]).to_numpy(dtype="datetime64[ns]"))
        sources = group[source_col].astype(str).to_numpy() if source_col else None
        vendors = group[vendor_col].astype(str).to_numpy() if vendor_col else None
        headlines = group[headline_col].astype(str).to_numpy() if headline_col else None
        for current_np in date_values:
            end = int(np.searchsorted(times, current_np, side="right"))
            if end <= 0:
                continue
            start_1d = int(np.searchsorted(times, current_np - one_day, side="left"))
            start_7d = int(np.searchsorted(times, current_np - seven_days, side="left"))
            start_37d = int(np.searchsorted(times, current_np - thirty_seven_days, side="left"))
            recent_1d_count = end - start_1d
            recent_7d_count = end - start_7d
            prior_30d_count = max(start_7d - start_37d, 0)
            source_count = len(set(sources[start_7d:end])) if sources is not None and recent_7d_count > 0 else 0
            vendor_count = len(set(vendors[start_7d:end])) if vendors is not None and recent_7d_count > 0 else 0
            baseline = max(float(prior_30d_count) / 30.0 * 7.0, 1.0)
            novelty = (
                float(len(set(headlines[start_7d:end]))) / max(1.0, float(recent_7d_count))
                if headlines is not None and recent_7d_count > 0
                else 0.0
            )
            current = pd.Timestamp(current_np)
            last_age = (current - pd.Timestamp(times[end - 1])).days
            rows.append(
                {
                    "date": current,
                    "symbol": str(symbol),
                    "news_count_1d": float(recent_1d_count),
                    "news_count_7d": float(recent_7d_count),
                    "news_source_count_7d": float(source_count),
                    "news_vendor_count_7d": float(vendor_count),
                    "news_abnormal_volume_7d": float(recent_7d_count) / baseline,
                    "news_novelty_proxy_7d": novelty,
                    "news_days_since_last": float(max(last_age, 0)),
                }
            )
    return pd.DataFrame(rows)


def _minute_daily_features(*, data_root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    calendar = sorted(pd.to_datetime(panel["date"]).dropna().unique())
    if not calendar:
        return pd.DataFrame()
    frame, _sources = _load_feature_source_frame(
        data_root=data_root,
        dataset="equities_minute",
        start_date=pd.Timestamp(calendar[0]) - pd.Timedelta(days=7),
        end_date=pd.Timestamp(calendar[-1]),
        columns=["symbol", "timestamp", "vendor_ts", "open", "high", "low", "close", "volume"],
    )
    if frame.empty:
        return pd.DataFrame()
    timestamp_col = "timestamp" if "timestamp" in frame.columns else "vendor_ts" if "vendor_ts" in frame.columns else None
    required = {"symbol", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    if timestamp_col is None:
        return pd.DataFrame()
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"])
    frame["source_date"] = frame["timestamp"].dt.tz_convert(None).dt.normalize()
    rows: list[dict[str, Any]] = []
    for (source_date, symbol), group in frame.sort_values("timestamp").groupby(["source_date", "symbol"]):
        close = pd.to_numeric(group["close"], errors="coerce")
        high = pd.to_numeric(group["high"], errors="coerce")
        low = pd.to_numeric(group["low"], errors="coerce")
        open_ = pd.to_numeric(group["open"], errors="coerce")
        volume = pd.to_numeric(group.get("volume", 0.0), errors="coerce").fillna(0.0)
        if close.dropna().empty or open_.dropna().empty:
            continue
        returns = close.pct_change().dropna()
        rows.append(
            {
                "source_date": pd.Timestamp(source_date),
                "symbol": str(symbol),
                "minute_intraday_return": float(close.iloc[-1] / open_.iloc[0] - 1.0),
                "minute_intraday_range": float(high.max() / max(low.min(), 1e-12) - 1.0),
                "minute_realized_vol": float(returns.std(ddof=0) if not returns.empty else 0.0),
                "minute_first_30m_return": float(close.iloc[min(len(close) - 1, 29)] / open_.iloc[0] - 1.0),
                "minute_last_30m_return": float(close.iloc[-1] / close.iloc[max(0, len(close) - 30)] - 1.0),
                "minute_volume_sum": float(volume.sum()),
            }
        )
    minute = pd.DataFrame(rows)
    if minute.empty:
        return minute
    minute = minute.sort_values(["symbol", "source_date"]).reset_index(drop=True)
    minute["minute_volume_ratio"] = (
        minute["minute_volume_sum"]
        / minute.groupby("symbol")["minute_volume_sum"].transform(lambda series: series.rolling(20, min_periods=1).mean()).replace(0.0, pd.NA)
    ).fillna(0.0)
    minute["minute_close_imbalance_proxy"] = minute["minute_last_30m_return"] * minute["minute_volume_ratio"]
    next_dates = {pd.Timestamp(calendar[idx - 1]): pd.Timestamp(calendar[idx]) for idx in range(1, len(calendar))}
    minute["date"] = minute["source_date"].map(next_dates)
    minute = minute.dropna(subset=["date"]).drop(columns=["source_date"])
    return minute


def _modeling_labels(
    *,
    panel: pd.DataFrame,
    label_definition: str,
    label_version: str,
    label_horizons: tuple[int, ...],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for horizon in label_horizons:
        labels = build_labels(panel, horizon=int(horizon), include_target_date=True)
        labels["date"] = pd.to_datetime(labels["date"])
        labels = labels.rename(columns={"target_date": f"target_date_{int(horizon)}d"})
        if merged is None:
            merged = labels
        else:
            merged = merged.merge(labels, on=["date", "symbol"], how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["date", "symbol"])
    merged["label_definition"] = label_definition
    merged["label_version"] = label_version
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_curated_panel(*, data_root: Path, report_date: str | None = None) -> pd.DataFrame:
    root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    if not root.exists():
        raise FileNotFoundError(f"missing curated equities root: {root}")
    cutoff = pd.Timestamp(report_date) if report_date else None
    files = []
    for path in sorted(root.glob("date=*/data.parquet")):
        date_value = pd.Timestamp(path.parent.name.partition("=")[2])
        if cutoff is None or date_value <= cutoff:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no curated parquet files found under {root}")
    frames = [pd.read_parquet(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_partitioned(*, root: Path, report_date: str | None = None) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"missing modeling artifact root: {root}")
    cutoff = pd.Timestamp(report_date) if report_date else None
    files = []
    for path in sorted(root.glob("date=*/data.parquet")):
        date_value = pd.Timestamp(path.parent.name.partition("=")[2])
        if cutoff is None or date_value <= cutoff:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no modeling parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _write_partitioned(*, frame: pd.DataFrame, root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    partitions: list[str] = []
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    for date_value, day_frame in working.groupby(working["date"].dt.strftime("%Y-%m-%d"), sort=True):
        partition = root / f"date={date_value}"
        partition.mkdir(parents=True, exist_ok=True)
        day_frame.sort_values(["date", "symbol"]).to_parquet(partition / "data.parquet", index=False)
        partitions.append(str(partition))
    return partitions


def _write_registry(*, data_root: Path, payload: dict[str, Any]) -> None:
    path = _registry_path(data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{datetime.now(tz=UTC).timestamp()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _data_revision(*, data_root: Path) -> str:
    root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    pieces = []
    for path in sorted(root.glob("date=*/data.parquet")):
        pieces.append(f"{path.parent.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    for dataset, paths in sorted(_feature_source_contract(data_root).items()):
        for root_path in paths:
            for path in _source_files(root_path):
                pieces.append(f"{dataset}:{path.relative_to(data_root) if path.is_relative_to(data_root) else path}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()


def _features_root(*, data_root: Path, feature_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "features" / feature_version


def _labels_root(*, data_root: Path, label_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "labels" / label_version


def _registry_path(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research" / "feature_registry.json"


def _source_contract_path(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research" / "feature_source_contract" / "latest.json"


def _source_availability_path(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "data" / "source_availability" / "latest.json"
