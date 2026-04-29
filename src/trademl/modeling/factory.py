"""Point-in-time modeling feature and label artifact factory."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from trademl.features.equities import build_features
from trademl.labels.returns import build_labels

DEFAULT_FEATURE_SET = "daily_price_liquidity_v1"
DEFAULT_FEATURE_VERSION = "price_liquidity_v1"
DEFAULT_LABEL_DEFINITION = "universe_relative_forward_return"
DEFAULT_LABEL_VERSION = "universe_relative_forward_return_v1"
DEFAULT_LABEL_HORIZONS = (1, 5, 20)


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
    feature_config: dict[str, Any],
    feature_set: str,
    feature_version: str,
) -> pd.DataFrame:
    features = build_features(panel, feature_config)
    features["date"] = pd.to_datetime(features["date"])
    features["feature_available_at"] = features["date"].dt.tz_localize("UTC")
    features["feature_set"] = feature_set
    features["feature_version"] = feature_version
    features["universe_member"] = True
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)


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
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()


def _features_root(*, data_root: Path, feature_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "features" / feature_version


def _labels_root(*, data_root: Path, label_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "labels" / label_version


def _registry_path(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research" / "feature_registry.json"
