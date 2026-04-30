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
        "feature_groups": _feature_groups_for(feature_set=feature_set, feature_version=feature_version),
        "feature_group_metadata": _feature_group_metadata(
            frame=feature_frame,
            feature_set=feature_set,
            feature_version=feature_version,
        ),
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
    if "multi_source" in name or "fundamental" in name or "sec" in name or "filing" in name:
        groups.append("fundamentals_sec")
    if "multi_source" in name or "news" in name or "event" in name:
        groups.append("news_events")
    if "multi_source" in name or "minute" in name or "intraday" in name:
        groups.append("minute_daily")
    return groups


def _feature_group_metadata(*, frame: pd.DataFrame, feature_set: str, feature_version: str) -> dict[str, Any]:
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
            "source_datasets": ["sec_filing_index", "fundamentals_daily"],
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


def _fundamental_features(*, data_root: Path, dates: pd.Series) -> pd.DataFrame:
    path = data_root / "data" / "reference" / "fundamentals_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = {"symbol", "metric_date", "metric_name", "metric_value"}
    if not required.issubset(frame.columns):
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
    path = data_root / "data" / "reference" / "sec_filing_index.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if not {"symbol", "form"}.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    date_col = "accepted_at" if "accepted_at" in frame.columns else "filing_date"
    frame["available_at"] = pd.to_datetime(frame.get(date_col), errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["available_at"])
    rows: list[dict[str, Any]] = []
    for date_value in sorted(pd.to_datetime(dates).dropna().unique()):
        current = pd.Timestamp(date_value)
        eligible = frame[frame["available_at"] <= current]
        if eligible.empty:
            continue
        for symbol, group in eligible.groupby("symbol"):
            age = (current - group["available_at"].max()).days
            rows.append(
                {
                    "date": current,
                    "symbol": str(symbol),
                    "sec_filings_30d": float((group["available_at"] >= current - pd.Timedelta(days=30)).sum()),
                    "sec_filings_90d": float((group["available_at"] >= current - pd.Timedelta(days=90)).sum()),
                    "sec_8k_30d": float(((group["form"].astype(str).str.upper() == "8-K") & (group["available_at"] >= current - pd.Timedelta(days=30))).sum()),
                    "sec_10q_90d": float(((group["form"].astype(str).str.upper() == "10-Q") & (group["available_at"] >= current - pd.Timedelta(days=90))).sum()),
                    "sec_10k_90d": float(((group["form"].astype(str).str.upper() == "10-K") & (group["available_at"] >= current - pd.Timedelta(days=90))).sum()),
                    "sec_days_since_last_filing": float(max(age, 0)),
                }
            )
    return pd.DataFrame(rows)


def _news_event_features(*, data_root: Path, dates: pd.Series) -> pd.DataFrame:
    files = sorted((data_root / "data" / "raw" / "ticker_news").glob("date=*/data.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
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
    source_col = "source" if "source" in frame.columns else "vendor" if "vendor" in frame.columns else None
    rows: list[dict[str, Any]] = []
    for date_value in sorted(pd.to_datetime(dates).dropna().unique()):
        current = pd.Timestamp(date_value)
        eligible = frame[frame["available_at"] <= current]
        if eligible.empty:
            continue
        for symbol, group in eligible.groupby("symbol"):
            last_age = (current - group["available_at"].max()).days
            recent_1d = group[group["available_at"] >= current - pd.Timedelta(days=1)]
            recent_7d = group[group["available_at"] >= current - pd.Timedelta(days=7)]
            prior_30d = group[(group["available_at"] < current - pd.Timedelta(days=7)) & (group["available_at"] >= current - pd.Timedelta(days=37))]
            source_count = recent_7d[source_col].nunique() if source_col else 0
            vendor_col = "vendor" if "vendor" in group.columns else "source_name" if "source_name" in group.columns else source_col
            vendor_count = recent_7d[vendor_col].nunique() if vendor_col else 0
            baseline = max(float(len(prior_30d)) / 30.0 * 7.0, 1.0)
            headline_col = "headline" if "headline" in group.columns else "title" if "title" in group.columns else None
            novelty = float(recent_7d[headline_col].astype(str).nunique()) / max(1.0, float(len(recent_7d))) if headline_col and not recent_7d.empty else 0.0
            rows.append(
                {
                    "date": current,
                    "symbol": str(symbol),
                    "news_count_1d": float(len(recent_1d)),
                    "news_count_7d": float(len(recent_7d)),
                    "news_source_count_7d": float(source_count),
                    "news_vendor_count_7d": float(vendor_count),
                    "news_abnormal_volume_7d": float(len(recent_7d)) / baseline,
                    "news_novelty_proxy_7d": novelty,
                    "news_days_since_last": float(max(last_age, 0)),
                }
            )
    return pd.DataFrame(rows)


def _minute_daily_features(*, data_root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    files = sorted((data_root / "data" / "raw" / "equities_minute").glob("date=*/data.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {"symbol", "timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
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
    calendar = sorted(pd.to_datetime(panel["date"]).dropna().unique())
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
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()


def _features_root(*, data_root: Path, feature_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "features" / feature_version


def _labels_root(*, data_root: Path, label_version: str) -> Path:
    return data_root / "data" / "curated" / "modeling" / "labels" / label_version


def _registry_path(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research" / "feature_registry.json"
