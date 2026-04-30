"""Negative-control gate helpers for research candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trademl.validation.metrics import rank_ic


NEGATIVE_CONTROL_KEYS = (
    "shuffled_label_max_abs_ic",
    "date_shifted_label_max_abs_ic",
    "random_feature_max_abs_ic",
    "ticker_news_permutation_max_abs_ic",
)


def evaluate_negative_controls(
    diagnostics: dict[str, Any],
    *,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Return machine-readable failures for configured negative controls."""
    controls = diagnostics.get("negative_controls")
    if not isinstance(controls, dict):
        if bool(gate.get("require_negative_controls", True)):
            return {"present": False, "gate_failures": ["negative_controls.missing"], "controls": {}}
        return {"present": False, "gate_failures": [], "controls": {}}
    failures: list[str] = []
    max_abs_ic = float(gate.get("max_abs_negative_control_ic", 0.10) or 0.10)
    for key in NEGATIVE_CONTROL_KEYS:
        if key in controls and abs(float(controls.get(key) or 0.0)) > max_abs_ic:
            failures.append(f"negative_control.{key}>{max_abs_ic}")
    future_limit = float(gate.get("max_abs_future_news_leak_ic", max_abs_ic) or max_abs_ic)
    if "future_news_leak_sentinel_ic" in controls and abs(float(controls.get("future_news_leak_sentinel_ic") or 0.0)) > future_limit:
        failures.append(f"future_news_leak_sentinel_ic>{future_limit}")
    max_drop = controls.get("max_single_feature_score_drop")
    if max_drop is not None:
        max_drop_limit = float(gate.get("max_single_feature_score_drop", 0.75) or 0.75)
        if float(max_drop or 0.0) > max_drop_limit:
            failures.append(f"single_feature_dependence>{max_drop_limit}")
    min_ablation_ratio = controls.get("min_feature_ablation_score_ratio")
    if min_ablation_ratio is not None:
        ratio_floor = float(gate.get("min_feature_ablation_score_ratio", 0.25) or 0.25)
        if float(min_ablation_ratio or 0.0) < ratio_floor:
            failures.append(f"feature_ablation_ratio<{ratio_floor}")
    return {"present": True, "gate_failures": failures, "controls": controls}


def compute_negative_control_diagnostics(
    *,
    predictions: pd.DataFrame,
    label_col: str,
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Compute deterministic negative-control diagnostics for one candidate."""
    if predictions.empty or label_col not in predictions.columns:
        return {
            "shuffled_label_max_abs_ic": 0.0,
            "date_shifted_label_max_abs_ic": 0.0,
            "random_feature_max_abs_ic": 0.0,
            "ticker_news_permutation_max_abs_ic": 0.0,
            "future_news_leak_sentinel_ic": 0.0,
            "controls_version": "negative_controls_v1",
        }
    frame = predictions[["date", "symbol", "prediction", label_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(0.0)
    rng = np.random.default_rng(42)
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=frame.index)
    random_feature = pd.Series(rng.normal(size=len(frame)), index=frame.index)
    shifted = frame[["date", "symbol", label_col]].copy()
    shifted[f"{label_col}_shifted"] = shifted.groupby("symbol", sort=False)[label_col].shift(-1)
    has_news = any(str(column).startswith("news_") or str(column).startswith("ticker_news_") for column in feature_cols)
    controls = {
        "shuffled_label_max_abs_ic": abs(float(rank_ic(frame["prediction"], shuffled))),
        "date_shifted_label_max_abs_ic": abs(float(rank_ic(frame["prediction"], shifted[f"{label_col}_shifted"].fillna(0.0)))),
        "random_feature_max_abs_ic": abs(float(rank_ic(random_feature, labels))),
        "ticker_news_permutation_max_abs_ic": 0.0,
        "future_news_leak_sentinel_ic": 0.0,
        "controls_version": "negative_controls_v1",
    }
    if has_news:
        permuted = frame.copy()
        permuted["symbol"] = rng.permutation(permuted["symbol"].astype(str).to_numpy())
        controls["ticker_news_permutation_max_abs_ic"] = abs(float(rank_ic(permuted["prediction"], labels)))
        future_label = frame.groupby("symbol", sort=False)[label_col].shift(-1)
        controls["future_news_leak_sentinel_ic"] = abs(float(rank_ic(future_label.fillna(0.0), labels)))
    ablation = feature_frame.attrs.get("feature_ablation") if hasattr(feature_frame, "attrs") else None
    if isinstance(ablation, dict):
        for key in ("max_single_feature_score_drop", "min_feature_ablation_score_ratio"):
            if key in ablation:
                controls[key] = ablation[key]
    return controls
