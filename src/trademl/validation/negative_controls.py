"""Negative-control gate helpers for research candidates."""

from __future__ import annotations

from typing import Any


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
