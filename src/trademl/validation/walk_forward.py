"""Expanding walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trademl.validation.metrics import bucket_metrics, rank_ic


@dataclass(slots=True)
class FoldResult:
    """Per-fold walk-forward metrics and predictions."""

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rank_ic: float
    decile_spread: float
    hit_rate: float
    bucket_returns: dict[str, float]
    predictions: pd.DataFrame


def expanding_walk_forward(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_fn,
    config: dict,
) -> list[FoldResult]:
    """Run expanding-window walk-forward validation with a 5-day purge."""
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    unique_dates = pd.Index(sorted(working["date"].dropna().unique()))
    if unique_dates.empty:
        return []

    initial_years = int(config.get("initial_train_years", 2))
    step_months = int(config.get("step_months", 3))
    purge_days = int(config.get("purge_days", 5))
    initial_cutoff = unique_dates.min() + pd.DateOffset(years=initial_years)
    fold_starts = pd.date_range(initial_cutoff, unique_dates.max(), freq=pd.DateOffset(months=step_months))

    results: list[FoldResult] = []
    for test_start in fold_starts:
        test_end = min(test_start + pd.DateOffset(months=step_months) - pd.Timedelta(days=1), unique_dates.max())
        train_dates = unique_dates[unique_dates < test_start]
        if len(train_dates) <= purge_days:
            continue
        purged_train_dates = train_dates[:-purge_days]
        test_dates = unique_dates[(unique_dates >= test_start) & (unique_dates <= test_end)]
        if purged_train_dates.empty or test_dates.empty:
            continue

        train_frame = working.loc[working["date"].isin(purged_train_dates)].dropna(subset=[label_col])
        test_frame = working.loc[working["date"].isin(test_dates)].dropna(subset=[label_col])
        if train_frame.empty or test_frame.empty:
            continue

        train_X = train_frame[feature_cols].replace([np.inf, -np.inf], 0.0).astype(float).fillna(0.0)
        test_X = test_frame[feature_cols].replace([np.inf, -np.inf], 0.0).astype(float).fillna(0.0)
        model = model_fn()
        model.fit(train_X, train_frame[label_col])
        predictions = test_frame[["date", "symbol", label_col]].copy()
        predictions["prediction"] = model.predict(test_X)
        fold_rank_ic = rank_ic(predictions["prediction"], predictions[label_col])
        decile_spread, hit_rate, bucket_returns = bucket_metrics(predictions, label_col=label_col)
        results.append(
            FoldResult(
                train_end=pd.Timestamp(purged_train_dates[-1]),
                test_start=pd.Timestamp(test_dates[0]),
                test_end=pd.Timestamp(test_dates[-1]),
                rank_ic=fold_rank_ic,
                decile_spread=decile_spread,
                hit_rate=hit_rate,
                bucket_returns=bucket_returns,
                predictions=predictions,
            )
        )
    return results
