"""Simplified combinatorially purged cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class CPCVFoldResult:
    """Single CPCV fold."""

    fold: int
    train_rows: int
    test_rows: int
    retention: float
    oof_predictions: pd.DataFrame


def combinatorially_purged_cv(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_fn,
    *,
    n_folds: int = 8,
    embargo_days: int = 10,
) -> list[CPCVFoldResult]:
    """Run contiguous blocked CPCV with embargo around each test fold."""
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    unique_dates = pd.Index(sorted(working["date"].dropna().unique()))
    if unique_dates.empty:
        return []
    folds = list(_split_dates(unique_dates, n_folds))
    results: list[CPCVFoldResult] = []

    for fold_idx, test_dates in enumerate(folds):
        if len(test_dates) == 0:
            continue
        test_start = test_dates[0]
        test_end = test_dates[-1]
        lower_bound = test_start - pd.Timedelta(days=embargo_days)
        upper_bound = test_end + pd.Timedelta(days=embargo_days)
        train_frame = working.loc[~working["date"].between(lower_bound, upper_bound)].dropna(subset=feature_cols + [label_col])
        test_frame = working.loc[working["date"].isin(test_dates)].dropna(subset=feature_cols + [label_col])
        if train_frame.empty or test_frame.empty:
            continue
        model = model_fn()
        model.fit(train_frame[feature_cols], train_frame[label_col])
        predictions = test_frame[["date", "symbol", label_col]].copy()
        predictions["prediction"] = model.predict(test_frame[feature_cols])
        retention = float(len(train_frame) / max(1, len(working.dropna(subset=feature_cols + [label_col]))))
        results.append(
            CPCVFoldResult(
                fold=fold_idx,
                train_rows=len(train_frame),
                test_rows=len(test_frame),
                retention=retention,
                oof_predictions=predictions,
            )
        )
    return results


def _split_dates(unique_dates: pd.Index, n_folds: int) -> list[pd.Index]:
    fold_sizes = [len(unique_dates) // n_folds] * n_folds
    for idx in range(len(unique_dates) % n_folds):
        fold_sizes[idx] += 1
    folds: list[pd.Index] = []
    cursor = 0
    for size in fold_sizes:
        folds.append(unique_dates[cursor : cursor + size])
        cursor += size
    return folds
