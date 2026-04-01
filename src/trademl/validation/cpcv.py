"""Simplified combinatorially purged cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr


@dataclass(slots=True)
class CPCVFoldResult:
    """Single CPCV fold."""

    fold: int
    test_folds: tuple[int, ...]
    train_rows: int
    test_rows: int
    retention: float
    in_sample_score: float
    out_of_sample_score: float
    oof_predictions: pd.DataFrame


def combinatorially_purged_cv(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_fn,
    *,
    n_folds: int = 8,
    embargo_days: int = 10,
    n_test_folds: int = 2,
) -> list[CPCVFoldResult]:
    """Run contiguous blocked CPCV with embargo around each test fold."""
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    unique_dates = pd.Index(sorted(working["date"].dropna().unique()))
    if unique_dates.empty:
        return []
    folds = list(_split_dates(unique_dates, n_folds))
    results: list[CPCVFoldResult] = []

    for fold_idx, combo in enumerate(combinations(range(len(folds)), n_test_folds)):
        test_dates = pd.Index(sorted(pd.concat([pd.Series(folds[idx]) for idx in combo]).tolist()))
        if len(test_dates) == 0:
            continue
        purge_mask = pd.Series(False, index=working.index)
        for fold_number in combo:
            local_test_dates = folds[fold_number]
            if len(local_test_dates) == 0:
                continue
            lower_bound = local_test_dates[0] - pd.Timedelta(days=embargo_days)
            upper_bound = local_test_dates[-1] + pd.Timedelta(days=embargo_days)
            local_symbols = set(working.loc[working["date"].isin(local_test_dates), "symbol"])
            purge_mask = purge_mask | (
                working["date"].between(lower_bound, upper_bound) & working["symbol"].isin(local_symbols)
            )
        train_frame = working.loc[~purge_mask].dropna(subset=[label_col])
        test_frame = working.loc[working["date"].isin(test_dates)].dropna(subset=[label_col])
        if train_frame.empty or test_frame.empty:
            continue
        model = model_fn()
        train_X = train_frame[feature_cols].replace([float("inf"), float("-inf")], 0.0).astype(float).fillna(0.0)
        test_X = test_frame[feature_cols].replace([float("inf"), float("-inf")], 0.0).astype(float).fillna(0.0)
        model.fit(train_X, train_frame[label_col])
        predictions = test_frame[["date", "symbol", label_col]].copy()
        predictions["prediction"] = model.predict(test_X)
        retention = float(len(train_frame) / max(1, len(working.dropna(subset=[label_col]))))
        in_sample_score = float(spearmanr(model.predict(train_X), train_frame[label_col], nan_policy="omit").statistic or 0.0)
        out_of_sample_score = float(spearmanr(predictions["prediction"], predictions[label_col], nan_policy="omit").statistic or 0.0)
        results.append(
            CPCVFoldResult(
                fold=fold_idx,
                test_folds=combo,
                train_rows=len(train_frame),
                test_rows=len(test_frame),
                retention=retention,
                in_sample_score=in_sample_score,
                out_of_sample_score=out_of_sample_score,
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
