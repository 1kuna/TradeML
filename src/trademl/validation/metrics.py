"""Shared prediction-ranking metrics."""

from __future__ import annotations

import math
import warnings

import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


def rank_ic(predictions: pd.Series, actuals: pd.Series) -> float:
    """Return Spearman rank IC with non-finite results normalized to zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        stat = spearmanr(predictions, actuals, nan_policy="omit").statistic
    value = float(stat)
    return value if math.isfinite(value) else 0.0


def prediction_buckets(predictions: pd.Series, *, max_buckets: int = 10) -> pd.Series:
    """Return stable rank buckets for prediction scores."""
    bucket_count = min(max_buckets, len(predictions))
    if bucket_count < 2:
        return pd.Series([pd.NA] * len(predictions), index=predictions.index, dtype="Int64")
    return pd.qcut(
        predictions.rank(method="first"),
        q=bucket_count,
        labels=False,
        duplicates="drop",
    )


def bucket_metrics(predictions: pd.DataFrame, *, label_col: str) -> tuple[float, float, dict[str, float]]:
    """Return top-bottom spread, top-bucket hit rate, and mean return by bucket."""
    scores = predictions.copy()
    scores["bucket"] = prediction_buckets(scores["prediction"])
    if scores["bucket"].nunique() < 2:
        return 0.0, 0.0, {}
    top_bucket = scores["bucket"].max()
    bottom_bucket = scores["bucket"].min()
    top = scores.loc[scores["bucket"] == top_bucket, label_col]
    bottom = scores.loc[scores["bucket"] == bottom_bucket, label_col]
    decile_spread = float(top.mean() - bottom.mean())
    hit_rate = float((top > 0).mean())
    bucket_returns = {str(int(bucket) + 1): float(group[label_col].mean()) for bucket, group in scores.groupby("bucket")}
    return decile_spread, hit_rate, bucket_returns


def mean_daily_bucket_spread(predictions: pd.DataFrame, *, label_col: str) -> float:
    """Return average per-date top-bottom bucket spread."""
    spreads: list[float] = []
    for _, group in predictions.groupby(pd.to_datetime(predictions["date"])):
        spread, _hit_rate, _returns = bucket_metrics(group, label_col=label_col)
        if _returns:
            spreads.append(spread)
    return float(sum(spreads) / len(spreads)) if spreads else 0.0
