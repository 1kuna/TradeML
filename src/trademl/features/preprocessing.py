"""Feature preprocessing helpers."""

from __future__ import annotations

import pandas as pd


def rank_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    date_col: str = "date",
    missing_threshold: float = 0.30,
    date_level_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Rank-normalize cross-sectional features into the [-1, 1] range."""
    normalized = df.copy()
    date_level_cols = date_level_cols or set()

    for date, date_frame in normalized.groupby(date_col):
        index = date_frame.index
        for column in feature_cols:
            if column in date_level_cols:
                continue
            series = date_frame[column]
            missing_fraction = float(series.isna().mean())
            if missing_fraction > missing_threshold:
                normalized.loc[index, column] = 0.0
                continue
            ranked = series.rank(method="average", pct=True)
            normalized.loc[index, column] = ranked.fillna(0.5) * 2.0 - 1.0
    return normalized
