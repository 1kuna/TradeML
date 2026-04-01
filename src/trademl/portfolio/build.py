"""Portfolio construction helpers."""

from __future__ import annotations

import math

import pandas as pd


def build_portfolio(scores: pd.Series | pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build equal-weight top-quintile target weights."""
    if isinstance(scores, pd.DataFrame):
        frame = scores.copy()
    else:
        frame = scores.rename("score").reset_index()
        frame.columns = ["symbol", "score"]
    if "score" not in frame.columns:
        raise ValueError("scores dataframe must include a 'score' column")
    if "date" not in frame.columns:
        frame["date"] = config.get("date")

    targets: list[pd.DataFrame] = []
    for date, group in frame.groupby("date"):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        n_positions = max(1, math.ceil(len(ordered) * 0.2))
        top = ordered.head(n_positions).copy()
        top["target_weight"] = 1.0 / n_positions
        top["date"] = date
        targets.append(top[["date", "symbol", "score", "target_weight"]])
    return pd.concat(targets, ignore_index=True) if targets else pd.DataFrame(columns=["date", "symbol", "score", "target_weight"])
