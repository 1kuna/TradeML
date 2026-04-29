"""Portfolio construction helpers."""

from __future__ import annotations

import math

import pandas as pd


def build_portfolio(scores: pd.Series | pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build deterministic long-only target weights."""
    if isinstance(scores, pd.DataFrame):
        frame = scores.copy()
    else:
        frame = scores.rename("score").reset_index()
        frame.columns = ["symbol", "score"]
    if "score" not in frame.columns:
        raise ValueError("scores dataframe must include a 'score' column")
    if "date" not in frame.columns:
        frame["date"] = config.get("date")
    rebalance_day = str(config.get("rebalance_day", "FRI")).upper()[:3]
    weekday_lookup = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    if frame["date"].notna().all():
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.loc[frame["date"].dt.weekday == weekday_lookup.get(rebalance_day, 4)].copy()
    if "earnings_within_5d" in frame.columns:
        frame = frame.loc[~frame["earnings_within_5d"].fillna(False)].copy()

    profile = str(config.get("profile") or config.get("portfolio_profile") or config.get("method") or "equal_weight_top_quintile")
    targets: list[pd.DataFrame] = []
    for date, group in frame.groupby("date"):
        if profile == "cost_aware_long_only_v1" and "adv_dollar_20d" in group.columns:
            min_adv = float(config.get("min_adv_dollar", 0.0) or 0.0)
            group = group.loc[group["adv_dollar_20d"].fillna(0.0) >= min_adv].copy()
        if group.empty:
            continue
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        n_positions = max(1, math.ceil(len(ordered) * 0.2))
        top = ordered.head(n_positions).copy()
        weight = 1.0 / n_positions
        if profile == "cost_aware_long_only_v1":
            max_weight = float(config.get("max_single_name_weight", weight) or weight)
            weight = min(weight, max_weight)
        top["target_weight"] = weight
        top["date"] = date
        targets.append(top[["date", "symbol", "score", "target_weight"]])
    return pd.concat(targets, ignore_index=True) if targets else pd.DataFrame(columns=["date", "symbol", "score", "target_weight"])
