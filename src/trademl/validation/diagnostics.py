"""Additional model diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ic_by_year(predictions: pd.Series, actuals: pd.Series, dates: pd.Series) -> dict[int, float]:
    """Return per-year rank ICs."""
    frame = pd.DataFrame({"prediction": predictions, "actual": actuals, "date": pd.to_datetime(dates)})
    return {
        int(year): float(spearmanr(group["prediction"], group["actual"], nan_policy="omit").statistic or 0.0)
        for year, group in frame.groupby(frame["date"].dt.year)
    }


def ic_by_sector(predictions: pd.Series, actuals: pd.Series, sectors: pd.Series) -> dict[str, float]:
    """Return per-sector rank ICs."""
    frame = pd.DataFrame({"prediction": predictions, "actual": actuals, "sector": sectors})
    return {
        str(sector): float(spearmanr(group["prediction"], group["actual"], nan_policy="omit").statistic or 0.0)
        for sector, group in frame.groupby("sector")
    }


def placebo_test(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_fn,
    n_shuffles: int = 5,
    *,
    validation_runner=None,
    validation_config: dict | None = None,
) -> list[float]:
    """Train on shuffled labels; resulting out-of-sample ICs should collapse near zero."""
    rng = np.random.default_rng(42)
    scores: list[float] = []
    for _ in range(n_shuffles):
        shuffled = frame.copy()
        shuffled[label_col] = rng.permutation(shuffled[label_col].to_numpy())
        if validation_runner is not None:
            folds = validation_runner(
                shuffled,
                feature_cols,
                label_col,
                model_fn,
                validation_config or {},
            )
            score = float(np.mean([fold.rank_ic for fold in folds])) if folds else 0.0
            scores.append(float(score))
            continue
        if "date" in shuffled.columns:
            shuffled = shuffled.sort_values("date").reset_index(drop=True)
        split_idx = max(1, int(len(shuffled) * 0.8))
        train = shuffled.iloc[:split_idx]
        test = shuffled.iloc[split_idx:]
        if test.empty:
            test = train.iloc[-max(1, len(train) // 5) :]
            train = train.iloc[: len(train) - len(test)]
        model = model_fn()
        model.fit(train[feature_cols], train[label_col])
        predictions = model.predict(test[feature_cols])
        score = spearmanr(predictions, test[label_col], nan_policy="omit").statistic or 0.0
        scores.append(float(score))
    return scores


def cost_stress_test(results: pd.DataFrame, multiplier: float = 2.0) -> dict[str, float]:
    """Scale explicit cost columns and recompute net return."""
    gross_return = float(results["gross_return"].sum())
    stressed_cost = float(results["cost"].sum() * multiplier)
    return {"gross_return": gross_return, "stressed_cost": stressed_cost, "net_return": gross_return - stressed_cost}
