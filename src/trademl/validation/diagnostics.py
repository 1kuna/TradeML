"""Additional model diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.backtest.engine import run_backtest
from trademl.costs.models import apply_costs
from trademl.portfolio.build import build_portfolio
from trademl.validation.metrics import mean_daily_bucket_spread, rank_ic


def ic_by_year(predictions: pd.Series, actuals: pd.Series, dates: pd.Series) -> dict[int, float]:
    """Return per-year rank ICs."""
    frame = pd.DataFrame({"prediction": predictions, "actual": actuals, "date": pd.to_datetime(dates)})
    return {
        int(year): rank_ic(group["prediction"], group["actual"])
        for year, group in frame.groupby(frame["date"].dt.year)
    }


def ic_by_sector(predictions: pd.Series, actuals: pd.Series, sectors: pd.Series) -> dict[str, float]:
    """Return per-sector rank ICs."""
    frame = pd.DataFrame({"prediction": predictions, "actual": actuals, "sector": sectors})
    return {
        str(sector): rank_ic(group["prediction"], group["actual"])
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
        score = rank_ic(pd.Series(predictions, index=test.index), test[label_col])
        scores.append(float(score))
    return scores


def cost_stress_test(results: pd.DataFrame, multiplier: float = 2.0) -> dict[str, float]:
    """Scale explicit cost columns and recompute net return."""
    gross_return = float(results["gross_return"].sum())
    stressed_cost = float(results["cost"].sum() * multiplier)
    return {"gross_return": gross_return, "stressed_cost": stressed_cost, "net_return": gross_return - stressed_cost}


def portfolio_cost_stress_test(
    *,
    prices: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    multiplier: float = 2.0,
    initial_capital: float = 1_000_000.0,
    cost_spread_bps: float = 5.0,
    rebalance_day: str = "FRI",
) -> dict[str, float]:
    """Backtest the prediction stream and report gross/base/stressed portfolio returns."""
    if prediction_frame.empty:
        return {
            "gross_return": 0.0,
            "base_net_return": 0.0,
            "net_return": 0.0,
            "stressed_cost": 0.0,
            "trade_rows": 0,
            "target_rows": 0,
        }
    scores = prediction_frame.groupby(["date", "symbol"], as_index=False)["prediction"].mean().rename(columns={"prediction": "score"})
    targets = build_portfolio(scores, {"rebalance_day": rebalance_day})
    if targets.empty:
        return {
            "gross_return": 0.0,
            "base_net_return": 0.0,
            "net_return": 0.0,
            "stressed_cost": 0.0,
            "trade_rows": 0,
            "target_rows": 0,
        }
    price_frame = prices[["date", "symbol", "close"]].copy()
    base_config = {"initial_capital": initial_capital, "cost_spread_bps": cost_spread_bps}
    gross_result = run_backtest(
        price_frame,
        targets,
        apply_costs,
        {**base_config, "cost_spread_bps": 0.0},
        prediction_frame=prediction_frame,
    )
    base_result = run_backtest(
        price_frame,
        targets,
        apply_costs,
        base_config,
        prediction_frame=prediction_frame,
    )
    stressed_result = run_backtest(
        price_frame,
        targets,
        apply_costs,
        {**base_config, "stress_multiplier": multiplier},
        prediction_frame=prediction_frame,
    )
    gross_equity = float(gross_result.equity_curve["equity"].iloc[-1]) if not gross_result.equity_curve.empty else initial_capital
    base_equity = float(base_result.equity_curve["equity"].iloc[-1]) if not base_result.equity_curve.empty else initial_capital
    stressed_equity = float(stressed_result.equity_curve["equity"].iloc[-1]) if not stressed_result.equity_curve.empty else initial_capital
    return {
        "gross_return": (gross_equity / initial_capital) - 1.0,
        "base_net_return": (base_equity / initial_capital) - 1.0,
        "net_return": (stressed_equity / initial_capital) - 1.0,
        "stressed_cost": max(0.0, (gross_equity - stressed_equity) / initial_capital),
        "trade_rows": int(len(stressed_result.trade_log)),
        "target_rows": int(len(targets)),
    }


def sign_flip_canary(prediction_frame: pd.DataFrame, *, label_col: str = "label_5d") -> dict[str, object]:
    """Compare original vs sign-flipped ranking diagnostics."""
    if prediction_frame.empty or label_col not in prediction_frame.columns:
        return {
            "preferred_direction": "original",
            "original_mean_rank_ic": 0.0,
            "flipped_mean_rank_ic": 0.0,
            "original_mean_decile_spread": 0.0,
            "flipped_mean_decile_spread": 0.0,
            "original_ic_by_year": {},
            "flipped_ic_by_year": {},
        }

    original = prediction_frame.copy()
    flipped = prediction_frame.copy()
    flipped["prediction"] = -flipped["prediction"]
    original_mean_ic = rank_ic(original["prediction"], original[label_col])
    flipped_mean_ic = rank_ic(flipped["prediction"], flipped[label_col])
    original_year = ic_by_year(original["prediction"], original[label_col], original["date"])
    flipped_year = ic_by_year(flipped["prediction"], flipped[label_col], flipped["date"])
    original_spread = mean_daily_bucket_spread(original, label_col=label_col)
    flipped_spread = mean_daily_bucket_spread(flipped, label_col=label_col)
    preferred_direction = "flipped" if (flipped_mean_ic > original_mean_ic and flipped_spread >= original_spread) else "original"
    return {
        "preferred_direction": preferred_direction,
        "original_mean_rank_ic": original_mean_ic,
        "flipped_mean_rank_ic": flipped_mean_ic,
        "original_mean_decile_spread": original_spread,
        "flipped_mean_decile_spread": flipped_spread,
        "original_ic_by_year": original_year,
        "flipped_ic_by_year": flipped_year,
    }
