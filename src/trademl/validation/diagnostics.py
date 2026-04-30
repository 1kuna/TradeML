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


def ic_by_quarter(predictions: pd.Series, actuals: pd.Series, dates: pd.Series) -> dict[str, float]:
    """Return per-quarter rank ICs."""
    frame = pd.DataFrame({"prediction": predictions, "actual": actuals, "date": pd.to_datetime(dates)})
    return {
        str(quarter): rank_ic(group["prediction"], group["actual"])
        for quarter, group in frame.groupby(frame["date"].dt.to_period("Q"))
    }


def fold_window_summary(folds: list[object]) -> list[dict[str, object]]:
    """Return a compact fold stability summary."""
    rows: list[dict[str, object]] = []
    for idx, fold in enumerate(folds, start=1):
        predictions = getattr(fold, "predictions", pd.DataFrame())
        date_min = date_max = None
        if isinstance(predictions, pd.DataFrame) and not predictions.empty and "date" in predictions.columns:
            dates = pd.to_datetime(predictions["date"])
            date_min = dates.min().date().isoformat()
            date_max = dates.max().date().isoformat()
        rows.append(
            {
                "fold": idx,
                "rank_ic": float(getattr(fold, "rank_ic", 0.0) or 0.0),
                "decile_spread": float(getattr(fold, "decile_spread", 0.0) or 0.0),
                "hit_rate": float(getattr(fold, "hit_rate", 0.0) or 0.0),
                "date_start": date_min,
                "date_end": date_max,
            }
        )
    return rows


def feature_dependence_summary(frame: pd.DataFrame, feature_cols: list[str], label_col: str, *, primary_score: float) -> dict[str, object]:
    """Return lightweight single-feature dependence diagnostics."""
    if frame.empty or not feature_cols or label_col not in frame.columns:
        return {
            "max_single_feature_abs_ic": 0.0,
            "max_single_feature_score_drop": 0.0,
            "min_feature_ablation_score_ratio": 1.0,
            "top_fragile_features": [],
        }
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        if feature not in frame.columns:
            continue
        value = rank_ic(frame[feature].fillna(0.0), frame[label_col])
        rows.append({"feature": feature, "single_feature_ic": float(value), "abs_ic": abs(float(value))})
    rows = sorted(rows, key=lambda item: float(item["abs_ic"]), reverse=True)
    max_abs = float(rows[0]["abs_ic"]) if rows else 0.0
    baseline = abs(float(primary_score or 0.0))
    if baseline <= 0:
        drop = 0.0
        ratio = 1.0
    else:
        drop = min(1.0, max_abs / baseline)
        ratio = max(0.0, 1.0 - drop)
    return {
        "max_single_feature_abs_ic": max_abs,
        "max_single_feature_score_drop": drop,
        "min_feature_ablation_score_ratio": ratio,
        "top_fragile_features": rows[:10],
    }


def model_comparison_summary(report: dict[str, object]) -> dict[str, object]:
    """Return cross-model primary-score deltas for a report."""
    scores: dict[str, float] = {}
    for model_name in ("ridge", "lightgbm", "catboost", "ensemble"):
        payload = report.get(model_name)
        if isinstance(payload, dict) and not payload.get("skipped") and payload.get("mean_rank_ic") is not None:
            scores[model_name] = float(payload.get("mean_rank_ic") or 0.0)
    best_name = max(scores, key=scores.get) if scores else None
    ridge_score = scores.get("ridge")
    return {
        "scores": scores,
        "best_model": best_name,
        "best_score": scores.get(best_name) if best_name else None,
        "best_minus_ridge": (scores[best_name] - ridge_score) if best_name and ridge_score is not None else None,
        "catboost_minus_lightgbm": (
            scores["catboost"] - scores["lightgbm"]
            if "catboost" in scores and "lightgbm" in scores
            else None
        ),
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
