from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.backtest.engine import run_backtest
from trademl.costs.models import apply_costs
from trademl.features.equities import build_features
from trademl.features.preprocessing import rank_normalize
from trademl.labels.returns import build_labels
from trademl.models.ridge import RidgeModel
from trademl.portfolio.build import build_portfolio
from trademl.validation.walk_forward import expanding_walk_forward


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    rows = []
    for symbol_idx, symbol in enumerate(symbols):
        signal = np.sin(np.arange(len(dates)) / 15 + symbol_idx)
        close = 50 + symbol_idx * 10 + np.cumsum(0.1 + signal * 0.3 + rng.normal(0, 0.4, len(dates)))
        close = np.maximum(close, 5)
        open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
        volume = 1_000_000 + rng.integers(0, 100_000, len(dates))
        for date, open_price, close_price, vol in zip(dates, open_, close, volume, strict=False):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": float(open_price),
                    "high": float(max(open_price, close_price) * 1.01),
                    "low": float(min(open_price, close_price) * 0.99),
                    "close": float(close_price),
                    "vwap": float((open_price + close_price) / 2),
                    "volume": int(vol),
                }
            )
    return pd.DataFrame(rows)


def test_full_backtest_pipeline_is_deterministic() -> None:
    panel = _synthetic_panel()
    features = build_features(
        panel,
        {
            "price": {"momentum": [5, 20, 60, 126], "reversal": [1, 5], "drawdown": [20, 60]},
            "volatility": {"realized": [20, 60], "idiosyncratic": [60]},
            "liquidity": {"adv_dollar": [20], "amihud": [20]},
            "controls": {"log_price": True},
        },
    )
    labels = build_labels(panel, horizon=5)
    merged = features.merge(labels, on=["date", "symbol"]).dropna()
    feature_cols = [column for column in merged.columns if column not in {"date", "symbol", "raw_forward_return_5d", "label_5d", "earnings_within_5d"}]
    normalized = rank_normalize(merged, feature_cols)

    folds = expanding_walk_forward(
        normalized,
        feature_cols,
        "label_5d",
        lambda: RidgeModel(alpha=1.0),
        {"initial_train_years": 2, "step_months": 3, "purge_days": 5},
    )
    assert folds

    prediction_frame = pd.concat([fold.predictions for fold in folds], ignore_index=True)
    rebalance_scores = prediction_frame.groupby(["date", "symbol"], as_index=False)["prediction"].mean()
    targets = build_portfolio(rebalance_scores.rename(columns={"prediction": "score"}), {})
    prices = panel[["date", "symbol", "close"]]

    first = run_backtest(prices, targets, apply_costs, {"initial_capital": 1_000_000.0, "cost_spread_bps": 5.0})
    second = run_backtest(prices, targets, apply_costs, {"initial_capital": 1_000_000.0, "cost_spread_bps": 5.0})

    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    assert not first.trade_log.empty
