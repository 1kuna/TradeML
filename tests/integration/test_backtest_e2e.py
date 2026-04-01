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

    corp_actions = pd.DataFrame(
        [
            {"symbol": "AAPL", "event_type": "dividend", "ex_date": prediction_frame["date"].min(), "ratio": 0.1},
        ]
    )
    first = run_backtest(
        prices,
        targets,
        apply_costs,
        {"initial_capital": 1_000_000.0, "cost_spread_bps": 5.0},
        corp_actions=corp_actions,
        prediction_frame=prediction_frame,
    )
    second = run_backtest(
        prices,
        targets,
        apply_costs,
        {"initial_capital": 1_000_000.0, "cost_spread_bps": 5.0},
        corp_actions=corp_actions,
        prediction_frame=prediction_frame,
    )

    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    pd.testing.assert_frame_equal(first.ic_time_series, second.ic_time_series)
    pd.testing.assert_frame_equal(first.decile_returns, second.decile_returns)
    assert not first.trade_log.empty
    assert not first.ic_time_series.empty
    assert not first.decile_returns.empty


def test_backtest_applies_corporate_actions_and_emits_diagnostics() -> None:
    prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")] * 2,
            "symbol": ["AAPL"] * 3 + ["MSFT"] * 3,
            "close": [100.0, 50.0, 51.0, 80.0, 81.0, 82.0],
        }
    )
    targets = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02")],
            "symbol": ["AAPL", "MSFT"],
            "score": [2.0, 1.0],
            "target_weight": [0.5, 0.5],
        }
    )
    corp_actions = pd.DataFrame(
        [
            {"symbol": "AAPL", "event_type": "split", "ex_date": "2026-01-05", "ratio": 0.5},
            {"symbol": "MSFT", "event_type": "dividend", "ex_date": "2026-01-06", "ratio": 1.0},
        ]
    )
    prediction_frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")] * 2 + [pd.Timestamp("2026-01-05")] * 2,
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "prediction": [0.20, 0.10, 0.15, 0.05],
            "label_5d": [0.03, 0.01, 0.02, -0.01],
        }
    )

    result = run_backtest(
        prices,
        targets,
        apply_costs,
        {"initial_capital": 1_000_000.0, "cost_spread_bps": 5.0},
        corp_actions=corp_actions,
        prediction_frame=prediction_frame,
    )

    assert not result.trade_log.empty
    assert not result.ic_time_series.empty
    assert not result.decile_returns.empty
    assert "action" in result.trade_log.columns
    assert (result.trade_log["action"] == "CORP_ACTION").any()
    msft_dividend = result.trade_log.loc[
        (result.trade_log["symbol"] == "MSFT") & (result.trade_log["action"] == "CORP_ACTION"),
        "trade_value",
    ]
    assert not msft_dividend.empty
    assert msft_dividend.iloc[0] > 0
