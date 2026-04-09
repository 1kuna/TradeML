from __future__ import annotations

import pandas as pd

from trademl.validation.diagnostics import portfolio_cost_stress_test, sign_flip_canary


def test_portfolio_cost_stress_test_uses_backtest_path_not_row_sums() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-02",
            "2026-01-02",
            "2026-01-05",
            "2026-01-05",
            "2026-01-05",
            "2026-01-09",
            "2026-01-09",
            "2026-01-09",
            "2026-01-12",
            "2026-01-12",
            "2026-01-12",
        ]
    )
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAPL", "MSFT", "NVDA"] * 4,
            "close": [100.0, 100.0, 100.0, 105.0, 99.0, 98.0, 105.0, 100.0, 100.0, 110.0, 99.0, 97.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAPL", "MSFT", "NVDA"] * 4,
            "prediction": [0.9, 0.2, 0.1, 0.9, 0.2, 0.1, 0.9, 0.2, 0.1, 0.9, 0.2, 0.1],
            "label_5d": [0.0] * len(dates),
        }
    )

    diagnostics = portfolio_cost_stress_test(
        prices=prices,
        prediction_frame=predictions,
        multiplier=2.0,
        initial_capital=1_000_000.0,
        cost_spread_bps=5.0,
    )

    assert diagnostics["target_rows"] > 0
    assert diagnostics["trade_rows"] > 0
    assert diagnostics["gross_return"] > 0
    assert diagnostics["net_return"] > 0
    assert diagnostics["base_net_return"] > diagnostics["net_return"]
    assert diagnostics["gross_return"] > diagnostics["net_return"]
    assert diagnostics["stressed_cost"] > 0


def test_sign_flip_canary_detects_inverted_signal_direction() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-09",
                    "2026-01-09",
                    "2026-01-09",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "NVDA", "AAPL", "MSFT", "NVDA"],
            "prediction": [0.9, 0.5, 0.1, 0.8, 0.4, 0.2],
            "label_5d": [-0.09, -0.05, -0.01, -0.08, -0.04, -0.02],
        }
    )

    canary = sign_flip_canary(predictions, label_col="label_5d")

    assert canary["preferred_direction"] == "flipped"
    assert canary["flipped_mean_rank_ic"] > canary["original_mean_rank_ic"]
    assert canary["flipped_mean_decile_spread"] > canary["original_mean_decile_spread"]
