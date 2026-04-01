from __future__ import annotations

import pandas as pd

from trademl.portfolio.build import build_portfolio


def test_top_quintile_equal_weight_portfolio() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")] * 5 + [pd.Timestamp("2026-01-05")] * 5,
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "META"] * 2,
            "score": [5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "earnings_within_5d": [False, False, False, False, False, False, False, False, False, False],
        }
    )
    portfolio = build_portfolio(scores, {"rebalance_day": "FRI"})

    assert len(portfolio) == 1
    assert portfolio.iloc[0]["symbol"] == "AAPL"
    assert portfolio["target_weight"].sum() == 1.0


def test_earnings_flag_excludes_names_from_rebalance() -> None:
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")] * 5,
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
            "earnings_within_5d": [True, False, False, False, False],
        }
    )

    portfolio = build_portfolio(scores, {"rebalance_day": "FRI"})

    assert "AAPL" not in set(portfolio["symbol"])
