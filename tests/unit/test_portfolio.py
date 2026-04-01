from __future__ import annotations

import pandas as pd

from trademl.portfolio.build import build_portfolio


def test_top_quintile_equal_weight_portfolio() -> None:
    scores = pd.Series({"AAPL": 5.0, "MSFT": 4.0, "NVDA": 3.0, "AMZN": 2.0, "META": 1.0})
    portfolio = build_portfolio(scores, {"date": "2026-01-02"})

    assert len(portfolio) == 1
    assert portfolio.iloc[0]["symbol"] == "AAPL"
    assert portfolio["target_weight"].sum() == 1.0
