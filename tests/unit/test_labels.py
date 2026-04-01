from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.labels.returns import build_labels


def _label_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=12)
    rows = []
    for idx, symbol in enumerate(["AAPL", "MSFT", "NVDA"]):
        close = 100 + idx * 10 + np.arange(len(dates))
        for date, value in zip(dates, close, strict=False):
            rows.append({"date": date, "symbol": symbol, "close": float(value)})
    return pd.DataFrame(rows)


def test_universe_relative_labels_sum_to_zero() -> None:
    labels = build_labels(_label_panel(), horizon=5)
    per_date = labels.groupby("date")["label_5d"].sum().dropna()
    assert (per_date.abs() < 1e-10).all()


def test_label_horizon_uses_trading_day_order() -> None:
    labels = build_labels(_label_panel(), horizon=5)
    aapl = labels.loc[labels["symbol"] == "AAPL"].reset_index(drop=True)
    expected = np.log((100 + 5) / 100)
    assert aapl.loc[0, "raw_forward_return_5d"] == expected
