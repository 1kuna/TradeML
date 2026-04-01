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


def test_label_horizon_uses_exchange_sessions_not_calendar_days() -> None:
    dates = pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07"])
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["AAPL"] * 4 + ["MSFT"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0, 50.0, 51.0, 52.0, 53.0],
        }
    )

    labels = build_labels(panel, horizon=1)
    aapl = labels.loc[labels["symbol"] == "AAPL"].reset_index(drop=True)

    assert aapl.loc[1, "date"] == pd.Timestamp("2026-07-03")
    assert aapl.loc[1, "raw_forward_return_1d"] == np.log(102.0 / 101.0)
