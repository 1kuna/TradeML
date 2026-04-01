from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.calendars.exchange import ExchangeCalendarStore, get_trading_days, is_early_close, is_trading_day


def test_known_holidays_and_weekends() -> None:
    trading_days = get_trading_days("XNYS", "2025-07-03", "2025-07-07")

    assert pd.Timestamp("2025-07-04").date() not in trading_days
    assert pd.Timestamp("2025-07-05").date() not in trading_days
    assert pd.Timestamp("2025-07-06").date() not in trading_days
    assert is_trading_day("XNYS", "2025-07-03")
    assert not is_trading_day("XNYS", "2025-12-25")


def test_early_close_and_dst_handling(tmp_path: Path) -> None:
    store = ExchangeCalendarStore(root=tmp_path)
    output = store.write_calendar_parquet("XNYS", "2025-03-07", "2025-11-28")
    frame = pd.read_parquet(output)

    march_pre_dst = frame.loc[frame["date"] == pd.Timestamp("2025-03-07").date()].iloc[0]
    march_post_dst = frame.loc[frame["date"] == pd.Timestamp("2025-03-10").date()].iloc[0]
    thanksgiving_friday = frame.loc[frame["date"] == pd.Timestamp("2025-11-28").date()].iloc[0]

    assert march_pre_dst["market_open"].hour == 14
    assert march_post_dst["market_open"].hour == 13
    assert thanksgiving_friday["is_early_close"]
    assert is_early_close("XNYS", "2025-11-28")


def test_holiday_flags_are_persisted(tmp_path: Path) -> None:
    store = ExchangeCalendarStore(root=tmp_path)
    frame = store.build_calendar_frame("XNYS", "2025-11-26", "2025-11-30")

    thanksgiving = frame.loc[frame["date"] == pd.Timestamp("2025-11-27").date()].iloc[0]
    saturday = frame.loc[frame["date"] == pd.Timestamp("2025-11-29").date()].iloc[0]

    assert thanksgiving["is_holiday"]
    assert pd.isna(thanksgiving["market_open"])
    assert not saturday["is_holiday"]
