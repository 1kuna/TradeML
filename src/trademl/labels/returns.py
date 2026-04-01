"""Universe-relative forward return labels."""

from __future__ import annotations

import exchange_calendars as xcals
import numpy as np
import pandas as pd


def build_labels(panel: pd.DataFrame, horizon: int = 5, *, exchange: str = "XNYS") -> pd.DataFrame:
    """Build raw and universe-relative forward returns using trading-calendar horizons."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    calendar = xcals.get_calendar(exchange)
    sessions = pd.Index(
        calendar.sessions_in_range(
            frame["date"].min().normalize(),
            (frame["date"].max() + pd.tseries.offsets.BDay(horizon + 5)).normalize(),
        )
    )
    session_positions = {session: idx for idx, session in enumerate(sessions)}
    observed_dates = pd.Index(sorted(frame["date"].dt.normalize().unique()))
    observed_positions = {observed_date: idx for idx, observed_date in enumerate(observed_dates)}

    base_dates = frame["date"].dt.normalize()
    target_dates = []
    for current_date in base_dates:
        position = session_positions.get(current_date)
        if position is not None and position + horizon < len(sessions):
            target_dates.append(sessions[position + horizon].tz_localize(None))
            continue
        observed_position = observed_positions.get(current_date)
        if observed_position is None or observed_position + horizon >= len(observed_dates):
            target_dates.append(pd.NaT)
            continue
        target_dates.append(observed_dates[observed_position + horizon])
    frame["target_date"] = target_dates

    target_prices = frame[["symbol", "date", "close"]].rename(columns={"date": "target_date", "close": "target_close"})
    frame = frame.merge(target_prices, on=["symbol", "target_date"], how="left")
    frame[f"raw_forward_return_{horizon}d"] = np.log(frame["target_close"] / frame["close"])
    universe_mean = frame.groupby("date")[f"raw_forward_return_{horizon}d"].transform("mean")
    frame[f"label_{horizon}d"] = frame[f"raw_forward_return_{horizon}d"] - universe_mean
    return frame[["date", "symbol", f"raw_forward_return_{horizon}d", f"label_{horizon}d"]]
