"""Exchange calendar helpers backed by ``exchange_calendars``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd


def _to_timestamp(value: str | date_type | pd.Timestamp) -> pd.Timestamp:
    """Normalize incoming date-like values to a midnight timestamp."""
    return pd.Timestamp(value).normalize()


@dataclass(slots=True)
class ExchangeCalendarStore:
    """Build and persist exchange session calendars."""

    root: Path

    def build_calendar_frame(
        self,
        exchange: str,
        start: str | date_type | pd.Timestamp,
        end: str | date_type | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a dataframe covering all dates in the requested range."""
        calendar = xcals.get_calendar(exchange)
        start_ts = _to_timestamp(start)
        end_ts = _to_timestamp(end)
        all_days = pd.date_range(start_ts, end_ts, freq="D")
        session_index = calendar.sessions_in_range(start_ts, end_ts)

        records: list[dict[str, object]] = []
        for current_day in all_days:
            is_session = current_day in session_index
            if is_session:
                market_open = calendar.session_open(current_day)
                market_close = calendar.session_close(current_day)
                session_length = market_close - market_open
            else:
                market_open = pd.NaT
                market_close = pd.NaT
                session_length = pd.Timedelta(0)

            records.append(
                {
                    "date": current_day.date(),
                    "market_open": market_open,
                    "market_close": market_close,
                    "is_early_close": is_session and session_length < pd.Timedelta(hours=6, minutes=30),
                    "is_holiday": (not is_session) and current_day.dayofweek < 5,
                }
            )

        return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)

    def write_calendar_parquet(
        self,
        exchange: str,
        start: str | date_type | pd.Timestamp,
        end: str | date_type | pd.Timestamp,
    ) -> Path:
        """Generate and write the calendar parquet for an exchange."""
        output_path = self.root / f"{exchange}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.build_calendar_frame(exchange=exchange, start=start, end=end).to_parquet(output_path, index=False)
        return output_path


def get_trading_days(
    exchange: str,
    start: str | date_type | pd.Timestamp,
    end: str | date_type | pd.Timestamp,
) -> list[date_type]:
    """Return the exchange trading sessions in the given range."""
    calendar = xcals.get_calendar(exchange)
    return [session.date() for session in calendar.sessions_in_range(_to_timestamp(start), _to_timestamp(end))]


def is_trading_day(exchange: str, day: str | date_type | pd.Timestamp) -> bool:
    """Return whether the supplied day is a valid trading session."""
    calendar = xcals.get_calendar(exchange)
    return bool(calendar.is_session(_to_timestamp(day)))


def is_early_close(exchange: str, day: str | date_type | pd.Timestamp) -> bool:
    """Return whether the supplied trading session closes early."""
    normalized_day = _to_timestamp(day)
    if not is_trading_day(exchange, normalized_day):
        return False
    calendar = xcals.get_calendar(exchange)
    return (calendar.session_close(normalized_day) - calendar.session_open(normalized_day)) < pd.Timedelta(
        hours=6, minutes=30
    )
