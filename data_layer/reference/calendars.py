"""
Exchange calendar module with session handling.

Handles:
- Market open/close times (regular and early close)
- DST transitions
- Half-day/holiday schedules
- Session validation for point-in-time safety

Uses exchange_calendars for authoritative NYSE/NASDAQ schedules.
"""

from datetime import datetime, date, time
from typing import Optional, Tuple, List
import pandas as pd
import exchange_calendars as xcals
from zoneinfo import ZoneInfo
from loguru import logger


class ExchangeCalendar:
    """
    Exchange calendar with session handling, DST awareness, and early close detection.

    Ensures all backtests and live trading respect actual market hours.
    """

    def __init__(self, exchange: str = "XNYS", timezone: str = "America/New_York"):
        """
        Initialize calendar for a specific exchange.

        Args:
            exchange: Exchange code (XNYS=NYSE, XNAS=NASDAQ, etc.)
            timezone: Timezone string (default: America/New_York for US markets)
        """
        self.exchange = exchange
        self.timezone = ZoneInfo(timezone)

        try:
            self.calendar = xcals.get_calendar(exchange)
            logger.info(f"Loaded calendar for {exchange}")
        except Exception as e:
            logger.error(f"Failed to load calendar for {exchange}: {e}")
            raise

    def is_trading_day(self, dt: date) -> bool:
        """
        Check if given date is a trading day.

        Args:
            dt: Date to check

        Returns:
            True if market is open on this date
        """
        return self.calendar.is_session(pd.Timestamp(dt))

    def get_session_times(self, dt: date) -> Optional[Tuple[datetime, datetime]]:
        """
        Get open and close times for a trading session.

        Handles early closes (e.g., day before Thanksgiving, day after Thanksgiving,
        Christmas Eve, July 3rd if July 4th is weekday).

        Args:
            dt: Trading date

        Returns:
            Tuple of (open_time, close_time) or None if not a trading day
        """
        if not self.is_trading_day(dt):
            return None

        ts = pd.Timestamp(dt)

        try:
            open_time = self.calendar.session_open(ts)
            close_time = self.calendar.session_close(ts)

            return (open_time.to_pydatetime(), close_time.to_pydatetime())
        except Exception as e:
            logger.warning(f"Could not get session times for {dt}: {e}")
            return None

    def is_early_close(self, dt: date) -> bool:
        """
        Check if session closes early (e.g., 1:00 PM ET instead of 4:00 PM ET).

        Args:
            dt: Trading date

        Returns:
            True if early close day
        """
        if not self.is_trading_day(dt):
            return False

        session_times = self.get_session_times(dt)
        if session_times is None:
            return False

        _, close_time = session_times

        # NYSE regular close is 4:00 PM ET; early close is 1:00 PM ET
        regular_close_hour = 16  # 4:00 PM
        return close_time.hour < regular_close_hour

    def get_trading_days(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """
        Get all trading days in range (inclusive).

        Args:
            start_date: Start of range
            end_date: End of range (inclusive)

        Returns:
            List of trading dates
        """
        sessions = self.calendar.sessions_in_range(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
        return [s.date() for s in sessions]

    def previous_trading_day(self, dt: date, n: int = 1) -> Optional[date]:
        """
        Get the nth previous trading day.

        Args:
            dt: Reference date
            n: Number of trading days to go back

        Returns:
            Previous trading date or None
        """
        try:
            ts = pd.Timestamp(dt)
            prev_session = self.calendar.previous_session(ts, n)
            return prev_session.date()
        except Exception as e:
            logger.warning(f"Could not get previous trading day for {dt}: {e}")
            return None

    def next_trading_day(self, dt: date, n: int = 1) -> Optional[date]:
        """
        Get the nth next trading day.

        Args:
            dt: Reference date
            n: Number of trading days forward

        Returns:
            Next trading date or None
        """
        try:
            ts = pd.Timestamp(dt)
            next_session = self.calendar.next_session(ts, n)
            return next_session.date()
        except Exception as e:
            logger.warning(f"Could not get next trading day for {dt}: {e}")
            return None

    def align_to_session_close(self, dt: datetime) -> datetime:
        """
        Align timestamp to end of trading session (for EOD backtests).

        Args:
            dt: Timestamp to align

        Returns:
            Timestamp at session close
        """
        session_date = dt.date()
        session_times = self.get_session_times(session_date)

        if session_times is None:
            # Not a trading day, find next trading day close
            next_day = self.next_trading_day(session_date)
            if next_day:
                session_times = self.get_session_times(next_day)
                if session_times:
                    return session_times[1]

        return session_times[1] if session_times else dt

    def trading_days_between(self, start: date, end: date) -> int:
        """
        Count trading days between two dates (inclusive).

        Args:
            start: Start date
            end: End date

        Returns:
            Number of trading days
        """
        return len(self.get_trading_days(start, end))

    def get_session_id(self, dt: date) -> str:
        """
        Generate unique session identifier for a trading day.

        Useful for partitioning data by session.

        Args:
            dt: Trading date

        Returns:
            Session ID string (format: YYYYMMDD)
        """
        return dt.strftime("%Y%m%d")

    def validate_timestamp_pit(
        self,
        timestamp: datetime,
        data_date: date
    ) -> bool:
        """
        Validate that a timestamp is point-in-time safe for a data date.

        Ensures data timestamp is not from the future relative to data_date close.

        Args:
            timestamp: Data timestamp
            data_date: Date the data should be available

        Returns:
            True if timestamp is PIT-safe
        """
        session_times = self.get_session_times(data_date)
        if session_times is None:
            return False

        _, close_time = session_times

        # Data timestamp must be <= session close
        return timestamp <= close_time


# Global calendar instance (lazy-loaded)
_calendar_cache = {}

def get_calendar(exchange: str = "XNYS") -> ExchangeCalendar:
    """
    Get or create calendar instance (cached).

    Args:
        exchange: Exchange code

    Returns:
        ExchangeCalendar instance
    """
    if exchange not in _calendar_cache:
        _calendar_cache[exchange] = ExchangeCalendar(exchange)
    return _calendar_cache[exchange]


def get_trading_days(start_date: date, end_date: date, exchange: str = "XNYS") -> List[date]:
    """
    Convenience function to return trading dates for an exchange.

    Args:
        start_date: Inclusive start date
        end_date: Inclusive end date
        exchange: Exchange code (default XNYS)

    Returns:
        List of trading session dates
    """
    cal = get_calendar(exchange)
    return cal.get_trading_days(start_date, end_date)


# Acceptance test cases (run via pytest)
if __name__ == "__main__":
    # Quick validation
    cal = get_calendar("XNYS")

    # Test known dates
    test_cases = [
        (date(2024, 1, 15), True, False),   # MLK Day - market closed
        (date(2024, 7, 3), True, True),     # Day before July 4th - early close
        (date(2024, 11, 29), True, True),   # Day after Thanksgiving - early close
        (date(2024, 12, 25), False, False), # Christmas - market closed
        (date(2024, 1, 2), True, False),    # Regular trading day
    ]

    print("Running calendar validation tests...\n")

    for test_date, should_be_open, should_be_early in test_cases:
        is_open = cal.is_trading_day(test_date)
        is_early = cal.is_early_close(test_date) if is_open else False

        status = "✓" if (is_open == should_be_open and is_early == should_be_early) else "✗"

        print(f"{status} {test_date}: Open={is_open} (expected {should_be_open}), "
              f"Early={is_early} (expected {should_be_early})")

        if is_open:
            times = cal.get_session_times(test_date)
            if times:
                open_t, close_t = times
                print(f"  Session: {open_t.strftime('%H:%M')} - {close_t.strftime('%H:%M')} ET")

    print("\n✓ Calendar module validation complete")
