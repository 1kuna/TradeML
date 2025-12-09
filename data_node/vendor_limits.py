"""
Vendor capability limits shared across the data-node.

Currently contains Massive.com free-tier window helpers so both leasing and
dispatch use the same bounds.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple

# Massive free-tier constraints
MASSIVE_HISTORY_DAYS = 730  # ~2 years back
MASSIVE_DELAY_DAYS = 1      # T+1 delay on free tier


def massive_window(today: date | None = None) -> Tuple[date, date]:
    """Return (earliest, latest) inclusive window for Massive free tier."""
    today = today or date.today()
    earliest = today - timedelta(days=MASSIVE_HISTORY_DAYS)
    latest = today - timedelta(days=MASSIVE_DELAY_DAYS)
    return earliest, latest


def _ensure_date(val) -> date:
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return date.fromisoformat(val)
    raise TypeError(f"Expected date or str, got {type(val).__name__}")


def massive_in_window(start_date, end_date, today: date | None = None) -> bool:
    """Check whether a date range is serviceable by Massive free tier."""
    try:
        start_dt = _ensure_date(start_date)
        end_dt = _ensure_date(end_date)
    except Exception:
        return False

    earliest, latest = massive_window(today)

    if end_dt < earliest or start_dt < earliest:
        return False
    if start_dt > latest:
        return False
    return True
