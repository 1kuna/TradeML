"""Reference data package (calendars, corp_actions, delistings, universe).

Provides utilities such as:
- get_calendar, get_trading_days in calendars.py
"""

# Expose common utilities for convenience imports
try:
    from .calendars import get_calendar, get_trading_days  # noqa: F401
except Exception:
    # Avoid import-time failures if optional deps for calendars are missing
    pass

