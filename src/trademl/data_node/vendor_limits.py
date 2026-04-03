"""Docs-backed default vendor request budgets for the data node."""

from __future__ import annotations

from copy import deepcopy


# These defaults intentionally run at 95% of the publicly documented theoretical
# caps to leave a small safety buffer. For vendors that publish only daily or
# second-level throttles, the minute budget is derived from the documented limit
# and rounded down to an integer because BudgetManager operates on integer units.
#
# Source notes:
# - alpaca: 200 historical API calls / min on the Basic plan
# - tiingo: 10,000 requests / hour and 100,000 requests / day
# - twelve_data: 8 API credits / min and 800 / day on the free plan
# - massive: 5 API calls / min on Stocks Basic
# - finnhub: 60 API calls / min on the free plan
# - alpha_vantage: 25 requests / day on the free plan; no current free-tier
#   minute cap is published, so the minute budget is the minimum integer floor
#   supported by the local budget manager
# - fred: up to 2 requests / second before 429 throttling
# - fmp: 250 requests / day on the free plan; no current free-tier minute cap is
#   published, so the minute budget is the minimum integer floor
# - sec_edgar: 10 requests / second under SEC fair-access guidance
DEFAULT_VENDOR_LIMITS: dict[str, dict[str, int]] = {
    "alpaca": {"rpm": 190, "daily_cap": 273600},
    "tiingo": {"rpm": 158, "daily_cap": 95000},
    "twelve_data": {"rpm": 7, "daily_cap": 760},
    "massive": {"rpm": 4, "daily_cap": 6840},
    "finnhub": {"rpm": 57, "daily_cap": 82080},
    "alpha_vantage": {"rpm": 1, "daily_cap": 23},
    "fred": {"rpm": 114, "daily_cap": 164160},
    "fmp": {"rpm": 1, "daily_cap": 237},
    "sec_edgar": {"rpm": 570, "daily_cap": 820800},
}


def default_vendor_limits() -> dict[str, dict[str, int]]:
    """Return a defensive copy of the researched default vendor limits."""

    return deepcopy(DEFAULT_VENDOR_LIMITS)
