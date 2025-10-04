from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Iterator, Optional


def _today_date():
    return datetime.now().date()


def alpaca_bars_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    last_ts = edge.bookmarks.get_last_timestamp("alpaca", "equities_bars") if edge.bookmarks else None
    start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=30))
    try:
        if start_date == today and not edge._should_fetch_eod_for_day("alpaca", today):
            prev = today - timedelta(days=1)
            if (not last_ts) or (datetime.fromisoformat(last_ts).date() < prev):
                start_date = prev
            else:
                return
    except Exception:
        pass
    d = start_date
    from math import ceil
    alp_bad = set([s.upper() for s in edge._vendor_bad_symbols.get("alpaca", set())]) | edge.bad_symbols.vendor_set("alpaca")
    syms = [s for s in symbols if s.upper() not in alp_bad]
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        tokens = max(1, ceil(len(syms) / 100))
        yield {
            "vendor": "alpaca",
            "desc": f"alpaca {d}",
            "tokens": tokens,
            "run": (lambda day=d, syms=syms: edge._run_alpaca_day(syms, day, budget_mgr)),
        }
        d += timedelta(days=1)


def alpaca_minute_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    last_ts = edge.bookmarks.get_last_timestamp("alpaca", "equities_bars_minute") if edge.bookmarks else None
    start_days = 7
    try:
        start_days = int(os.getenv("ALPACA_MINUTE_START_DAYS", "7"))
    except Exception:
        start_days = 7
    start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=start_days))
    from math import ceil
    alp_bad = set([s.upper() for s in edge._vendor_bad_symbols.get("alpaca", set())]) | edge.bad_symbols.vendor_set("alpaca")
    syms = [s for s in symbols if s.upper() not in alp_bad]
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        tokens = max(1, ceil(len(syms) / 100))
        yield {
            "vendor": "alpaca",
            "desc": f"alpaca-minute {d}",
            "tokens": tokens,
            "run": (lambda day=d, syms=syms: edge._run_alpaca_minute_day(syms, day, budget_mgr)),
        }
        d += timedelta(days=1)


def polygon_bars_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "polygon" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    last_ts = edge.bookmarks.get_last_timestamp("polygon", "equities_bars") if edge.bookmarks else None
    start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=7))
    try:
        if start_date == today and not edge._should_fetch_eod_for_day("polygon", today):
            prev = today - timedelta(days=1)
            if (not last_ts) or (datetime.fromisoformat(last_ts).date() < prev):
                start_date = prev
            else:
                return
    except Exception:
        pass
    try:
        BATCH = max(1, int(os.getenv("NODE_POLYGON_SYMBOLS_PER_UNIT", "3")))
    except Exception:
        BATCH = 3
    from itertools import islice
    pg_bad = set([s.upper() for s in edge._vendor_bad_symbols.get("polygon", set())]) | edge.bad_symbols.vendor_set("polygon")
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("polygon", d):
            d += timedelta(days=1)
            continue
        val_syms = [s for s in symbols if s.upper() not in pg_bad]
        for i in range(0, len(val_syms), BATCH):
            chunk = val_syms[i:i + BATCH]
            tokens = len(chunk)
            yield {
                "vendor": "polygon",
                "desc": f"polygon {d} [{i}:{i+len(chunk)}]",
                "tokens": tokens,
                "run": (lambda day=d, syms=chunk: edge._run_polygon_day(syms, day, budget_mgr)),
            }
        d += timedelta(days=1)


def finnhub_options_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "finnhub" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    try:
        per_run = max(1, int(os.getenv("NODE_FINNHUB_UL_PER_UNIT", "5")))
    except Exception:
        per_run = 5
    fh_bad = set([s.upper() for s in edge._vendor_bad_symbols.get("finnhub", set())]) | edge.bad_symbols.vendor_set("finnhub")
    picked = 0
    for sym in symbols:
        if picked >= per_run or edge.shutdown_requested:
            break
        if sym.upper() in fh_bad:
            continue
        picked += 1
        yield {
            "vendor": "finnhub",
            "desc": f"finnhub options {sym}",
            "tokens": 1,
            "run": (lambda s=sym: edge._run_finnhub_options_underlier(s, today, budget_mgr)),
        }


def fred_treasury_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "fred" not in edge.connectors:
        return
    today = _today_date()
    last_ts = edge.bookmarks.get_last_timestamp("fred", "macro_treasury") if edge.bookmarks else None
    start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=7))
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("fred", d):
            d += timedelta(days=1)
            continue
        yield {
            "vendor": "fred",
            "desc": f"fred treasury {d}",
            "tokens": 1,
            "run": (lambda day=d: edge._run_fred_treasury_day(day, budget_mgr)),
        }
        d += timedelta(days=1)

