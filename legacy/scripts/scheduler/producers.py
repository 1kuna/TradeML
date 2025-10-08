from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Iterator, Optional, List

from ops.ssot.router import route_dataset


def _provider_allowed(provider: str, dataset: str, want_date, universe: List[str]) -> bool:
    try:
        order = route_dataset(dataset, want_date, universe)
    except Exception:
        return True
    if not order:
        return True
    return order[0] == provider


def _today_date():
    return datetime.now().date()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _start_date(edge, vendor: str, table: str, default_days: int):
    """Resolve start date using bookmark when present, else default_days back."""
    bm = edge.bookmarks.get_last_timestamp(vendor, table) if edge.bookmarks else None
    today = _today_date()
    if bm:
        try:
            return (datetime.fromisoformat(bm).date() + timedelta(days=1))
        except Exception:
            pass
    return today - timedelta(days=default_days)


def alpaca_bars_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    if not _provider_allowed("finnhub", "options_chain", today, symbols):
        return
    # Default: 15 years of daily bars (SSOT green threshold ~15y)
    default_days = _int_env("ALPACA_DAY_START_DAYS", 365 * 15)
    start_date = _start_date(edge, "alpaca", "equities_bars", default_days)
    try:
        if start_date == today and not edge._should_fetch_eod_for_day("alpaca", today):
            prev = today - timedelta(days=1)
            last_ts = edge.bookmarks.get_last_timestamp("alpaca", "equities_bars") if edge.bookmarks else None
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
        if not _provider_allowed("alpaca", "equities_eod", d, syms):
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
    # Default: 2 years of minute bars (free-tier practical window per Playbook)
    default_days = _int_env("ALPACA_MINUTE_START_DAYS", 365 * 2)
    start_date = _start_date(edge, "alpaca", "equities_bars_minute", default_days)
    from math import ceil
    alp_bad = set([s.upper() for s in edge._vendor_bad_symbols.get("alpaca", set())]) | edge.bad_symbols.vendor_set("alpaca")
    syms = [s for s in symbols if s.upper() not in alp_bad]
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        if not _provider_allowed("alpaca", "equities_minute", d, syms):
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
    # Default: 10 years of daily bars
    default_days = _int_env("POLYGON_DAY_START_DAYS", 365 * 10)
    start_date = _start_date(edge, "polygon", "equities_bars", default_days)
    try:
        if start_date == today and not edge._should_fetch_eod_for_day("polygon", today):
            prev = today - timedelta(days=1)
            last_ts = edge.bookmarks.get_last_timestamp("polygon", "equities_bars") if edge.bookmarks else None
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
        if not _provider_allowed("polygon", "equities_eod", d, symbols):
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
    # Default: 50 years of macro history (effectively full modern range)
    default_days = _int_env("FRED_TREASURY_START_DAYS", 365 * 50)
    start_date = _start_date(edge, "fred", "macro_treasury", default_days)
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


def alpaca_options_bars_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    # Default: collect the past 7 days unless bookmark exists
    default_days = _int_env("ALPACA_OPTIONS_BARS_START_DAYS", 7)
    start_date = _start_date(edge, "alpaca", "options_bars", default_days)
    try:
        ul_per_unit = max(1, int(os.getenv("NODE_ALPACA_OPTIONS_UL_PER_UNIT", "3")))
    except Exception:
        ul_per_unit = 3
    try:
        timeframe = os.getenv("ALPACA_OPTIONS_TIMEFRAME", "1Day")
    except Exception:
        timeframe = "1Day"
    picked = 0
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        if not _provider_allowed("alpaca", "options_bars", d, symbols):
            d += timedelta(days=1)
            continue
        picked = 0
        for sym in symbols:
            if picked >= ul_per_unit:
                break
            picked += 1
            run_fn = (lambda ul=sym, day=d, tf=timeframe: edge._run_alpaca_options_bars_for_underlier(ul, day, tf, budget_mgr))
            yield {
                "vendor": "alpaca",
                "desc": f"alpaca options bars {sym} {d}",
                "tokens": 1,
                "run": run_fn,
            }
        d += timedelta(days=1)


def alpaca_options_chain_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    today = _today_date()
    default_days = _int_env("ALPACA_OPTIONS_CHAIN_START_DAYS", 7)
    start_date = _start_date(edge, "alpaca", "options_chain", default_days)
    try:
        ul_per_unit = max(1, int(os.getenv("NODE_ALPACA_OPTIONS_UL_PER_UNIT", "3")))
    except Exception:
        ul_per_unit = 3
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        if not _provider_allowed("alpaca", "options_chain", d, symbols):
            d += timedelta(days=1)
            continue
        picked = 0
        for sym in symbols:
            if picked >= ul_per_unit:
                break
            picked += 1
            yield {
                "vendor": "alpaca",
                "desc": f"alpaca options chain {sym} {d}",
                "tokens": 1,
                "run": (lambda ul=sym, day=d: edge._run_alpaca_options_chain_underlier(ul, day, budget_mgr)),
            }
        d += timedelta(days=1)


def alpaca_corporate_actions_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "alpaca" not in edge.connectors:
        return
    today = _today_date()
    default_days = _int_env("ALPACA_CORPACTIONS_START_DAYS", 5475)
    start_date = _start_date(edge, "alpaca", "corporate_actions", default_days)
    d = start_date
    while d <= today and not edge.shutdown_requested:
        # Corporate actions are day-stamped; no EOD gating needed, but reuse alpaca gating
        if not edge._should_fetch_eod_for_day("alpaca", d):
            d += timedelta(days=1)
            continue
        if not _provider_allowed("alpaca", "corp_actions", d, [""]):
            d += timedelta(days=1)
            continue
        yield {
            "vendor": "alpaca",
            "desc": f"alpaca corporate actions {d}",
            "tokens": 1,
            "run": (lambda day=d: edge._run_alpaca_corporate_actions_day(day, budget_mgr)),
        }
        d += timedelta(days=1)


def finnhub_daily_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "finnhub" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    if not symbols:
        return
    today = _today_date()
    default_days = _int_env("FINNHUB_DAILY_START_DAYS", 365)
    start_date = _start_date(edge, "finnhub", "equities_eod_fn", default_days)
    chunk = max(1, int(os.getenv("NODE_FINNHUB_DAILY_SYMBOLS_PER_UNIT", "5")))
    vendor_key = "finnhub-daily"
    d = start_date
    while d <= today and not edge.shutdown_requested:
        if not edge._should_fetch_eod_for_day("finnhub", d):
            d += timedelta(days=1)
            continue
        if not _provider_allowed("finnhub", "equities_eod", d, symbols):
            d += timedelta(days=1)
            continue
        chunk_syms = edge._rotate_symbols(vendor_key, symbols, chunk) if hasattr(edge, "_rotate_symbols") else symbols[:chunk]
        if not chunk_syms:
            break
        yield {
            "vendor": "finnhub",
            "desc": f"finnhub daily {d} ({len(chunk_syms)} syms)",
            "tokens": len(chunk_syms),
            "run": (lambda day=d, syms=list(chunk_syms): edge._run_finnhub_daily_day(syms, day, budget_mgr)),
        }
        d += timedelta(days=1)


def av_corp_actions_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "av" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    if not symbols:
        return
    today = _today_date()
    if not _provider_allowed("av", "corp_actions", today, symbols):
        return
    per_unit = max(1, int(os.getenv("NODE_AV_CORPACTIONS_PER_UNIT", "5")))
    chunk_syms = edge._rotate_symbols("av-corp-actions", symbols, per_unit) if hasattr(edge, "_rotate_symbols") else symbols[:per_unit]
    if not chunk_syms:
        return
    yield {
        "vendor": "av",
        "desc": f"av corp actions {','.join(chunk_syms[:2])}{'…' if len(chunk_syms) > 2 else ''}",
        "tokens": len(chunk_syms),
        "run": (lambda syms=list(chunk_syms): edge._run_alpha_vantage_corp_actions(syms, budget_mgr)),
    }


def av_options_hist_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "av" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    if not symbols:
        return
    today = _today_date()
    if not _provider_allowed("av", "options_chain_hist", today, symbols):
        return
    per_unit = max(1, int(os.getenv("NODE_AV_OPTIONS_PER_UNIT", "1")))
    chunk_syms = edge._rotate_symbols("av-options-hist", symbols, per_unit) if hasattr(edge, "_rotate_symbols") else symbols[:per_unit]
    for sym in chunk_syms:
        yield {
            "vendor": "av",
            "desc": f"av options hist {sym}",
            "tokens": 1,
            "run": (lambda s=sym: edge._run_alpha_vantage_options_hist(s, expiry=None, budget=budget_mgr)),
        }


def fmp_fundamentals_units(edge, budget_mgr=None) -> Iterator[dict]:
    if "fmp" not in edge.connectors:
        return
    symbols = edge._symbols_universe()
    if not symbols:
        return
    today = _today_date()
    if not _provider_allowed("fmp", "fundamentals", today, symbols):
        return
    per_unit = max(1, int(os.getenv("NODE_FMP_FUNDS_PER_UNIT", "1")))
    period = os.getenv("FMP_FUNDS_PERIOD", "annual")
    chunk_syms = edge._rotate_symbols("fmp-fundamentals", symbols, per_unit) if hasattr(edge, "_rotate_symbols") else symbols[:per_unit]
    for sym in chunk_syms:
        yield {
            "vendor": "fmp",
            "desc": f"fmp fundamentals {sym} ({period})",
            "tokens": 3,
            "run": (lambda s=sym, per=period: edge._run_fmp_fundamentals(s, per, budget_mgr)),
        }
