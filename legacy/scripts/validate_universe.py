#!/usr/bin/env python
"""
Universe pre-prune validator.

Inputs:
  - CSV with a 'Ticker' column (default: IWB_holdings.csv)
  - .env for vendor keys

Process:
  1) Load tickers from input CSV.
  2) FMP bulk coverage: stock/list → mark fmp_ok for symbols present (cheap, single call).
  3) Alpaca day check: previous business day bars in 100-symbol batches → mark alpaca_ok.
  4) Compose table and write:
       data_layer/reference/universe_validity.parquet
       data_layer/reference/universe_symbols_valid.txt (equities_ok = alpaca_ok OR fmp_ok)
     Optionally update universe_symbols.txt when --update-universe is set.

Notes:
  - Polygon and Finnhub validation are kept for in-node incremental validation to avoid heavy preflight.
  - Prev business day skips weekend only (simple heuristic).
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import List, Set

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from data_layer.connectors.fmp_connector import FMPConnector
from data_layer.connectors.alpaca_connector import AlpacaConnector
import os
import time
import requests


def _prev_business_day(d: date) -> date:
    # Weekend-only skip; holidays will just be empty (acceptable for coarse validation)
    dd = d - timedelta(days=1)
    while dd.weekday() >= 5:  # 5=Sat,6=Sun
        dd -= timedelta(days=1)
    return dd


def load_tickers_from_csv(path: Path) -> List[str]:
    # Robustly locate header row containing Ticker/Symbol and read from there
    header_idx = None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("ticker,") or \
               ("ticker" in line.lower() and "," in line):
                header_idx = i
                break
    if header_idx is None:
        # Fallback: try naive read
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", errors="ignore")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=header_idx, header=0)

    col = None
    for c in df.columns:
        if str(c).strip().lower() in ("ticker", "symbol"):
            col = c
            break
    if not col:
        raise RuntimeError("Input CSV missing a Ticker/Symbol column")
    tickers = [str(t).strip().upper() for t in df[col].astype(str).tolist() if str(t).strip()]
    # Basic cleanup: remove likely non-tickers
    tickers = [t for t in tickers if t not in ("nan", "-", "")]  # remove empties
    # Deduplicate preserving order
    seen = set()
    ordered: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def fmp_bulk_symbols() -> Set[str]:
    try:
        fmp = FMPConnector()
        df = fmp.fetch_available_symbols(exchange="NYSE,NASDAQ")
        if df.empty:
            return set()
        return set(df["symbol"].astype(str).str.upper().tolist())
    except Exception as e:
        logger.warning(f"FMP bulk symbols failed: {e}")
        return set()


def alpaca_day_presence(symbols: List[str], day: date) -> Set[str]:
    try:
        alp = AlpacaConnector()
    except Exception as e:
        logger.warning(f"Alpaca init failed: {e}")
        return set()
    try:
        df = alp.fetch_bars(symbols=symbols, start_date=day, end_date=day, timeframe="1Day")
        if df.empty:
            return set()
        return set(df["symbol"].astype(str).str.upper().unique().tolist())
    except Exception as e:
        logger.warning(f"Alpaca day check failed: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description="Pre-prune universe against vendor coverage")
    parser.add_argument("--input", default="IWB_holdings.csv", help="Path to CSV with Ticker column")
    parser.add_argument("--update-universe", action="store_true", help="Overwrite data_layer/reference/universe_symbols.txt with pruned list")
    parser.add_argument("--reverse", action="store_true", help="Process tickers bottom-up for per-symbol checks")
    parser.add_argument("--fmp-per-symbol", action="store_true", help="Validate FMP coverage via per-symbol company profile")
    parser.add_argument("--fmp-max", type=int, default=250, help="Max symbols to check with FMP per-symbol (respects free tier)")
    parser.add_argument("--fmp-stable-search", action="store_true", help="Use FMP stable search endpoint per symbol (preferred on some plans)")
    parser.add_argument("--fmp-delay-ms", type=int, default=150, help="Delay between FMP per-symbol requests (milliseconds)")
    args = parser.parse_args()

    load_dotenv()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    tickers = load_tickers_from_csv(inp)
    logger.info(f"Loaded {len(tickers)} tickers from {inp}")

    # FMP coverage
    fmp_ok: Set[str] = set()
    fmp_set = fmp_bulk_symbols()
    if fmp_set:
        fmp_ok = {t for t in tickers if t in fmp_set}
        logger.info(f"FMP coverage (bulk): {len(fmp_ok)} / {len(tickers)}")
    # Optional per-symbol pass (bottom-up) with budget cap
    if args.fmp_per_symbol or args.fmp_stable_search:
        try:
            order = list(reversed(tickers)) if args.reverse else tickers
            checked = 0
            for sym in order:
                if checked >= max(0, int(args.fmp_max)):
                    break
                if sym in fmp_ok:
                    continue
                ok = False
                if args.fmp_stable_search:
                    # Use stable search endpoint directly
                    api_key = os.getenv("FMP_API_KEY")
                    if not api_key:
                        raise RuntimeError("FMP_API_KEY missing for stable search")
                    url = f"https://financialmodelingprep.com/stable/search-symbol"
                    try:
                        r = requests.get(url, params={"query": sym, "apikey": api_key}, timeout=20)
                        if r.status_code == 200:
                            arr = r.json() if r.headers.get('Content-Type','').startswith('application/json') else []
                            # True if exact symbol match appears in results
                            ok = any((isinstance(x, dict) and str(x.get('symbol','')).upper() == sym) for x in (arr or []))
                        elif r.status_code in (401, 403):
                            logger.warning(f"FMP stable search auth error for {sym}: {r.status_code}")
                            ok = False
                        elif r.status_code == 429:
                            logger.warning("FMP stable search rate-limited; stopping further checks")
                            break
                    except Exception as e:
                        logger.warning(f"FMP stable search failed for {sym}: {e}")
                    # pacing
                    time.sleep(max(0, args.fmp_delay_ms) / 1000.0)
                else:
                    # Fallback: company profile via connector
                    try:
                        fmp = FMPConnector()
                        prof = fmp.fetch_company_profile(sym)
                        ok = bool(prof.get("symbol") or prof)
                    except Exception:
                        ok = False
                if ok:
                    fmp_ok.add(sym)
                checked += 1
            logger.info(f"FMP per-symbol validated: +{checked} checked, total fmp_ok={len(fmp_ok)}")
        except Exception as e:
            logger.warning(f"FMP per-symbol validation skipped: {e}")

    # Alpaca day presence on previous business day
    pbd = _prev_business_day(date.today())
    # Batch in chunks of 100
    present: Set[str] = set()
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        got = alpaca_day_presence(batch, pbd)
        present |= got
    alp_ok = present
    logger.info(f"Alpaca presence on {pbd}: {len(alp_ok)} / {len(tickers)}")

    # Compose table
    df = pd.DataFrame({
        "symbol": tickers,
        "fmp_ok": [t in fmp_ok for t in tickers],
        "alpaca_ok": [t in alp_ok for t in tickers],
    })
    df["equities_ok"] = df["fmp_ok"] | df["alpaca_ok"]
    df["last_checked"] = pd.Timestamp.utcnow()

    out_dir = Path("data_layer/reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "universe_validity.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Wrote validity table: {parquet_path}")

    keep = df[df["equities_ok"][df.index]].symbol.tolist()
    valid_txt = out_dir / "universe_symbols_valid.txt"
    valid_txt.write_text("\n".join(keep))
    logger.info(f"Wrote pruned universe ({len(keep)}): {valid_txt}")

    if args.update_universe:
        uni_path = out_dir / "universe_symbols.txt"
        uni_path.write_text("\n".join(keep))
        logger.info(f"Updated universe: {uni_path}")


if __name__ == "__main__":
    main()
