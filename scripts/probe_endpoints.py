#!/usr/bin/env python
"""
Probe vendor endpoints and connector calls with minimal requests to verify
URL formats, auth, and transform outputs. Safe to run locally; does not write S3.

Tests (single symbol: AAPL, prev business day):
- Alpaca: 1Day, 1Min, 5Min bars via connector
- Polygon: day aggregates via connector
- Finnhub: options chain via connector
- FRED: treasury curve for a single day via connector
- FMP (stable): search-symbol, historical-price-eod, dividends, earnings, splits, calendars, delisted
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger


def prev_business_day(d: date) -> date:
    dd = d - timedelta(days=1)
    while dd.weekday() >= 5:
        dd -= timedelta(days=1)
    return dd


def head_cols(df: pd.DataFrame, n: int = 5) -> str:
    return ", ".join(list(df.columns)[:n])


def run():
    load_dotenv()
    sym = os.getenv("PROBE_SYMBOL", "AAPL").upper()
    today = date.today()
    pbd = prev_business_day(today)
    fmp_key = os.getenv("FMP_API_KEY", "")

    results: Dict[str, str] = {}

    # ---- Alpaca (connector) ----
    try:
        from data_layer.connectors.alpaca_connector import AlpacaConnector
        alp = AlpacaConnector()
        d1 = alp.fetch_bars([sym], pbd, pbd, timeframe="1Day")
        results["alpaca_1d"] = f"ok rows={len(d1)} cols=[{head_cols(d1)}]" if not d1.empty else "empty"
        d5 = alp.fetch_bars([sym], pbd, pbd, timeframe="5Min")
        results["alpaca_5m"] = f"ok rows={len(d5)} cols=[{head_cols(d5)}]" if not d5.empty else "empty"
        d1m = alp.fetch_bars([sym], pbd, pbd, timeframe="1Min")
        results["alpaca_1m"] = f"ok rows={len(d1m)} cols=[{head_cols(d1m)}]" if not d1m.empty else "empty"
    except Exception as e:
        results["alpaca"] = f"error: {e}"

    # ---- Polygon (connector) ----
    try:
        from data_layer.connectors.polygon_connector import PolygonConnector
        pg = PolygonConnector()
        pd1 = pg.fetch_aggregates(sym, pbd, pbd, timespan="day")
        results["polygon_day"] = f"ok rows={len(pd1)} cols=[{head_cols(pd1)}]" if not pd1.empty else "empty"
    except Exception as e:
        results["polygon"] = f"error: {e}"

    # ---- Finnhub (connector) ----
    try:
        from data_layer.connectors.finnhub_connector import FinnhubConnector
        fh = FinnhubConnector()
        ch = fh.fetch_options_chain(sym)
        results["finnhub_chain"] = f"ok rows={len(ch)} cols=[{head_cols(ch)}]" if not ch.empty else "empty"
    except Exception as e:
        results["finnhub"] = f"error: {e}"

    # ---- FRED (connector) ----
    try:
        from data_layer.connectors.fred_connector import FREDConnector
        fr = FREDConnector()
        td = fr.fetch_treasury_curve(start_date=pbd, end_date=pbd)
        results["fred_curve"] = f"ok rows={len(td)} cols=[{head_cols(td)}]" if not td.empty else "empty"
    except Exception as e:
        results["fred"] = f"error: {e}"

    # ---- FMP (stable endpoints) ----
    def _q(u: str) -> str:
        try:
            r = requests.get(u, timeout=20)
            return f"{r.status_code} len={len(r.content)} ct={r.headers.get('Content-Type','')}"
        except Exception as e:
            return f"error: {e}"

    if fmp_key:
        base = "https://financialmodelingprep.com/stable"
        eod_url = f"{base}/historical-price-eod?symbol={sym}&from={pbd.isoformat()}&to={pbd.isoformat()}&apikey={fmp_key}"
        results["fmp_eod"] = _q(eod_url)
        results["fmp_search"] = _q(f"{base}/search-symbol?query={sym}&apikey={fmp_key}")
        results["fmp_dividends"] = _q(f"{base}/dividends?symbol={sym}&apikey={fmp_key}")
        results["fmp_div_cal"] = _q(f"{base}/dividends-calendar?apikey={fmp_key}")
        results["fmp_earnings"] = _q(f"{base}/earnings?symbol={sym}&apikey={fmp_key}")
        results["fmp_earn_cal"] = _q(f"{base}/earnings-calendar?apikey={fmp_key}")
        results["fmp_splits"] = _q(f"{base}/splits?symbol={sym}&apikey={fmp_key}")
        results["fmp_splits_cal"] = _q(f"{base}/splits-calendar?apikey={fmp_key}")
        results["fmp_delisted"] = _q(f"{base}/delisted-companies?page=0&limit=10&apikey={fmp_key}")
        # Alternate EOD forms (light/full) and index example (^GSPC)
        results["fmp_eod_light_aapl"] = _q(f"{base}/historical-price-eod/light?symbol={sym}&apikey={fmp_key}")
        results["fmp_eod_full_aapl"] = _q(f"{base}/historical-price-eod/full?symbol={sym}&apikey={fmp_key}")
        results["fmp_eod_light_gspc"] = _q(f"{base}/historical-price-eod/light?symbol=%5EGSPC&apikey={fmp_key}")
        results["fmp_eod_full_gspc"] = _q(f"{base}/historical-price-eod/full?symbol=%5EGSPC&apikey={fmp_key}")
    else:
        results["fmp"] = "no FMP_API_KEY set"

    # Print summary
    print("Probe results (", sym, ")")
    for k, v in results.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    run()
