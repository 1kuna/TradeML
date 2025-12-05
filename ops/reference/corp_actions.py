from __future__ import annotations

"""
Corporate actions updater using Alpha Vantage (free-tier friendly).

For a set of symbols, fetch splits and dividends and write per-symbol parquet
to data_layer/reference/corp_actions/{SYM}_corp_actions.parquet
"""

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
from data_layer.connectors.massive_connector import MassiveConnector
from ops.ssot.budget import BudgetManager
from data_layer.storage.s3_client import get_s3_client


def update_corp_actions(symbols: Iterable[str]) -> int:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.debug("ALPHA_VANTAGE_API_KEY missing; skipping corp actions update")
        return 0
    conn = AlphaVantageConnector(api_key=api_key)
    # Optional Massive for cross-checks
    mkey = os.getenv("MASSIVE_API_KEY")
    mconn = None
    bm = None
    try:
        if mkey:
            mconn = MassiveConnector(api_key=mkey)
            s3 = get_s3_client() if os.getenv("STORAGE_BACKEND", "local").lower() == "s3" else None
            bm = BudgetManager(initial_limits={"massive": 7000}, s3_client=s3, manifest_key="manifests/reference_budget.json")
    except Exception:
        mconn = None
        bm = None
    out_dir = Path("data_layer/reference/corp_actions")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for sym in symbols:
        try:
            df = conn.fetch_corporate_actions(sym)
            # Cross-check/merge with Massive (best-effort, budgeted)
            if mconn and (bm is None or bm.try_consume("massive", 1)):
                try:
                    ps = mconn.fetch_splits(sym)
                    pdv = mconn.fetch_dividends(sym)
                    if not ps.empty or not pdv.empty:
                        extra = pd.concat([ps, pdv], ignore_index=True).dropna(subset=["ex_date"])
                        if not extra.empty:
                            df = pd.concat([df, extra], ignore_index=True)
                            df = df.drop_duplicates(subset=["symbol", "event_type", "ex_date", "ratio", "amount"], keep="first")
                except Exception as _e:
                    logger.debug(f"Massive CA merge skipped for {sym}: {_e}")
            if df.empty:
                continue
            p = out_dir / f"{sym}_corp_actions.parquet"
            df.to_parquet(p, index=False)
            count += len(df)
        except Exception as e:
            logger.warning(f"CA fetch failed for {sym}: {e}")
    logger.info(f"Corp actions updated for {len(list(symbols))} symbols, rows={count}")
    return count
