from __future__ import annotations

"""Reference data updaters (corporate actions, delistings).

Free-tier friendly: fetch delisted lists from Alpha Vantage and/or FMP when
API keys are present; write individual source files and a merged canonical
file. Corporate actions per symbol are handled elsewhere.
"""

import os
from pathlib import Path
from datetime import datetime
from loguru import logger

import io
import pandas as pd


def _update_delistings_av() -> pd.DataFrame:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.debug("ALPHA_VANTAGE_API_KEY missing; skipping AV delistings")
        return pd.DataFrame()
    try:
        import requests
        from data_layer.schemas import DataType, get_schema
        from data_layer.connectors.base import BaseConnector

        out_dir = Path("data_layer/reference/delistings")
        out_dir.mkdir(parents=True, exist_ok=True)
        url = "https://www.alphavantage.co/query"
        params = {"function": "LISTING_STATUS", "state": "delisted", "apikey": api_key}
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty:
            logger.info("No delisted symbols returned from Alpha Vantage")
            return pd.DataFrame()
        delist_rows = []
        for _, r in df.iterrows():
            delist_rows.append(
                {
                    "symbol": r.get("symbol"),
                    "delist_date": pd.to_datetime(r.get("delistingDate"), errors="coerce").date() if pd.notna(r.get("delistingDate")) else None,
                    "reason": r.get("status", "unknown"),
                    "source_name": "alpha_vantage",
                    "source_uri": "alpha_vantage://LISTING_STATUS/delisted",
                    "ingested_at": pd.Timestamp.utcnow(),
                }
            )
        out = pd.DataFrame(delist_rows)
        out_path = out_dir / "delistings_av.parquet"
        out.to_parquet(out_path, index=False)
        logger.info(f"Delistings (AV) updated: {len(out)} rows → {out_path}")
        return out
    except Exception as e:
        logger.warning(f"AV delistings update failed: {e}")
        return pd.DataFrame()


def _update_delistings_fmp(max_attempts: int = 5) -> pd.DataFrame:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        logger.debug("FMP_API_KEY missing; skipping FMP delistings")
        return pd.DataFrame()
    try:
        from data_layer.connectors.fmp_connector import FMPConnector
        out_dir = Path("data_layer/reference/delistings")
        out_dir.mkdir(parents=True, exist_ok=True)
        conn = FMPConnector(api_key=api_key)
        attempt = 0
        last_err = None
        while attempt < max_attempts:
            try:
                df = conn.fetch_delisted_companies()
                if not df.empty:
                    p = out_dir / "delistings_fmp.parquet"
                    df.to_parquet(p, index=False)
                    logger.info(f"Delistings (FMP) updated: {len(df)} rows → {p}")
                    return df
                else:
                    logger.warning("FMP returned empty delistings; retrying")
            except Exception as e:
                last_err = e
                logger.warning(f"FMP attempt {attempt+1}/{max_attempts} failed: {e}")
            # Backoff before retry
            import time
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
            attempt += 1
        if last_err:
            logger.warning(f"FMP delistings update failed after {max_attempts} attempts: {last_err}")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"FMP delistings update failed: {e}")
        return pd.DataFrame()


def update_reference() -> None:
    out_dir = Path("data_layer/reference/delistings")
    out_dir.mkdir(parents=True, exist_ok=True)
    av = _update_delistings_av()
    fmp = _update_delistings_fmp()
    if av.empty and fmp.empty:
        return
    # Merge
    cols = ["symbol", "delist_date", "reason"]
    merged = pd.concat([av[cols]] if not av.empty else [], ignore_index=True)
    if not fmp.empty:
        merged = pd.concat([merged, fmp[cols]], ignore_index=True) if not merged.empty else fmp[cols]
    if merged.empty:
        return
    merged = merged.dropna(subset=["symbol", "delist_date"]).drop_duplicates(subset=["symbol", "delist_date"], keep="first")
    merged = merged.sort_values(["delist_date", "symbol"]).reset_index(drop=True)
    merged_path = out_dir / "delistings.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info(f"Delistings merged: {len(merged)} rows → {merged_path}")
