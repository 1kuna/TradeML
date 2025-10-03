from __future__ import annotations

"""
Polygon active tickers snapshot (free-tier paging, limited per run).

Fetch up to max_pages pages of active stock tickers and write a snapshot to
data_layer/reference/polygon_tickers.parquet (overwrites). Intended to be run
infrequently (e.g., daily) and governed by the 'polygon' API budget.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from data_layer.connectors.polygon_connector import PolygonConnector
from ops.ssot.budget import BudgetManager
from data_layer.storage.s3_client import get_s3_client


def update_polygon_universe(max_pages: int = 1, page_limit: int = 1000) -> int:
    pkey = os.getenv("POLYGON_API_KEY")
    if not pkey:
        logger.debug("POLYGON_API_KEY missing; skipping polygon universe update")
        return 0

    pc = PolygonConnector(api_key=pkey)
    s3 = get_s3_client() if os.getenv("STORAGE_BACKEND", "local").lower() == "s3" else None
    bm = BudgetManager(initial_limits={"polygon": 7000}, s3_client=s3, manifest_key="manifests/reference_budget.json")

    frames = []
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        if not bm.try_consume("polygon", 1):
            logger.warning("Polygon budget exhausted; stopping ticker paging")
            break
        df, next_cursor = pc.list_active_tickers(cursor=cursor, limit=page_limit)
        if df.empty:
            break
        frames.append(df)
        pages += 1
        cursor = next_cursor
        if not cursor:
            break

    if not frames:
        return 0
    out = pd.concat(frames, ignore_index=True)
    out_dir = Path("data_layer/reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "polygon_tickers.parquet"
    out.to_parquet(p, index=False)
    logger.info(f"Polygon tickers snapshot written: {len(out)} rows → {p}")
    return int(len(out))
