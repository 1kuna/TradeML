from __future__ import annotations

"""
Delistings snapshot via FMP stable endpoint.

Writes data_layer/reference/delistings_fmp.parquet using DELISTINGS schema.
"""

from pathlib import Path
from loguru import logger
import pandas as pd

from data_layer.connectors.fmp_connector import FMPConnector
from data_layer.schemas import get_schema, DataType


def update_delistings_fmp(output_dir: str = "data_layer/reference") -> int:
    try:
        conn = FMPConnector()
    except Exception as e:
        logger.warning(f"FMP connector unavailable: {e}")
        return 0
    df = conn.fetch_delisted_companies()
    if df.empty:
        logger.info("FMP delistings: no rows")
        return 0
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "delistings_fmp.parquet"
    try:
        # Validate against schema then write
        schema = get_schema(DataType.DELISTINGS)
        # enforce columns order to schema fields
        cols = [f.name for f in schema]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_parquet(p, index=False)
        logger.info(f"FMP delistings snapshot: {len(df)} rows -> {p}")
        return int(len(df))
    except Exception as e:
        logger.warning(f"Failed to write FMP delistings: {e}")
        return 0

