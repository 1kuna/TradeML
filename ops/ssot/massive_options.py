from __future__ import annotations

"""
Utilities to sample Massive.com options endpoints (contracts + aggregates) and persist raw snapshots.

Best-effort, bounded usage with BudgetManager ('massive' vendor), persisted under:
  - data_layer/raw/massive/options_contracts/date=YYYY-MM-DD/underlier=SYM/data.parquet
  - data_layer/raw/massive/options_aggregates/date=YYYY-MM-DD/ticker=OPT_TICKER/data.parquet
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from data_layer.connectors.massive_connector import MassiveConnector
from ops.ssot.budget import BudgetManager
from data_layer.storage.s3_client import get_s3_client


def _persist_df_local(df: pd.DataFrame, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.parquet").write_bytes(b"")  # ensure file exists
    df.to_parquet(root / "data.parquet", index=False)


def sample_and_persist(underliers: List[str], asof: date, max_contracts_per_ul: int = 1) -> int:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        logger.debug("MASSIVE_API_KEY missing; skipping massive options sampling")
        return 0
    mc = MassiveConnector(api_key=api_key)
    s3 = get_s3_client() if os.getenv("STORAGE_BACKEND", "local").lower() == "s3" else None
    bm = BudgetManager(initial_limits={"massive": 7000}, s3_client=s3, manifest_key="manifests/reference_budget.json")

    saved = 0
    for ul in underliers:
        # Contracts list
        if not bm.try_consume("massive", 1):
            break
        df, _ = mc.list_options_contracts(ul, as_of=asof, limit=1000)
        if df.empty:
            continue
        # Persist contracts snapshot
        if os.getenv("STORAGE_BACKEND", "local").lower() == "s3":
            try:
                import io
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                s3.put_object(key=f"raw/massive/options_contracts/date={asof.isoformat()}/underlier={ul}/data.parquet", data=buf.getvalue())
            except Exception:
                pass
        else:
            _persist_df_local(df, Path("data_layer/raw/massive/options_contracts") / f"date={asof.isoformat()}" / f"underlier={ul}")
        saved += len(df)

        # Pick up to N contracts and fetch recent aggregates
        picks = df.head(max_contracts_per_ul)["ticker"].dropna().tolist()
        end = asof
        start = max(asof - timedelta(days=5), asof - timedelta(days=2))
        for opt in picks:
            if not bm.try_consume("massive", 1):
                break
            ag = mc.fetch_option_aggregates(opt, start, end, multiplier=1, timespan="day")
            if ag.empty:
                continue
            if os.getenv("STORAGE_BACKEND", "local").lower() == "s3":
                try:
                    import io
                    safe = opt.replace(":", "_")
                    buf = io.BytesIO()
                    ag.to_parquet(buf, index=False)
                    s3.put_object(key=f"raw/massive/options_aggregates/date={asof.isoformat()}/ticker={safe}/data.parquet", data=buf.getvalue())
                except Exception:
                    pass
            else:
                safe = opt.replace(":", "_")
                _persist_df_local(ag, Path("data_layer/raw/massive/options_aggregates") / f"date={asof.isoformat()}" / f"ticker={safe}")
            saved += len(ag)

    logger.info(f"Massive options sampling saved rows: {saved}")
    return saved
