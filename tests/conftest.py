from __future__ import annotations

import os
import sys
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_ROOT = _REPO_ROOT / "legacy"
_cleaned_path = []
for entry in sys.path:
    if not entry:
        _cleaned_path.append(entry)
        continue
    try:
        if Path(entry).resolve() == _LEGACY_ROOT:
            continue
    except Exception:
        pass
    _cleaned_path.append(entry)
sys.path[:] = _cleaned_path
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from data_node.db import NodeDB, reset_db
from data_node.budgets import reset_budget_manager


@pytest.fixture()
def temp_data_root(tmp_path, monkeypatch):
    """
    Isolated DATA_ROOT for tests (keeps sqlite/state files away from the repo).
    Cleans up on teardown.
    """
    data_root = tmp_path / "data_root"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    yield data_root
    reset_db()
    reset_budget_manager()
    shutil.rmtree(data_root, ignore_errors=True)


@pytest.fixture()
def node_db(temp_data_root):
    """
    Fresh NodeDB instance backed by a throwaway sqlite file.
    """
    db_path = temp_data_root / "data_layer" / "control" / "node.sqlite"
    reset_db()
    db = NodeDB(db_path)
    db.init_db()
    yield db
    db.close()
    reset_db()


@pytest.fixture()
def curated_prices(temp_data_root, monkeypatch) -> Dict[str, object]:
    """
    Synthetic curated OHLCV history for two symbols (~90 business days).
    Written to the same layout the feature/label code expects.
    """
    base_dir = temp_data_root / "data_layer" / "curated" / "equities_ohlcv_adj"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CURATED_EQUITY_BARS_ADJ_DIR", str(base_dir))

    start = date.today() - timedelta(days=140)
    dates = pd.bdate_range(start=start, periods=90).date

    def _frame(symbol: str, start_price: float) -> pd.DataFrame:
        prices = np.linspace(start_price, start_price * 1.1, num=len(dates))
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "session_id": [d.strftime("%Y%m%d") for d in dates],
                "open_adj": prices * 0.99,
                "high_adj": prices * 1.01,
                "low_adj": prices * 0.98,
                "close_adj": prices,
                "vwap_adj": prices,
                "volume_adj": 1_000_000,
                "close_raw": prices,
                "adjustment_factor": 1.0,
                "last_adjustment_date": None,
                "ingested_at": pd.Timestamp.utcnow(),
                "source_name": "synthetic",
                "source_uri": "curator://synthetic",
                "transform_id": "synthetic_test",
            }
        )

    frames: List[pd.DataFrame] = []
    for sym, start_px in (("AAPL", 100.0), ("MSFT", 220.0)):
        df = _frame(sym, start_px)
        frames.append(df)
        df.to_parquet(base_dir / f"{sym}_adj.parquet", index=False)

    panel = pd.concat(frames, ignore_index=True)
    return {"base_dir": base_dir, "symbols": ["AAPL", "MSFT"], "dates": dates, "panel": panel}
