import os
from datetime import date
from pathlib import Path

import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_curator_rebuild_on_ca_change(tmp_path):
    os.environ["STORAGE_BACKEND"] = "local"
    import uuid
    ds = date(2001, 1, 1).isoformat()
    sym = f"ZZZ_{uuid.uuid4().hex[:6]}"

    # Raw bar without CA
    raw = pd.DataFrame([
        {"date": ds, "symbol": sym, "open": 20.0, "high": 22.0, "low": 19.5, "close": 21.0, "vwap": 20.5, "volume": 1000}
    ])
    _write(raw, Path("data_layer/raw/alpaca/equities_bars") / f"date={ds}" / "data.parquet")

    # Curate once (no CA)
    from scripts.curator import Curator
    cfg = tmp_path / "curator.yml"
    cfg.write_text(
        f"""
watermark:
  bookmark_key: manifests/curator_ci.json
jobs:
  - name: equities_bars_ohlcv
    source: alpaca
    table: equities_bars
    input_prefix: raw/alpaca/equities_bars
    output_prefix: curated/equities_ohlcv_adj
    partition: date
    idempotent: true
"""
    )
    # Ensure no pre-existing curated for this date
    cur_path = Path("data_layer/curated/equities_ohlcv_adj") / f"date={ds}"
    if cur_path.exists():
        for p in cur_path.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            cur_path.rmdir()
        except Exception:
            pass

    Curator(str(cfg)).run()
    cur_file = cur_path / "data.parquet"
    cur = pd.read_parquet(cur_file)
    close_before = cur.loc[0, "close_adj"]

    # Add split on ds (2:1) and run again
    ca = pd.DataFrame([
        {"symbol": sym, "event_type": "split", "ex_date": pd.to_datetime(ds).date(), "ratio": 2.0}
    ])
    _write(ca, Path("data_layer/reference/corp_actions") / f"{sym}_corp_actions.parquet")
    Curator(str(cfg)).run()

    cur2 = pd.read_parquet(cur_file)
    close_after = float(cur2[cur2["symbol"] == sym].iloc[0]["close_adj"]) if not cur2.empty else None
    assert close_after == close_before / 2.0
