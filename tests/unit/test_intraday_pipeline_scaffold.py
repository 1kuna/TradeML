from datetime import date
from pathlib import Path
import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_run_intraday_scaffold(tmp_path):
    ds = date(2020, 1, 1)
    # Create minimal curated minute file
    rows = []
    for m in range(5):
        rows.append({"date": ds, "symbol": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + m * 0.1})
    df = pd.DataFrame(rows)
    _write(df, Path("data_layer/curated/equities_minute") / f"date={ds.isoformat()}" / "data.parquet")

    from ops.pipelines.intraday_xs import IntradayConfig, run_intraday
    out = run_intraday(IntradayConfig(start_date=ds.isoformat(), end_date=ds.isoformat(), universe=["AAPL"]))
    assert out["status"] in ("ok", "no_data")

