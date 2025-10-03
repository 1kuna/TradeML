import os
from datetime import date
from pathlib import Path

import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_run_options_vol_scaffold_no_data():
    from ops.pipelines.options_vol import OptionsVolConfig, run_options_vol
    out = run_options_vol(OptionsVolConfig(asof=date.today().isoformat(), underliers=["AAPL"]))
    assert out["status"] in ("ok", "no_data")

