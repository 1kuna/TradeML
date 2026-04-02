from __future__ import annotations

from pathlib import Path

import pandas as pd

import trademl.data_node.training_control as training_control
from trademl.data_node.training_control import evaluate_training_gates


def test_evaluate_training_gates_requires_macro_vintages(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings", "ticker_changes"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"dataset": "equities_eod", "status": "GREEN"}]).to_parquet(qc_root / "partition_status.parquet", index=False)
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)

    readiness = evaluate_training_gates(data_root=tmp_path, stage_symbol_count=500, stage_years=10)

    assert readiness["phase1"]["ready"] is False
    assert "macro_vintages" in readiness["phase1"]["blockers"]
