from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import trademl.data_node.training_control as training_control
from trademl.data_node.training_control import auto_launch_phase_training, evaluate_training_gates


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


def test_auto_launch_phase_training_starts_ready_phases(tmp_path: Path, monkeypatch) -> None:
    stage_path = tmp_path / "stage.yml"
    stage_path.write_text(yaml.safe_dump({"symbols": [f"S{i:03d}" for i in range(500)], "years": 10}), encoding="utf-8")
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "sec_filings", "ticker_changes", "fred_vintagedates"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    pd.DataFrame([{"symbol": "OLD", "delistedDate": "2024-01-01"}]).to_parquet(reference_root / "delistings.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"dataset": "equities_eod", "status": "GREEN"}]).to_parquet(qc_root / "partition_status.parquet", index=False)
    curated_partition = tmp_path / "data" / "curated" / "equities_ohlcv_adj" / "date=2025-01-10"
    curated_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2025-01-10", "symbol": "AAPL", "close": 1.0, "volume": 1.0}]).to_parquet(curated_partition / "data.parquet", index=False)
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)

    launched: list[int] = []

    def fake_launch_training_process(**kwargs):  # noqa: ANN003
        launched.append(kwargs["phase"])
        return {"phase": kwargs["phase"], "pid": 1000 + kwargs["phase"], "running": True}

    monkeypatch.setattr(training_control, "launch_training_process", fake_launch_training_process)

    launched_payloads = auto_launch_phase_training(
        repo_root=tmp_path,
        data_root=tmp_path,
        local_state=tmp_path / "control",
        env_path=tmp_path / ".env",
        stage_path=stage_path,
    )

    assert launched == [1, 2]
    assert [payload["phase"] for payload in launched_payloads] == [1, 2]
