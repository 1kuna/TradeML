from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import trademl.data_node.training_control as training_control
from trademl.data_node.training_control import evaluate_training_gates, launch_training_process


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


def test_launch_training_process_persists_local_and_shared_runtime(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(training_control, "training_preflight", lambda **kwargs: {"ok": True, "sample_rows": 5})

    class _Proc:
        pid = 4242

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):  # noqa: ANN001
        seen["command"] = command
        seen["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(training_control.subprocess, "Popen", fake_popen)

    payload = launch_training_process(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        phase=1,
        model_suite="ridge_only",
        python_executable="/usr/bin/python3",
        report_date="2026-04-02",
    )

    shared_runtime_path = data_root / "control" / "cluster" / "state" / "training_phase_1.json"
    assert payload["phase"] == 1
    assert payload["model_suite"] == "ridge_only"
    assert "training_job.py" in " ".join(seen["command"])
    assert "--shared-runtime-path" in seen["command"]
    assert json.loads((local_state / "training_phase_1.json").read_text(encoding="utf-8"))["status"] == "starting"
    assert json.loads(shared_runtime_path.read_text(encoding="utf-8"))["status"] == "starting"
