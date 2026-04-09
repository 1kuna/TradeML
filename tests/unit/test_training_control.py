from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import trademl.data_node.training_control as training_control
from trademl.data_node.training_control import (
    evaluate_training_gates,
    launch_training_process,
    phase_freeze_state_path,
    read_pinned_phase_freeze,
    read_training_runtime,
    recommended_training_cutoff,
)


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
    assert readiness["freeze_cutoff"]["date"] is None


def test_evaluate_training_gates_ignores_unreadable_reference_parquet(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings", "ticker_changes"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    (reference_root / "fred_vintagedates.parquet").write_text("not parquet", encoding="utf-8")
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


def test_recommended_training_cutoff_uses_latest_complete_raw_partition_before_lagged_anchor(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "equities_bars"
    for trading_date, symbols in {
        "2026-04-06": ["AAPL", "MSFT"],
        "2026-04-02": ["AAPL", "MSFT", "NVDA"],
        "2026-04-01": ["AAPL", "MSFT", "NVDA"],
        "2026-03-31": ["AAPL", "MSFT", "NVDA"],
    }.items():
        partition = raw_root / f"date={trading_date}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"symbol": symbol, "date": trading_date} for symbol in symbols]).to_parquet(
            partition / "data.parquet",
            index=False,
        )

    cutoff = recommended_training_cutoff(
        data_root=tmp_path,
        expected_symbol_count=3,
        as_of="2026-04-07",
        lag_days=7,
    )

    assert cutoff["date"] == "2026-03-31"
    assert cutoff["complete_symbols"] == 3
    assert cutoff["coverage_ratio"] == 1.0
    assert cutoff["anchor_date"] == "2026-03-31"
    assert cutoff["pinned"] is True
    assert phase_freeze_state_path(data_root=tmp_path, phase=1).exists()


def test_recommended_training_cutoff_reuses_pinned_phase1_freeze(tmp_path: Path) -> None:
    freeze_path = phase_freeze_state_path(data_root=tmp_path, phase=1)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(
            {
                "date": "2026-03-31",
                "complete_symbols": 500,
                "expected_symbols": 500,
                "coverage_ratio": 1.0,
                "lag_days": 30,
                "anchor_date": "2026-03-31",
                "phase": 1,
                "pinned": True,
                "pinned_at": "2026-04-08T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    cutoff = recommended_training_cutoff(
        data_root=tmp_path,
        expected_symbol_count=500,
        as_of="2026-04-15",
        lag_days=30,
    )

    assert cutoff["date"] == "2026-03-31"
    assert cutoff["pinned"] is True
    assert cutoff["pin_path"].endswith("phase1_freeze.json")


def test_evaluate_training_gates_use_freeze_cutoff_for_bar_readiness(tmp_path: Path, monkeypatch) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    pd.DataFrame([{"series_id": "DGS10", "vintage_date": "2026-03-01"}]).to_parquet(
        reference_root / "fred_vintagedates.parquet",
        index=False,
    )
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset": "equities_eod", "source": "alpaca", "date": "2026-03-07", "status": "GREEN"},
        ]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)

    monkeypatch.setattr(
        training_control,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2026-03-07", "coverage_ratio": 1.0, "complete_symbols": 500, "expected_symbols": 500},
    )
    monkeypatch.setattr(training_control, "_planner_bars_ratio", lambda path: 0.0)

    readiness = evaluate_training_gates(data_root=tmp_path, stage_symbol_count=500, stage_years=10)

    assert readiness["phase1"]["ready"] is True
    assert "canonical_eod_bars" not in readiness["phase1"]["blockers"]
    assert readiness["freeze_cutoff"]["window_coverage_ratio"] == 1.0


def test_evaluate_training_gates_block_when_frozen_window_has_historical_gaps(tmp_path: Path, monkeypatch) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    pd.DataFrame([{"series_id": "DGS10", "vintage_date": "2026-03-01"}]).to_parquet(
        reference_root / "fred_vintagedates.parquet",
        index=False,
    )
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset": "equities_eod", "source": "alpaca", "date": "2021-04-05", "status": "AMBER"},
            {"dataset": "equities_eod", "source": "alpaca", "date": "2026-03-06", "status": "GREEN"},
        ]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)

    monkeypatch.setattr(
        training_control,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2026-03-06", "coverage_ratio": 1.0, "complete_symbols": 500, "expected_symbols": 500},
    )
    monkeypatch.setattr(training_control, "_planner_bars_ratio", lambda path: 1.0)

    readiness = evaluate_training_gates(data_root=tmp_path, stage_symbol_count=500, stage_years=10)

    assert readiness["phase1"]["ready"] is False
    assert "canonical_eod_bars" in readiness["phase1"]["blockers"]
    assert readiness["freeze_cutoff"]["window_coverage_ratio"] < 0.98
    assert readiness["freeze_cutoff"]["window_missing_dates"] == ["2021-04-05"]


def test_evaluate_training_gates_uses_planner_window_ratio_for_live_progress(tmp_path: Path, monkeypatch) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    pd.DataFrame([{"series_id": "DGS10", "vintage_date": "2026-03-01"}]).to_parquet(
        reference_root / "fred_vintagedates.parquet",
        index=False,
    )
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset": "equities_eod", "source": "alpaca", "date": "2021-04-05", "status": "AMBER"},
            {"dataset": "equities_eod", "source": "alpaca", "date": "2026-03-06", "status": "GREEN"},
        ]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)

    monkeypatch.setattr(
        training_control,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2026-03-06", "coverage_ratio": 1.0, "complete_symbols": 500, "expected_symbols": 500},
    )
    monkeypatch.setattr(training_control, "_planner_bars_ratio", lambda path: 1.0)
    monkeypatch.setattr(training_control, "_planner_window_ratio", lambda **kwargs: 1.0)

    readiness = evaluate_training_gates(data_root=tmp_path, stage_symbol_count=500, stage_years=10)

    assert readiness["phase1"]["ready"] is True
    assert readiness["freeze_cutoff"]["window_coverage_ratio"] < 0.98
    assert readiness["freeze_cutoff"]["planner_window_coverage_ratio"] == 1.0
    assert readiness["freeze_cutoff"]["effective_window_coverage_ratio"] == 1.0


def test_evaluate_training_gates_reads_planner_window_ratio_from_explicit_local_db(tmp_path: Path, monkeypatch) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ["corp_actions", "listing_history", "delistings", "sec_filings"]:
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / f"{name}.parquet", index=False)
    pd.DataFrame([{"series_id": "DGS10", "vintage_date": "2026-03-01"}]).to_parquet(
        reference_root / "fred_vintagedates.parquet",
        index=False,
    )
    for series in training_control.default_macro_series():
        partition = tmp_path / "data" / "raw" / "macros_fred" / f"series={series}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"series_id": series, "value": 1.0}]).to_parquet(partition / "data.parquet", index=False)
    qc_root = tmp_path / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"dataset": "equities_eod", "source": "alpaca", "date": "2026-03-06", "status": "AMBER"}]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)

    planner_db = tmp_path / "workspace" / "control" / "node.sqlite"
    planner_db.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        training_control,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2026-03-06", "coverage_ratio": 1.0, "complete_symbols": 500, "expected_symbols": 500},
    )
    monkeypatch.setattr(training_control, "_planner_bars_ratio", lambda path: 1.0 if path == planner_db else None)
    monkeypatch.setattr(training_control, "_planner_window_ratio", lambda **kwargs: 1.0 if kwargs["db_path"] == planner_db else None)

    readiness = evaluate_training_gates(
        data_root=tmp_path,
        stage_symbol_count=500,
        stage_years=10,
        planner_db_path=planner_db,
    )

    assert readiness["phase1"]["ready"] is True
    assert readiness["freeze_cutoff"]["planner_window_coverage_ratio"] == 1.0


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


def test_launch_training_process_defaults_report_date_to_freeze_cutoff(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(training_control, "training_preflight", lambda **kwargs: {"ok": True, "sample_rows": 5})
    monkeypatch.setattr(training_control, "recommended_training_cutoff", lambda **kwargs: {"date": "2026-04-02"})
    monkeypatch.setattr(training_control, "_stage_symbol_count", lambda root: 500)

    class _Proc:
        pid = 4242

    seen: dict[str, object] = {}

    def fake_popen(command, **kwargs):  # noqa: ANN001
        seen["command"] = command
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
    )

    assert payload["report_date"] == "2026-04-02"
    assert "2026-04-02" in " ".join(seen["command"])


def test_read_pinned_phase_freeze_logs_invalid_json(tmp_path: Path, caplog) -> None:
    freeze_path = phase_freeze_state_path(data_root=tmp_path, phase=1)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text("{invalid", encoding="utf-8")

    with caplog.at_level("WARNING"):
        payload = read_pinned_phase_freeze(data_root=tmp_path, phase=1)

    assert payload is None
    assert any("invalid_phase_freeze_json" in record.message for record in caplog.records)


def test_read_training_runtime_logs_invalid_json(tmp_path: Path, caplog) -> None:
    runtime_path = tmp_path / "control" / "training_phase_1.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text("{invalid", encoding="utf-8")

    with caplog.at_level("WARNING"):
        payload = read_training_runtime(path=runtime_path)

    assert payload == {}
    assert any("invalid_training_runtime_json" in record.message for record in caplog.records)
