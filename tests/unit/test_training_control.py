from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import trademl.data_node.training_control as training_control
from trademl.data_node.training_control import (
    evaluate_training_gates,
    launch_training_process,
    resolve_training_target,
    phase_freeze_state_path,
    read_pinned_phase_freeze,
    read_training_runtime,
    stop_training_process,
    training_status_snapshot,
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


def test_launch_training_process_uses_explicit_output_root_for_artifacts(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    output_root = tmp_path / "experiment_outputs" / "run_a"

    monkeypatch.setattr(training_control, "training_preflight", lambda **kwargs: {"ok": True, "sample_rows": 5})

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
        report_date="2026-04-02",
        output_root=output_root,
    )

    assert payload["output_root"] == str(output_root)
    assert "--output-root" in seen["command"]
    assert str(output_root) in seen["command"]


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


def test_resolve_training_target_prefers_configured_remote_default(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    local_state = tmp_path / "control"
    data_root = tmp_path / "nas"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "node.yml").write_text(
        """
training:
  default_target: workstation-remote
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )

    target = resolve_training_target(
        target_name=None,
        targets_config_path=repo_root / "configs" / "node.yml",
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable="/opt/python",
    )

    assert target.name == "workstation-remote"
    assert target.default is True
    assert target.host == "192.168.1.10"
    assert str(target.repo_root) == "/srv/trademl"


def test_training_preflight_uses_remote_target_checks(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    config_path = repo_root / "configs" / "equities_xs.yml"
    config_path.write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    node_config = repo_root / "configs" / "node.yml"
    node_config.write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(training_control, "_run_ssh_command", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": '{"ok": true, "sample_rows": 12, "sample_date": "2026-04-08", "qc_path": "/srv/qc.parquet"}\n', "stderr": ""})())
    monkeypatch.setattr(training_control, "_target_preflight", lambda **kwargs: {"ok": True, "target": "workstation-remote", "kind": "ssh"})

    payload = training_control.training_preflight(
        data_root=tmp_path / "nas",
        config_path=config_path,
        repo_root=repo_root,
        local_state=tmp_path / "state",
        targets_config_path=node_config,
        target="workstation-remote",
    )

    assert payload["ok"] is True
    assert payload["resolved_target"]["name"] == "workstation-remote"
    assert payload["dataset"]["sample_rows"] == 12


def test_local_dataset_preflight_uses_parquet_metadata_instead_of_full_read(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs" / "equities_xs.yml"
    curated_path = tmp_path / "nas" / "data" / "curated" / "equities_ohlcv_adj" / "date=2026-04-09" / "data.parquet"
    qc_path = tmp_path / "nas" / "data" / "qc" / "partition_status.parquet"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    pd.DataFrame({"date": ["2026-04-09"], "dataset": ["equities_eod"]}).to_parquet(qc_path, index=False)
    pd.DataFrame({"symbol": ["AAA", "BBB"], "close": [1.0, 2.0]}).to_parquet(curated_path, index=False)

    monkeypatch.setattr(
        training_control.pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full pandas parquet read should not run during local dataset preflight")),
    )

    payload = training_control._local_dataset_preflight(
        data_root=tmp_path / "nas",
        config_path=config_path,
    )

    assert payload["ok"] is True
    assert payload["sample_rows"] == 2
    assert payload["sample_date"] == "2026-04-09"


def test_launch_training_process_remote_persists_controller_runtime(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    (repo_root / "configs" / "node.yml").write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(training_control, "training_preflight", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(training_control, "recommended_training_cutoff", lambda **kwargs: {"date": "2026-04-02"})
    monkeypatch.setattr(training_control, "_stage_symbol_count", lambda root: 500)
    monkeypatch.setattr(training_control, "_launch_remote_training_process", lambda **kwargs: 5151)
    monkeypatch.setattr(training_control, "_resolve_default_report_date", lambda **kwargs: "2026-04-02")
    monkeypatch.setattr(
        training_control,
        "_sync_remote_training_config",
        lambda **kwargs: Path("/srv/trademl/control/configs/training_phase_1.yml"),
    )

    payload = launch_training_process(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    runtime_path = local_state / "training_runs" / "workstation-remote" / "training_phase_1.json"
    assert payload["pid"] == 5151
    assert payload["target"] == "workstation-remote"
    assert json.loads(runtime_path.read_text(encoding="utf-8"))["remote_runtime_path"].endswith("training_phase_1.json")
    assert payload["execution_config_path"] == "/srv/trademl/control/configs/training_phase_1.yml"


def test_launch_training_process_remote_serializes_nested_preflight_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    (repo_root / "configs" / "node.yml").write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        training_control,
        "training_preflight",
        lambda **kwargs: {
            "ok": True,
            "resolved_target": {
                "repo_root": Path("/srv/trademl"),
                "data_root": Path("/srv/trademl-data"),
            },
        },
    )
    monkeypatch.setattr(training_control, "recommended_training_cutoff", lambda **kwargs: {"date": "2026-04-02"})
    monkeypatch.setattr(training_control, "_stage_symbol_count", lambda root: 500)
    monkeypatch.setattr(training_control, "_launch_remote_training_process", lambda **kwargs: 5151)
    monkeypatch.setattr(training_control, "_resolve_default_report_date", lambda **kwargs: "2026-04-02")
    monkeypatch.setattr(
        training_control,
        "_sync_remote_training_config",
        lambda **kwargs: Path("/srv/trademl/control/configs/training_phase_1.yml"),
    )

    launch_training_process(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    runtime_path = local_state / "training_runs" / "workstation-remote" / "training_phase_1.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert payload["preflight"]["resolved_target"]["repo_root"] == "/srv/trademl"
    assert payload["preflight"]["resolved_target"]["data_root"] == "/srv/trademl-data"


def test_launch_training_process_remote_defaults_report_date_from_remote_freeze(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data:\n  green_threshold: 0.9\n", encoding="utf-8")
    (repo_root / "configs" / "node.yml").write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(training_control, "training_preflight", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(training_control, "_stage_symbol_count", lambda root: 500)
    monkeypatch.setattr(training_control, "_launch_remote_training_process", lambda **kwargs: 5151)
    monkeypatch.setattr(training_control, "_resolve_default_report_date", lambda **kwargs: "2026-03-09")
    monkeypatch.setattr(
        training_control,
        "_sync_remote_training_config",
        lambda **kwargs: Path("/srv/trademl/control/configs/training_phase_1.yml"),
    )
    monkeypatch.setattr(
        training_control,
        "_sync_remote_training_config",
        lambda **kwargs: Path("/srv/trademl/control/configs/training_phase_1.yml"),
    )

    payload = launch_training_process(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    assert payload["report_date"] == "2026-03-09"


def test_training_status_snapshot_and_stop_use_remote_runtime(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "node.yml").write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    local_runtime = local_state / "training_runs" / "workstation-remote" / "training_phase_1.json"
    local_runtime.parent.mkdir(parents=True, exist_ok=True)
    local_runtime.write_text(
        json.dumps(
            {
                "phase": 1,
                "pid": 5151,
                "target": "workstation-remote",
                "status": "running",
                "shared_runtime_path": "/srv/trademl-data/control/cluster/state/training_phase_1.json",
                "remote_runtime_path": "/srv/trademl/control/training_phase_1.json",
                "remote_log_path": "/srv/trademl/control/logs/training_phase_1.log",
            }
        ),
        encoding="utf-8",
    )
    writes: list[str] = []

    def fake_ssh(_target, command, *, check=False):  # noqa: ANN001
        if "read_training_runtime" in command:
            if "/srv/trademl-data/control/cluster/state/training_phase_1.json" in command:
                return type(
                    "R",
                    (),
                    {
                        "returncode": 0,
                        "stdout": '{"pid": 5151, "status": "running", "running": true, "host": "192.168.1.10", "report_date": "2026-03-09"}\n',
                        "stderr": "",
                    },
                )()
            if "/srv/trademl/control/training_phase_1.json" in command:
                return type(
                    "R",
                    (),
                    {
                        "returncode": 0,
                        "stdout": '{"pid": 5151, "status": "running", "running": true, "host": "192.168.1.10"}\n',
                        "stderr": "",
                    },
                )()
        if command.startswith("tail -n"):
            return type("R", (), {"returncode": 0, "stdout": "line-1\nline-2\n", "stderr": ""})()
        if command.startswith("kill -TERM"):
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if "_write_runtime_payload" in command:
            writes.append(command)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(training_control, "_run_ssh_command", fake_ssh)

    snapshot = training_status_snapshot(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    assert snapshot["runtime"]["pid"] == 5151
    assert snapshot["runtime"]["remote_log_path"] == "/srv/trademl/control/logs/training_phase_1.log"
    assert snapshot["shared"]["report_date"] == "2026-03-09"
    assert "line-2" in snapshot["log_tail"]

    stopped = stop_training_process(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    assert stopped["stopped"] is True
    assert stopped["runtime"]["status"] == "stopped"
    assert any("/srv/trademl/control/training_phase_1.json" in command for command in writes)
    assert any("/srv/trademl-data/control/cluster/state/training_phase_1.json" in command for command in writes)


def test_training_status_snapshot_refreshes_local_mirror_from_remote_runtime(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "node.yml").write_text(
        """
training_targets:
  workstation-remote:
    host: 192.168.1.10
    user: zach
    repo_root: /srv/trademl
    data_root: /srv/trademl-data
    python_executable: /usr/bin/python3
""".strip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local_state"
    local_runtime = local_state / "training_runs" / "workstation-remote" / "training_phase_1.json"
    local_runtime.parent.mkdir(parents=True, exist_ok=True)
    local_runtime.write_text(
        json.dumps(
            {
                "phase": 1,
                "pid": 5151,
                "target": "workstation-remote",
                "status": "starting",
                "running": True,
                "shared_runtime_path": str(data_root / "control" / "cluster" / "state" / "training_phase_1.json"),
                "remote_log_path": "/srv/trademl/control/logs/training_phase_1.log",
            }
        ),
        encoding="utf-8",
    )
    shared_runtime = data_root / "control" / "cluster" / "state" / "training_phase_1.json"
    shared_runtime.parent.mkdir(parents=True, exist_ok=True)
    shared_runtime.write_text(local_runtime.read_text(encoding="utf-8"), encoding="utf-8")

    def fake_ssh(_target, command, *, check=False):  # noqa: ANN001
        if "read_training_runtime" in command:
            return type(
                "R",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "pid": 5151,
                            "status": "completed",
                            "running": False,
                            "host": "192.168.1.10",
                            "finished_at": "2026-04-10T19:05:56.928912+00:00",
                        }
                    )
                    + "\n",
                    "stderr": "",
                },
            )()
        if command.startswith("tail -n"):
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        raise AssertionError(command)

    monkeypatch.setattr(training_control, "_run_ssh_command", fake_ssh)

    snapshot = training_status_snapshot(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        phase=1,
        target="workstation-remote",
        targets_config_path=repo_root / "configs" / "node.yml",
    )

    mirrored = json.loads(local_runtime.read_text(encoding="utf-8"))
    assert snapshot["runtime"]["status"] == "completed"
    assert snapshot["local"]["status"] == "completed"
    assert snapshot["local"]["running"] is False
    assert snapshot["local"]["host"] == "192.168.1.10"
    assert snapshot["local"]["remote_log_path"] == "/srv/trademl/control/logs/training_phase_1.log"
    assert mirrored["status"] == "completed"
    assert mirrored["running"] is False
    assert mirrored["host"] == "192.168.1.10"
    assert mirrored["remote_log_path"] == "/srv/trademl/control/logs/training_phase_1.log"
