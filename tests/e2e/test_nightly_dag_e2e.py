"""End-to-end tests for nightly DAG simulation."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_all_external_deps():
    """Mock all external dependencies for DAG simulation."""
    with patch("ops.dags.nightly.audit_scan") as mock_audit, \
         patch("ops.dags.nightly.curate_incremental") as mock_curate, \
         patch("ops.dags.nightly.load_partition_status") as mock_load_status, \
         patch("ops.dags.nightly.get_green_coverage") as mock_green:

        # Setup default returns
        mock_audit.return_value = None
        mock_curate.return_value = {"equities_eod": 10, "options_chains": 5}

        mock_load_status.return_value = pd.DataFrame({
            "source_name": ["alpaca", "alpaca"],
            "table_name": ["equities_eod", "equities_eod"],
            "symbol": ["AAPL", "MSFT"],
            "dt": [date.today(), date.today()],
            "status": ["GREEN", "GREEN"],
        })

        mock_green.return_value = (0.99, {"GREEN": 99, "AMBER": 1, "RED": 0})

        yield {
            "audit": mock_audit,
            "curate": mock_curate,
            "load_status": mock_load_status,
            "green": mock_green,
        }


def test_full_dag_run_all_steps_disabled():
    """Test DAG with all steps disabled completes successfully."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert results["status"] == "ok"
    assert "steps" in results
    # Only QC step runs by default (it's always enabled)
    assert "qc" in results["steps"]


def test_dag_runs_audit_and_curate(mock_all_external_deps):
    """Test DAG runs audit and curate steps correctly."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=True,
        enable_backfill=False,
        enable_curate=True,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert results["status"] == "ok"
    assert "audit" in results["steps"]
    assert "curate" in results["steps"]

    # Verify mocks were called
    mock_all_external_deps["audit"].assert_called_once()
    mock_all_external_deps["curate"].assert_called_once()


def test_dag_qc_refresh_collects_coverage(mock_all_external_deps):
    """Test QC refresh step collects coverage stats."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert "qc" in results["steps"]
    qc_result = results["steps"]["qc"]
    assert qc_result["status"] == "ok"
    assert "coverage" in qc_result


def test_dag_training_skipped_when_threshold_not_met(mock_all_external_deps):
    """Test training is skipped when GREEN threshold not met."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    # Set coverage below threshold
    mock_all_external_deps["green"].return_value = (0.50, {"GREEN": 50, "AMBER": 30, "RED": 20})

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=True,  # Enable but should skip
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
        equities_green_threshold=0.98,
    )

    results = run_nightly_dag(cfg)

    assert "train_equities" in results["steps"]
    train_result = results["steps"]["train_equities"]
    assert train_result["status"] == "skipped"
    assert train_result["reason"] == "green_threshold_not_met"


def test_dag_evaluate_step(mock_all_external_deps):
    """Test evaluate step runs correctly."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=True,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert "evaluate" in results["steps"]
    eval_result = results["steps"]["evaluate"]
    assert eval_result["status"] == "ok"
    assert "models_evaluated" in eval_result


def test_dag_saves_results_json(mock_all_external_deps, tmp_path, monkeypatch):
    """Test DAG saves results to JSON file."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    # Create reports directory
    reports_dir = tmp_path / "ops" / "reports" / "dag_runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    cfg = NightlyConfig(
        asof="2024-06-15",
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    # Check result file exists
    result_file = reports_dir / "nightly_2024-06-15.json"
    assert result_file.exists()

    # Verify contents
    with open(result_file) as f:
        saved = json.load(f)

    assert saved["asof"] == "2024-06-15"
    assert saved["status"] == "ok"


def test_dag_handles_step_errors_gracefully(mock_all_external_deps):
    """Test DAG handles errors in individual steps."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    # Make audit raise an exception
    mock_all_external_deps["audit"].side_effect = Exception("Test error")

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=True,
        enable_backfill=False,
        enable_curate=True,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    # DAG should complete but mark as partial
    assert results["status"] in ("ok", "partial")
    assert "audit" in results["steps"]

    # Curate should still run
    assert "curate" in results["steps"]


def test_dag_step_sequence():
    """Test DAG steps run in correct sequence."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    call_order = []

    def track_audit(*args, **kwargs):
        call_order.append("audit")

    def track_curate(*args, **kwargs):
        call_order.append("curate")
        return {}

    with patch("ops.dags.nightly.audit_scan", track_audit), \
         patch("ops.dags.nightly.curate_incremental", track_curate), \
         patch("ops.dags.nightly.load_partition_status") as mock_status, \
         patch("ops.dags.nightly.get_green_coverage") as mock_green:

        mock_status.return_value = pd.DataFrame()
        mock_green.return_value = (1.0, {})

        cfg = NightlyConfig(
            enable_ingest=False,
            enable_audit=True,
            enable_backfill=False,
            enable_curate=True,
            enable_train_equities=False,
            enable_train_options=False,
            enable_train_intraday=False,
            enable_evaluate=False,
            enable_promote=False,
            enable_report=False,
        )

        run_nightly_dag(cfg)

    # Audit should run before curate
    assert call_order.index("audit") < call_order.index("curate")


def test_dag_with_custom_universes(mock_all_external_deps):
    """Test DAG respects custom universe settings."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        equities_universe=["SPY", "QQQ", "IWM"],
        options_universe=["AAPL", "MSFT"],
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert results["status"] == "ok"


def test_dag_backfill_with_no_gaps(mock_all_external_deps):
    """Test backfill step with no gaps found."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    # Return empty gaps
    mock_all_external_deps["load_status"].return_value = pd.DataFrame({
        "source_name": ["alpaca"],
        "table_name": ["equities_eod"],
        "symbol": ["AAPL"],
        "dt": [date.today()],
        "status": ["GREEN"],
    })

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=True,
        enable_backfill=True,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert "backfill" in results["steps"]
    # With no gaps, backfill should be skipped or succeed quickly
    assert results["steps"]["backfill"]["status"] in ("ok", "skipped")


def test_dag_config_lookback_days():
    """Test DAG uses lookback_days config correctly."""
    from ops.dags.nightly import NightlyConfig

    cfg = NightlyConfig(lookback_days=500)
    assert cfg.lookback_days == 500

    cfg_default = NightlyConfig()
    assert cfg_default.lookback_days == 252  # Default


def test_dag_timestamps_recorded():
    """Test DAG records start and finish timestamps."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    cfg = NightlyConfig(
        enable_ingest=False,
        enable_audit=False,
        enable_backfill=False,
        enable_curate=False,
        enable_train_equities=False,
        enable_train_options=False,
        enable_train_intraday=False,
        enable_evaluate=False,
        enable_promote=False,
        enable_report=False,
    )

    results = run_nightly_dag(cfg)

    assert "started_at" in results
    assert "finished_at" in results
    assert results["started_at"] < results["finished_at"]
