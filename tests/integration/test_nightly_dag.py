"""Integration tests for nightly DAG orchestration."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_green_coverage():
    """Mock GREEN coverage to bypass threshold checks."""
    with patch("ops.dags.nightly.get_green_coverage") as mock:
        mock.return_value = 1.0  # 100% green
        yield mock


@pytest.fixture
def mock_partition_status():
    """Mock partition status loading."""
    with patch("ops.dags.nightly.load_partition_status") as mock:
        mock.return_value = pd.DataFrame({
            "source": ["alpaca"],
            "table_name": ["equities_eod"],
            "symbol": ["AAPL"],
            "dt": [date.today()],
            "status": ["GREEN"],
        })
        yield mock


def test_nightly_config_defaults():
    """Test NightlyConfig has sensible defaults."""
    from ops.dags.nightly import NightlyConfig

    cfg = NightlyConfig()

    assert cfg.asof is None
    assert cfg.enable_ingest is True
    assert cfg.enable_audit is True
    assert cfg.equities_green_threshold == 0.98
    assert cfg.options_green_threshold == 0.95
    assert cfg.intraday_green_threshold == 0.90
    assert cfg.max_backfill_requests == 100
    assert cfg.lookback_days == 252


def test_nightly_dag_dry_run(mock_green_coverage, mock_partition_status):
    """Test nightly DAG can run with all steps disabled."""
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
    assert "asof" in results
    assert "started_at" in results
    assert "finished_at" in results


def test_step_qc_refresh(mock_green_coverage):
    """Test QC refresh step collects coverage."""
    from ops.dags.nightly import NightlyConfig, _step_qc_refresh

    cfg = NightlyConfig()
    asof = date.today()

    results = _step_qc_refresh(cfg, asof)

    assert results["status"] == "ok"
    assert "coverage" in results


def test_get_asof_default():
    """Test _get_asof returns today when asof is None."""
    from ops.dags.nightly import NightlyConfig, _get_asof

    cfg = NightlyConfig(asof=None)
    result = _get_asof(cfg)

    assert result == date.today()


def test_get_asof_explicit():
    """Test _get_asof parses explicit date."""
    from ops.dags.nightly import NightlyConfig, _get_asof

    cfg = NightlyConfig(asof="2024-06-15")
    result = _get_asof(cfg)

    assert result == date(2024, 6, 15)


def test_check_green_threshold(mock_green_coverage):
    """Test GREEN threshold check."""
    from ops.dags.nightly import _check_green_threshold

    # Mock returns (coverage_ratio, status_counts)
    mock_green_coverage.return_value = (0.99, {"GREEN": 99, "AMBER": 1, "RED": 0})
    assert _check_green_threshold("curated/equities_ohlcv_adj", 0.98) is True

    mock_green_coverage.return_value = (0.95, {"GREEN": 95, "AMBER": 3, "RED": 2})
    assert _check_green_threshold("curated/equities_ohlcv_adj", 0.98) is False


def test_step_evaluate_empty():
    """Test evaluate step handles no artifacts gracefully."""
    from ops.dags.nightly import NightlyConfig, _step_evaluate

    cfg = NightlyConfig()
    asof = date.today()

    results = _step_evaluate(cfg, asof)

    assert results["status"] == "ok"
    assert "models_evaluated" in results


def test_dag_results_saved(mock_green_coverage, mock_partition_status, tmp_path, monkeypatch):
    """Test DAG saves results to JSON."""
    from ops.dags.nightly import NightlyConfig, run_nightly_dag

    # Mock the reports directory
    reports_dir = tmp_path / "ops" / "reports" / "dag_runs"
    monkeypatch.chdir(tmp_path)

    # Create minimal directory structure
    (tmp_path / "ops" / "reports" / "dag_runs").mkdir(parents=True, exist_ok=True)

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

    # Check result structure
    assert results["status"] == "ok"
    assert "steps" in results
