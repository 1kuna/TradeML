from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_training_job_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "scripts" / "training_job.py"
    spec = importlib.util.spec_from_file_location("training_job_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    train_stub = types.ModuleType("train")
    train_stub.run_training = lambda **kwargs: {}
    prior = sys.modules.get("train")
    sys.modules["train"] = train_stub
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if prior is None:
            sys.modules.pop("train", None)
        else:
            sys.modules["train"] = prior


training_job = _load_training_job_module()


def test_prepare_runtime_for_start_clears_stale_terminal_fields() -> None:
    runtime = {
        "status": "failed",
        "running": False,
        "started_at": "2026-04-10T16:00:00+00:00",
        "finished_at": "2026-04-10T16:05:00+00:00",
        "error": "old failure",
        "traceback": "trace",
        "assessment": {"decision": "NO_GO"},
        "report_path": "/tmp/report.json",
    }

    payload = training_job._prepare_runtime_for_start(
        runtime,
        phase=1,
        report_date="2026-03-09",
        model_suite="ridge_only",
    )

    assert payload["status"] == "running"
    assert payload["running"] is True
    assert payload["phase"] == 1
    assert payload["report_date"] == "2026-03-09"
    assert payload["model_suite"] == "ridge_only"
    assert payload["started_at"] != runtime["started_at"]
    assert "finished_at" not in payload
    assert "error" not in payload
    assert "traceback" not in payload
    assert "assessment" not in payload
    assert "report_path" not in payload


def test_prepare_runtime_for_success_clears_failure_fields() -> None:
    runtime = {
        "status": "running",
        "running": True,
        "error": "old failure",
        "traceback": "trace",
    }

    payload = training_job._prepare_runtime_for_success(
        runtime,
        output_root=Path("/tmp/out"),
        report_date="2026-03-09",
        assessment={"decision": "NO_GO"},
    )

    assert payload["status"] == "completed"
    assert payload["running"] is False
    assert payload["assessment"] == {"decision": "NO_GO"}
    assert payload["report_path"] == "/tmp/out/reports/daily/2026-03-09.json"
    assert "finished_at" in payload
    assert "error" not in payload
    assert "traceback" not in payload


def test_prepare_runtime_for_failure_clears_success_fields() -> None:
    runtime = {
        "status": "running",
        "running": True,
        "assessment": {"decision": "GO"},
        "report_path": "/tmp/out/reports/daily/2026-03-09.json",
    }

    payload = training_job._prepare_runtime_for_failure(
        runtime,
        error="boom",
        traceback_text="trace",
    )

    assert payload["status"] == "failed"
    assert payload["running"] is False
    assert payload["error"] == "boom"
    assert payload["traceback"] == "trace"
    assert "finished_at" in payload
    assert "assessment" not in payload
    assert "report_path" not in payload
