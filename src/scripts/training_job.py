"""Detached training job runner that persists shared NAS-visible runtime state."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from train import run_training


_TERMINAL_RUNTIME_KEYS = (
    "assessment",
    "error",
    "finished_at",
    "report_path",
    "traceback",
)


def main() -> int:
    """Run one detached training job and persist lifecycle state."""
    parser = argparse.ArgumentParser(description="Run one TradeML training job.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--model-suite", default="full", choices=["full", "ridge_only"])
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--local-runtime-path", required=True)
    parser.add_argument("--shared-runtime-path", required=True)
    args = parser.parse_args()

    local_runtime_path = Path(args.local_runtime_path)
    shared_runtime_path = Path(args.shared_runtime_path)
    runtime = _prepare_runtime_for_start(
        _read_runtime(local_runtime_path) or {},
        phase=args.phase,
        report_date=args.report_date,
        model_suite=args.model_suite,
    )
    _write_runtime(local_runtime_path, runtime)
    _write_runtime(shared_runtime_path, runtime)

    try:
        report = run_training(
            data_root=Path(args.data_root),
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            report_date=args.report_date,
            model_suite=args.model_suite,
        )
    except Exception as exc:
        runtime = _prepare_runtime_for_failure(
            runtime,
            error=str(exc),
            traceback_text=traceback.format_exc(limit=20),
        )
        _write_runtime(local_runtime_path, runtime)
        _write_runtime(shared_runtime_path, runtime)
        raise

    runtime = _prepare_runtime_for_success(
        runtime,
        output_root=Path(args.output_root),
        report_date=args.report_date,
        assessment=report.get("assessment", {}),
    )
    _write_runtime(local_runtime_path, runtime)
    _write_runtime(shared_runtime_path, runtime)
    return 0


def _read_runtime(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_runtime(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_runtime_for_start(
    runtime: dict,
    *,
    phase: int,
    report_date: str,
    model_suite: str,
) -> dict:
    payload = dict(runtime)
    for key in _TERMINAL_RUNTIME_KEYS:
        payload.pop(key, None)
    payload.update(
        {
            "phase": phase,
            "status": "running",
            "running": True,
            "started_at": datetime.now(tz=UTC).isoformat(),
            "report_date": report_date,
            "model_suite": model_suite,
        }
    )
    return payload


def _prepare_runtime_for_failure(
    runtime: dict,
    *,
    error: str,
    traceback_text: str,
) -> dict:
    payload = dict(runtime)
    payload.update(
        {
            "status": "failed",
            "running": False,
            "finished_at": datetime.now(tz=UTC).isoformat(),
            "error": error,
            "traceback": traceback_text,
        }
    )
    payload.pop("assessment", None)
    payload.pop("report_path", None)
    return payload


def _prepare_runtime_for_success(
    runtime: dict,
    *,
    output_root: Path,
    report_date: str,
    assessment: dict,
) -> dict:
    payload = dict(runtime)
    payload.update(
        {
            "status": "completed",
            "running": False,
            "finished_at": datetime.now(tz=UTC).isoformat(),
            "report_path": str(output_root / "reports" / "daily" / f"{report_date}.json"),
            "assessment": assessment,
        }
    )
    payload.pop("error", None)
    payload.pop("traceback", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
