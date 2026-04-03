"""Detached training job runner that persists shared NAS-visible runtime state."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from train import run_training


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
    runtime = _read_runtime(local_runtime_path) or {}
    runtime.update(
        {
            "phase": args.phase,
            "status": "running",
            "running": True,
            "started_at": runtime.get("started_at") or datetime.now(tz=UTC).isoformat(),
            "report_date": args.report_date,
            "model_suite": args.model_suite,
        }
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
        runtime.update(
            {
                "status": "failed",
                "running": False,
                "finished_at": datetime.now(tz=UTC).isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        _write_runtime(local_runtime_path, runtime)
        _write_runtime(shared_runtime_path, runtime)
        raise

    runtime.update(
        {
            "status": "completed",
            "running": False,
            "finished_at": datetime.now(tz=UTC).isoformat(),
            "report_path": str(Path(args.output_root) / "reports" / "daily" / f"{args.report_date}.json"),
            "assessment": report.get("assessment", {}),
        }
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


if __name__ == "__main__":
    raise SystemExit(main())
