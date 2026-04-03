"""Training readiness and off-node launch helpers for model runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.planner import training_readiness


def read_training_runtime(
    *,
    path: Path | None = None,
    local_state: Path | None = None,
    phase: int | None = None,
) -> dict[str, Any]:
    """Read persisted training runtime state and refresh local-process liveness when possible."""
    if path is None:
        if local_state is None or phase is None:
            raise ValueError("either path or local_state+phase must be provided")
        path = local_state / f"training_phase_{phase}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    pid = payload.get("pid")
    host = payload.get("host")
    status = str(payload.get("status", "")).lower()
    if isinstance(pid, int) and host == _hostname():
        payload["running"] = _is_process_running(pid)
        if status == "running" and not payload["running"]:
            payload["status"] = "unknown"
    else:
        payload["running"] = status in {"starting", "running"}
    return payload


def evaluate_training_gates(*, data_root: Path, stage_symbol_count: int, stage_years: int) -> dict[str, Any]:
    """Evaluate Phase 1 and Phase 2 readiness from current NAS-backed artifacts."""
    reference_root = data_root / "data" / "reference"
    reference_files = {path.name for path in reference_root.glob("*.parquet")} if reference_root.exists() else set()
    macro_root = data_root / "data" / "raw" / "macros_fred"
    macro_series = {path.name.partition("=")[2] for path in macro_root.glob("series=*")} if macro_root.exists() else set()
    phase1 = training_readiness(
        raw_green_ratio=_raw_green_ratio(data_root / "data" / "qc" / "partition_status.parquet"),
        has_corp_actions=("corp_actions.parquet" in reference_files) or {"dividends.parquet", "splits.parquet"}.issubset(reference_files),
        has_listing_history="listing_history.parquet" in reference_files,
        has_delistings="delistings.parquet" in reference_files,
        has_sec_filings="sec_filings.parquet" in reference_files,
        has_macro_vintages="fred_vintagedates.parquet" in reference_files,
        macro_series_count=len(macro_series),
        required_macro_series=len(default_macro_series()),
    )
    phase2_blockers = list(phase1["blockers"])
    if "ticker_changes.parquet" not in reference_files:
        phase2_blockers.append("ticker_changes")
    if stage_symbol_count < 500:
        phase2_blockers.append("expanded_universe")
    if stage_years < 10:
        phase2_blockers.append("history_depth")
    if not _terminal_delisting_returns_present(reference_root / "delistings.parquet"):
        phase2_blockers.append("terminal_delisting_returns")
    return {
        "phase1": phase1,
        "phase2": {"ready": not phase2_blockers, "blockers": phase2_blockers},
    }


def training_preflight(*, data_root: Path, config_path: Path) -> dict[str, Any]:
    """Run a lightweight training preflight before launching long-running work."""
    qc_path = data_root / "data" / "qc" / "partition_status.parquet"
    curated_root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    if not config_path.exists():
        return {"ok": False, "reason": f"missing config: {config_path}"}
    if not qc_path.exists():
        return {"ok": False, "reason": f"missing qc parquet: {qc_path}"}
    curated_files = sorted(curated_root.glob("date=*/data.parquet"))
    if not curated_files:
        return {"ok": False, "reason": f"no curated parquet files under {curated_root}"}
    sample = pd.read_parquet(curated_files[-1])
    if sample.empty:
        return {"ok": False, "reason": "latest curated partition is empty"}
    return {
        "ok": True,
        "sample_rows": int(len(sample)),
        "sample_date": curated_files[-1].parent.name.partition("=")[2],
        "qc_path": str(qc_path),
    }


def launch_training_process(
    *,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    phase: int,
    model_suite: str | None = None,
    python_executable: str = sys.executable,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Launch or resume the background training process for a phase."""
    existing = read_training_runtime(local_state=local_state, phase=phase)
    if existing.get("running"):
        return existing

    config_path = repo_root / "configs" / "equities_xs.yml"
    preflight = training_preflight(data_root=data_root, config_path=config_path)
    if not preflight["ok"]:
        raise RuntimeError(f"phase {phase} training preflight failed: {preflight['reason']}")

    log_path = local_state / "logs" / f"training_phase_{phase}.log"
    runtime_path = local_state / f"training_phase_{phase}.json"
    shared_runtime_path = shared_training_runtime_path(data_root=data_root, phase=phase)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_payload = {
        "phase": phase,
        "pid": None,
        "host": _hostname(),
        "status": "starting",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "config_path": str(config_path),
        "data_root": str(data_root),
        "report_date": report_date or date.today().isoformat(),
        "model_suite": model_suite or ("ridge_only" if phase == 1 else "full"),
        "log_path": str(log_path),
        "shared_runtime_path": str(shared_runtime_path),
        "running": True,
        "preflight": preflight,
    }
    _write_runtime_payload(runtime_path, runtime_payload)
    _write_runtime_payload(shared_runtime_path, runtime_payload)
    command = [
        python_executable,
        str(repo_root / "src" / "scripts" / "training_job.py"),
        "--data-root",
        str(data_root),
        "--config",
        str(config_path),
        "--output-root",
        str(data_root),
        "--report-date",
        report_date or date.today().isoformat(),
        "--model-suite",
        model_suite or ("ridge_only" if phase == 1 else "full"),
        "--phase",
        str(phase),
        "--local-runtime-path",
        str(runtime_path),
        "--shared-runtime-path",
        str(shared_runtime_path),
    ]
    env = os.environ.copy()
    env.update(_read_env_file(env_path))
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runtime_payload["pid"] = process.pid
    runtime_payload["command"] = command
    _write_runtime_payload(runtime_path, runtime_payload)
    _write_runtime_payload(shared_runtime_path, runtime_payload)
    return runtime_payload


def shared_training_runtime_path(*, data_root: Path, phase: int) -> Path:
    """Return the NAS-visible runtime state path for a training phase."""
    return data_root / "control" / "cluster" / "state" / f"training_phase_{phase}.json"


def _raw_green_ratio(qc_path: Path) -> float | None:
    if not qc_path.exists():
        return None
    frame = pd.read_parquet(qc_path)
    if frame.empty:
        return None
    equities = frame.loc[frame["dataset"] == "equities_eod"].copy()
    if equities.empty:
        return None
    return float((equities["status"] == "GREEN").mean())


def _terminal_delisting_returns_present(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    lowered = {column.lower() for column in frame.columns}
    return "delisteddate" in lowered or "delisted_date" in lowered


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_runtime_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _hostname() -> str:
    return os.getenv("HOSTNAME") or os.uname().nodename
