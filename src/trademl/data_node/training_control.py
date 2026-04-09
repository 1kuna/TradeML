"""Training readiness and off-node launch helpers for model runs."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.planner import training_readiness

LOGGER = logging.getLogger(__name__)


def phase_freeze_state_path(*, data_root: Path, phase: int = 1) -> Path:
    """Return the persisted freeze-cutoff state path for a training phase."""
    return data_root / "control" / "state" / f"phase{phase}_freeze.json"


def read_pinned_phase_freeze(*, data_root: Path, phase: int = 1) -> dict[str, Any] | None:
    """Return the persisted phase freeze payload when present."""
    path = phase_freeze_state_path(data_root=data_root, phase=phase)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("invalid_phase_freeze_json path=%s error=%s", path, exc)
        return None
    if not isinstance(payload, dict) or not payload.get("date"):
        return None
    return payload


def write_pinned_phase_freeze(*, data_root: Path, phase: int = 1, payload: dict[str, Any]) -> Path:
    """Persist the phase freeze payload to NAS-backed control state."""
    path = phase_freeze_state_path(data_root=data_root, phase=phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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
    except json.JSONDecodeError as exc:
        LOGGER.warning("invalid_training_runtime_json path=%s error=%s", path, exc)
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


def evaluate_training_gates(
    *,
    data_root: Path,
    stage_symbol_count: int,
    stage_years: int,
    planner_db_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Phase 1 and Phase 2 readiness from current NAS-backed artifacts."""
    reference_root = data_root / "data" / "reference"
    reference_files = {path.name for path in reference_root.glob("*.parquet")} if reference_root.exists() else set()
    macro_root = data_root / "data" / "raw" / "macros_fred"
    macro_series = {path.name.partition("=")[2] for path in macro_root.glob("series=*")} if macro_root.exists() else set()
    qc_path = data_root / "data" / "qc" / "partition_status.parquet"
    raw_green_ratio = _raw_green_ratio(qc_path)
    resolved_planner_db = planner_db_path or (data_root / "control" / "node.sqlite")
    planner_ratio = _planner_bars_ratio(resolved_planner_db)
    freeze_cutoff = recommended_training_cutoff(data_root=data_root, expected_symbol_count=stage_symbol_count)
    frozen_window = _frozen_window_bar_coverage(qc_path=qc_path, report_date=freeze_cutoff.get("date"))
    planner_window_ratio = _planner_window_ratio(
        db_path=resolved_planner_db,
        window_start=frozen_window.get("window_start"),
        window_end=frozen_window.get("window_end"),
    )
    effective_window_ratio = max(
        float(frozen_window.get("coverage_ratio", 0.0) or 0.0),
        float(planner_window_ratio or 0.0),
    )
    freeze_cutoff = {
        **freeze_cutoff,
        "window_start": frozen_window.get("window_start"),
        "window_end": frozen_window.get("window_end"),
        "window_coverage_ratio": frozen_window.get("coverage_ratio"),
        "planner_window_coverage_ratio": planner_window_ratio,
        "effective_window_coverage_ratio": effective_window_ratio,
        "window_missing_dates": frozen_window.get("missing_dates"),
    }
    bars_ratio = effective_window_ratio
    phase1 = training_readiness(
        raw_green_ratio=bars_ratio,
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
        "freeze_cutoff": freeze_cutoff,
    }


def recommended_training_cutoff(
    *,
    data_root: Path,
    expected_symbol_count: int,
    max_partitions: int = 120,
    lag_days: int = 30,
    as_of: date | str | None = None,
    phase: int = 1,
    pin_if_available: bool = True,
) -> dict[str, Any]:
    """Return the latest complete canonical date at or before the lagged training freeze window."""
    pinned = read_pinned_phase_freeze(data_root=data_root, phase=phase)
    if pinned is not None:
        return {
            **pinned,
            "pinned": True,
            "pin_path": str(phase_freeze_state_path(data_root=data_root, phase=phase)),
        }

    raw_root = data_root / "data" / "raw" / "equities_bars"
    anchor = pd.Timestamp(as_of or date.today().isoformat()).normalize() - pd.Timedelta(days=lag_days)
    if expected_symbol_count <= 0 or not raw_root.exists():
        return {
            "date": None,
            "complete_symbols": 0,
            "expected_symbols": expected_symbol_count,
            "coverage_ratio": 0.0,
            "lag_days": lag_days,
            "anchor_date": anchor.date().isoformat(),
        }
    best_partial_date: str | None = None
    best_partial_symbols = 0
    raw_files = sorted(raw_root.glob("date=*/data.parquet"), reverse=True)[:max_partitions]
    for path in raw_files:
        partition_date = path.parent.name.partition("=")[2]
        partition_ts = pd.Timestamp(partition_date)
        if partition_ts > anchor:
            continue
        try:
            frame = pd.read_parquet(path, columns=["symbol"])
        except Exception:
            continue
        complete_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns else int(len(frame))
        if complete_symbols >= expected_symbol_count:
            cutoff = {
                "date": partition_date,
                "complete_symbols": complete_symbols,
                "expected_symbols": expected_symbol_count,
                "coverage_ratio": min(1.0, complete_symbols / expected_symbol_count) if expected_symbol_count else 0.0,
                "lag_days": lag_days,
                "anchor_date": anchor.date().isoformat(),
                "pinned": False,
            }
            if pin_if_available and phase == 1:
                pinned_payload = {
                    **cutoff,
                    "phase": phase,
                    "pinned": True,
                    "pinned_at": datetime.now(tz=UTC).isoformat(),
                }
                write_pinned_phase_freeze(data_root=data_root, phase=phase, payload=pinned_payload)
                return {
                    **pinned_payload,
                    "pin_path": str(phase_freeze_state_path(data_root=data_root, phase=phase)),
                }
            return cutoff
        if complete_symbols > best_partial_symbols:
            best_partial_symbols = complete_symbols
            best_partial_date = partition_date
    return {
        "date": None,
        "complete_symbols": best_partial_symbols,
        "expected_symbols": expected_symbol_count,
        "coverage_ratio": min(1.0, best_partial_symbols / expected_symbol_count) if expected_symbol_count else 0.0,
        "best_partial_date": best_partial_date,
        "lag_days": lag_days,
        "anchor_date": anchor.date().isoformat(),
        "pinned": False,
    }


def _frozen_window_bar_coverage(
    *,
    qc_path: Path,
    report_date: str | None,
    window_years: int = 5,
    source: str = "alpaca",
) -> dict[str, Any]:
    if report_date is None or not qc_path.exists():
        return {
            "coverage_ratio": 0.0,
            "missing_dates": [],
            "window_start": None,
            "window_end": report_date,
        }
    frame = pd.read_parquet(qc_path)
    if frame.empty:
        return {
            "coverage_ratio": 0.0,
            "missing_dates": [],
            "window_start": None,
            "window_end": report_date,
        }
    frame["date"] = pd.to_datetime(frame["date"])
    end_date = pd.Timestamp(report_date)
    frame = frame.loc[
        (frame["dataset"] == "equities_eod")
        & (frame["source"] == source)
        & (frame["date"] <= end_date)
    ].copy()
    window_start = end_date - pd.DateOffset(years=window_years)
    frame = frame.loc[frame["date"].between(window_start, end_date)].copy()
    if frame.empty:
        return {
            "coverage_ratio": 0.0,
            "missing_dates": [],
            "window_start": window_start.date().isoformat(),
            "window_end": end_date.date().isoformat(),
        }
    missing_dates = sorted(frame.loc[frame["status"] != "GREEN", "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    return {
        "coverage_ratio": float((frame["status"] == "GREEN").mean()),
        "missing_dates": missing_dates,
        "window_start": window_start.date().isoformat(),
        "window_end": end_date.date().isoformat(),
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
    resolved_report_date = report_date
    if resolved_report_date is None:
        freeze_cutoff = recommended_training_cutoff(data_root=data_root, expected_symbol_count=_stage_symbol_count(data_root))
        resolved_report_date = freeze_cutoff.get("date") or date.today().isoformat()

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
        "report_date": resolved_report_date,
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
        resolved_report_date,
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


def _planner_bars_ratio(db_path: Path) -> float | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path, timeout=5.0) as connection:
            row = connection.execute(
                """
                SELECT SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                WHERE planner_tasks.task_family = 'canonical_bars'
                """
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    expected_units = int(row[0] or 0)
    completed_units = int(row[1] or 0)
    if expected_units <= 0:
        return None
    return min(1.0, completed_units / expected_units)


def _planner_window_ratio(
    *,
    db_path: Path,
    window_start: str | None,
    window_end: str | None,
) -> float | None:
    if not db_path.exists() or not window_start or not window_end:
        return None
    try:
        with sqlite3.connect(db_path, timeout=5.0) as connection:
            row = connection.execute(
                """
                SELECT SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                WHERE planner_tasks.task_family = 'canonical_bars'
                  AND planner_tasks.start_date >= ?
                  AND planner_tasks.end_date <= ?
                """,
                (window_start, window_end),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    expected_units = int(row[0] or 0)
    completed_units = int(row[1] or 0)
    if expected_units <= 0:
        return None
    return min(1.0, completed_units / expected_units)


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


def _stage_symbol_count(data_root: Path) -> int:
    stage_path = data_root / "stage.yml"
    if not stage_path.exists():
        return 0
    payload = _read_yaml(stage_path)
    return len(payload.get("symbols", []))
