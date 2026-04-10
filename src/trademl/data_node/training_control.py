"""Training readiness and off-node launch helpers for model runs."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.db import DataNodeDB
from trademl.data_node.planner import training_readiness

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingTarget:
    """Resolved training execution target."""

    name: str
    kind: str
    default: bool
    host: str | None
    user: str | None
    port: int
    repo_root: Path
    data_root: Path
    python_executable: str
    env_path: Path | None
    identity_file: Path | None
    local_runtime_root: Path

    @property
    def label(self) -> str:
        """Return a compact display label for the target."""
        if self.kind == "local":
            return f"{self.name} ({_hostname()})"
        if self.host:
            return f"{self.name} ({self.host})"
        return self.name


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


def local_training_runtime_path(
    *,
    local_state: Path,
    phase: int,
    runtime_name: str | None = None,
    target_name: str | None = None,
) -> Path:
    """Return the controller-side runtime mirror path for a training run."""
    if runtime_name:
        target_segment = target_name or "local"
        return local_state / "training_runs" / target_segment / f"{runtime_name}.json"
    if target_name and target_name != "local":
        return local_state / "training_runs" / target_name / f"training_phase_{phase}.json"
    return local_state / f"training_phase_{phase}.json"


def shared_training_runtime_path(*, data_root: Path, phase: int, runtime_name: str | None = None) -> Path:
    """Return the NAS-visible runtime state path for a training phase or experiment run."""
    if runtime_name:
        return data_root / "control" / "cluster" / "state" / "training_runs" / f"{runtime_name}.json"
    return data_root / "control" / "cluster" / "state" / f"training_phase_{phase}.json"


def resolve_training_targets(
    *,
    targets_config_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    python_executable: str = sys.executable,
) -> dict[str, TrainingTarget]:
    """Resolve configured training targets plus the implicit local target."""
    config = _read_yaml(targets_config_path)
    training_settings = config.get("training", {}) if isinstance(config, dict) else {}
    configured_targets = config.get("training_targets", {}) if isinstance(config, dict) else {}
    targets: dict[str, TrainingTarget] = {
        "local": TrainingTarget(
            name="local",
            kind="local",
            default=False,
            host=_hostname(),
            user=None,
            port=22,
            repo_root=repo_root,
            data_root=data_root,
            python_executable=python_executable,
            env_path=None,
            identity_file=None,
            local_runtime_root=local_state,
        )
    }
    for name, payload in configured_targets.items():
        if not isinstance(payload, dict):
            continue
        kind = str(payload.get("kind") or "ssh").strip().lower()
        targets[str(name)] = TrainingTarget(
            name=str(name),
            kind=kind,
            default=False,
            host=_optional_target_value(payload.get("host")),
            user=_optional_target_value(payload.get("user")),
            port=int(payload.get("port") or 22),
            repo_root=Path(str(payload.get("repo_root") or repo_root)).expanduser(),
            data_root=Path(str(payload.get("data_root") or data_root)).expanduser(),
            python_executable=str(payload.get("python_executable") or python_executable),
            env_path=Path(str(payload["env_path"])).expanduser() if payload.get("env_path") else None,
            identity_file=Path(str(payload["identity_file"])).expanduser() if payload.get("identity_file") else None,
            local_runtime_root=Path(str(payload.get("local_runtime_root") or local_state)).expanduser(),
        )
    preferred_name = str(training_settings.get("default_target") or ("workstation-remote" if "workstation-remote" in targets else "local"))
    if preferred_name not in targets:
        preferred_name = "local"
    targets[preferred_name].default = True
    return targets


def resolve_training_target(
    *,
    target_name: str | None,
    targets_config_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    python_executable: str = sys.executable,
) -> TrainingTarget:
    """Resolve one training target by name, defaulting to the configured default."""
    targets = resolve_training_targets(
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    if target_name is None:
        for target in targets.values():
            if target.default:
                return target
        return targets["local"]
    if target_name not in targets:
        available = ", ".join(sorted(targets))
        raise ValueError(f"unknown training target {target_name!r}; available={available}")
    return targets[target_name]


def evaluate_training_gates(
    *,
    data_root: Path,
    stage_symbol_count: int,
    stage_years: int,
    planner_db_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Phase 1 and Phase 2 readiness from current NAS-backed artifacts."""
    reference_root = data_root / "data" / "reference"
    reference_files = _readable_reference_files(reference_root)
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
    effective_window_ratio = (
        float(planner_window_ratio)
        if planner_window_ratio is not None
        else float(frozen_window.get("coverage_ratio", 0.0) or 0.0)
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


def _readable_reference_files(reference_root: Path) -> set[str]:
    """Return only readable reference parquet filenames."""
    if not reference_root.exists():
        return set()
    readable: set[str] = set()
    for path in reference_root.glob("*.parquet"):
        try:
            pd.read_parquet(path, columns=[])
        except Exception:
            continue
        readable.add(path.name)
    return readable


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

    anchor = pd.Timestamp(as_of or date.today().isoformat()).normalize() - pd.Timedelta(days=lag_days)
    manifest_db = data_root / "control" / "node.sqlite"
    raw_root = data_root / "data" / "raw" / "equities_bars"
    if expected_symbol_count <= 0 or (not manifest_db.exists() and not raw_root.exists()):
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
    manifests = []
    if manifest_db.exists():
        try:
            manifests = DataNodeDB(manifest_db).fetch_raw_partition_manifests(dataset="equities_eod")
        except sqlite3.OperationalError:
            manifests = []
    if manifests:
        manifest_iter = (
            (manifest.trading_date, int(manifest.symbol_count), manifest.status == "HEALTHY")
            for manifest in sorted(manifests, key=lambda item: item.trading_date, reverse=True)[:max_partitions]
        )
    else:
        raw_files = sorted(raw_root.glob("date=*/data.parquet"), reverse=True)[:max_partitions]
        manifest_iter = []
        for path in raw_files:
            partition_date = path.parent.name.partition("=")[2]
            try:
                frame = pd.read_parquet(path, columns=["symbol"])
            except Exception:
                continue
            complete_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns else int(len(frame))
            manifest_iter.append((partition_date, complete_symbols, True))
    for partition_date, complete_symbols, healthy in manifest_iter:
        partition_ts = pd.Timestamp(partition_date)
        if partition_ts > anchor:
            continue
        if not healthy:
            continue
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


def training_preflight(
    *,
    data_root: Path,
    config_path: Path,
    repo_root: Path | None = None,
    local_state: Path | None = None,
    targets_config_path: Path | None = None,
    target: str | None = None,
    python_executable: str = sys.executable,
    execution_config_path: Path | None = None,
) -> dict[str, Any]:
    """Run control-plane, target, and dataset preflight before launching long-running work."""
    repo_root = repo_root or config_path.parents[1]
    local_state = local_state or Path("~/.trademl-training").expanduser()
    targets_config_path = targets_config_path or (repo_root / "configs" / "node.yml")
    if not config_path.exists():
        return {"ok": False, "reason": f"missing config: {config_path}", "control": {"ok": False, "config_path": str(config_path)}}
    resolved_target = resolve_training_target(
        target_name=target,
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    control = {"ok": True, "config_path": str(config_path), "repo_root": str(repo_root)}
    effective_config_path = execution_config_path or config_path
    target_report = _target_preflight(target=resolved_target, config_path=effective_config_path)
    dataset_report = _dataset_preflight(target=resolved_target, config_path=effective_config_path)
    ok = bool(control["ok"] and target_report.get("ok") and dataset_report.get("ok"))
    reason = None
    for report in (target_report, dataset_report):
        if not report.get("ok", False):
            reason = str(report.get("reason"))
            break
    payload = {
        "ok": ok,
        "reason": reason,
        "control": control,
        "target": target_report,
        "dataset": dataset_report,
        "resolved_target": asdict(resolved_target),
    }
    if dataset_report.get("sample_rows") is not None:
        payload["sample_rows"] = int(dataset_report["sample_rows"])
        payload["sample_date"] = dataset_report.get("sample_date")
        payload["qc_path"] = dataset_report.get("qc_path")
    return payload


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
    target: str | None = None,
    targets_config_path: Path | None = None,
    runtime_name: str | None = None,
    config_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Launch or resume the background training process for a phase."""
    targets_config_path = targets_config_path or (repo_root / "configs" / "node.yml")
    config_path = config_path or (repo_root / "configs" / "equities_xs.yml")
    resolved_target = resolve_training_target(
        target_name=target,
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    runtime_path = local_training_runtime_path(
        local_state=resolved_target.local_runtime_root,
        phase=phase,
        runtime_name=runtime_name,
        target_name=resolved_target.name,
    )
    existing = read_training_runtime(path=runtime_path)
    if existing.get("running"):
        return existing
    execution_config_path = config_path
    if resolved_target.kind != "local":
        execution_config_path = _sync_remote_training_config(
            target=resolved_target,
            local_config_path=config_path,
            runtime_name=runtime_name,
        )

    preflight = training_preflight(
        data_root=data_root,
        config_path=config_path,
        repo_root=repo_root,
        local_state=local_state,
        targets_config_path=targets_config_path,
        target=resolved_target.name,
        python_executable=python_executable,
        execution_config_path=execution_config_path,
    )
    if not preflight["ok"]:
        raise RuntimeError(f"phase {phase} training preflight failed: {preflight['reason']}")
    resolved_report_date = report_date
    if resolved_report_date is None:
        resolved_report_date = _resolve_default_report_date(target=resolved_target, phase=phase)
    resolved_output_root = output_root or resolved_target.data_root

    shared_runtime_path = shared_training_runtime_path(
        data_root=resolved_target.data_root,
        phase=phase,
        runtime_name=runtime_name,
    )
    log_path = _local_training_log_path(
        local_state=resolved_target.local_runtime_root,
        phase=phase,
        runtime_name=runtime_name,
        target_name=resolved_target.name,
    )
    remote_runtime_path = _remote_training_runtime_path(
        target=resolved_target,
        phase=phase,
        runtime_name=runtime_name,
    )
    remote_log_path = _remote_training_log_path(
        target=resolved_target,
        phase=phase,
        runtime_name=runtime_name,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_payload = {
        "phase": phase,
        "pid": None,
        "host": _hostname() if resolved_target.kind == "local" else resolved_target.host,
        "status": "starting",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "config_path": str(config_path),
        "execution_config_path": str(execution_config_path),
        "data_root": str(resolved_target.data_root),
        "output_root": str(resolved_output_root),
        "report_date": resolved_report_date,
        "model_suite": model_suite or ("ridge_only" if phase == 1 else "full"),
        "log_path": str(log_path),
        "remote_log_path": str(remote_log_path) if remote_log_path is not None else None,
        "remote_runtime_path": str(remote_runtime_path) if remote_runtime_path is not None else None,
        "target": resolved_target.name,
        "target_kind": resolved_target.kind,
        "shared_runtime_path": str(shared_runtime_path),
        "running": True,
        "preflight": preflight,
    }
    _write_runtime_payload(runtime_path, runtime_payload)
    if resolved_target.kind == "local":
        _write_runtime_payload(shared_runtime_path, runtime_payload)
        command = [
            python_executable,
            str(repo_root / "src" / "scripts" / "training_job.py"),
            "--data-root",
            str(resolved_target.data_root),
            "--config",
            str(config_path),
            "--output-root",
            str(resolved_output_root),
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

    remote_pid = _launch_remote_training_process(
        target=resolved_target,
        phase=phase,
        config_path=execution_config_path,
        report_date=resolved_report_date,
        model_suite=model_suite or ("ridge_only" if phase == 1 else "full"),
        shared_runtime_path=shared_runtime_path,
        runtime_name=runtime_name,
        output_root=resolved_output_root,
    )
    runtime_payload["pid"] = remote_pid
    runtime_payload["command"] = ["ssh", resolved_target.host or "", str(remote_log_path)]
    _write_runtime_payload(runtime_path, runtime_payload)
    return runtime_payload


def training_status_snapshot(
    *,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    phase: int,
    target: str | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
    runtime_name: str | None = None,
    tail_lines: int = 50,
) -> dict[str, Any]:
    """Return merged local/shared/remote runtime state for a training run."""
    targets_config_path = targets_config_path or (repo_root / "configs" / "node.yml")
    resolved_target = resolve_training_target(
        target_name=target,
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    local_path = local_training_runtime_path(
        local_state=resolved_target.local_runtime_root,
        phase=phase,
        runtime_name=runtime_name,
        target_name=resolved_target.name,
    )
    shared_path = shared_training_runtime_path(
        data_root=resolved_target.data_root,
        phase=phase,
        runtime_name=runtime_name,
    )
    local_runtime = read_training_runtime(path=local_path)
    shared_runtime = read_training_runtime(path=shared_path)
    if resolved_target.kind != "local" and not shared_runtime:
        shared_runtime = _remote_runtime_file_snapshot(target=resolved_target, path=shared_path)
    remote_runtime = _remote_runtime_snapshot(
        target=resolved_target,
        phase=phase,
        runtime_name=runtime_name,
        local_runtime=local_runtime,
        shared_runtime=shared_runtime,
    )
    effective = dict(local_runtime)
    if shared_runtime:
        effective.update(shared_runtime)
    if remote_runtime:
        effective.update({key: value for key, value in remote_runtime.items() if value is not None})
    if local_runtime.get("status") and not effective.get("status"):
        effective["status"] = local_runtime["status"]
    if "running" not in effective:
        effective["running"] = bool(local_runtime.get("running") or shared_runtime.get("running"))
    if resolved_target.kind != "local" and effective and effective != local_runtime:
        _write_runtime_payload(local_path, effective)
        local_runtime = read_training_runtime(path=local_path)
    log_tail = _training_log_tail(
        target=resolved_target,
        phase=phase,
        runtime_name=runtime_name,
        tail_lines=tail_lines,
        runtime=effective,
    )
    return {
        "target": asdict(resolved_target),
        "local": local_runtime,
        "shared": shared_runtime,
        "remote": remote_runtime,
        "runtime": effective,
        "log_tail": log_tail,
    }


def stop_training_process(
    *,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    phase: int,
    target: str | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
    runtime_name: str | None = None,
) -> dict[str, Any]:
    """Stop a detached training process and persist a stopped runtime state."""
    snapshot = training_status_snapshot(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        phase=phase,
        target=target,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        runtime_name=runtime_name,
        tail_lines=20,
    )
    runtime = dict(snapshot.get("runtime") or {})
    pid = runtime.get("pid")
    if not isinstance(pid, int):
        return {"stopped": False, "reason": "no active pid", **snapshot}
    resolved_target = TrainingTarget(**snapshot["target"])
    if resolved_target.kind == "local":
        os.kill(pid, signal.SIGTERM)
    else:
        _run_ssh_command(
            resolved_target,
            f"kill -TERM {pid}",
            check=True,
        )
    runtime.update(
        {
            "status": "stopped",
            "running": False,
            "finished_at": datetime.now(tz=UTC).isoformat(),
        }
    )
    local_path = local_training_runtime_path(
        local_state=resolved_target.local_runtime_root,
        phase=phase,
        runtime_name=runtime_name,
        target_name=resolved_target.name,
    )
    _write_runtime_payload(local_path, runtime)
    shared_path = Path(str(runtime.get("shared_runtime_path") or shared_training_runtime_path(data_root=resolved_target.data_root, phase=phase, runtime_name=runtime_name)))
    if resolved_target.kind == "local":
        _write_runtime_payload(shared_path, runtime)
    else:
        remote_runtime_path = Path(
            str(
                runtime.get("remote_runtime_path")
                or _remote_training_runtime_path(target=resolved_target, phase=phase, runtime_name=runtime_name)
            )
        )
        _write_remote_runtime_payload(resolved_target, remote_runtime_path, runtime)
        _write_remote_runtime_payload(resolved_target, shared_path, runtime)
    return {"stopped": True, "runtime": runtime, "target": snapshot["target"], "log_tail": snapshot["log_tail"]}


def training_log_tail(
    *,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    phase: int,
    target: str | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
    runtime_name: str | None = None,
    tail_lines: int = 50,
) -> dict[str, Any]:
    """Return the current runtime plus the latest training log tail."""
    return training_status_snapshot(
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        phase=phase,
        target=target,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        runtime_name=runtime_name,
        tail_lines=tail_lines,
    )


def _dataset_preflight(*, target: TrainingTarget, config_path: Path) -> dict[str, Any]:
    """Run dataset-level preflight on the resolved execution target."""
    if target.kind == "local":
        return _local_dataset_preflight(data_root=target.data_root, config_path=config_path)
    command = _remote_python_here_doc(
        target,
        f"""
import json
from pathlib import Path
from trademl.data_node.training_control import _local_dataset_preflight
print(json.dumps(_local_dataset_preflight(data_root=Path({target.data_root.as_posix()!r}), config_path=Path({config_path.as_posix()!r}))))
""",
    )
    result = _run_ssh_command(target, command)
    if result.returncode != 0:
        return {"ok": False, "reason": result.stderr.strip() or result.stdout.strip() or "remote dataset preflight failed"}
    return json.loads(result.stdout.strip() or "{}")


def _local_dataset_preflight(*, data_root: Path, config_path: Path) -> dict[str, Any]:
    """Run the existing local dataset preflight checks."""
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


def _target_preflight(*, target: TrainingTarget, config_path: Path) -> dict[str, Any]:
    """Verify that the requested training target is reachable and correctly configured."""
    if target.kind == "local":
        return {
            "ok": bool(config_path.exists()),
            "target": target.name,
            "kind": target.kind,
            "host": _hostname(),
            "repo_root": str(target.repo_root),
            "data_root": str(target.data_root),
            "python_executable": target.python_executable,
            "reason": None if config_path.exists() else f"missing config: {config_path}",
        }
    if not target.host or not target.user:
        return {"ok": False, "target": target.name, "kind": target.kind, "reason": "ssh target requires host and user"}
    command = " && ".join(
        [
            f"test -d {shlex.quote(str(target.repo_root))}",
            f"test -d {shlex.quote(str(target.data_root))}",
            f"test -f {shlex.quote(str(config_path))}",
            f"{shlex.quote(target.python_executable)} -V",
        ]
    )
    result = _run_ssh_command(target, command)
    return {
        "ok": result.returncode == 0,
        "target": target.name,
        "kind": target.kind,
        "host": target.host,
        "repo_root": str(target.repo_root),
        "data_root": str(target.data_root),
        "python_executable": target.python_executable,
        "reason": None if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip() or "remote target preflight failed"),
    }


def _launch_remote_training_process(
    *,
    target: TrainingTarget,
    phase: int,
    config_path: Path,
    report_date: str,
    model_suite: str,
    shared_runtime_path: Path,
    runtime_name: str | None,
    output_root: Path,
) -> int:
    """Launch a detached training job on a remote SSH target."""
    remote_runtime_path = _remote_training_runtime_path(target=target, phase=phase, runtime_name=runtime_name)
    remote_log_path = _remote_training_log_path(target=target, phase=phase, runtime_name=runtime_name)
    assert remote_runtime_path is not None
    assert remote_log_path is not None
    command = " && ".join(
        [
            f"mkdir -p {shlex.quote(str(remote_runtime_path.parent))}",
            f"mkdir -p {shlex.quote(str(remote_log_path.parent))}",
            f"cd {shlex.quote(str(target.repo_root))}",
            " ".join(
                [
                    "nohup",
                    shlex.quote(target.python_executable),
                    "src/scripts/training_job.py",
                    "--data-root",
                    shlex.quote(str(target.data_root)),
                    "--config",
                    shlex.quote(str(config_path)),
                    "--output-root",
                    shlex.quote(str(output_root)),
                    "--report-date",
                    shlex.quote(report_date),
                    "--model-suite",
                    shlex.quote(model_suite),
                    "--phase",
                    shlex.quote(str(phase)),
                    "--local-runtime-path",
                    shlex.quote(str(remote_runtime_path)),
                    "--shared-runtime-path",
                    shlex.quote(str(shared_runtime_path)),
                    f">>{shlex.quote(str(remote_log_path))}",
                    "2>&1",
                    "</dev/null",
                    "&",
                    "echo",
                    "$!",
                ]
            ),
        ]
    )
    result = _run_ssh_command(target, command, check=True)
    pid = int(str(result.stdout).strip().splitlines()[-1])
    return pid


def _remote_runtime_snapshot(
    *,
    target: TrainingTarget,
    phase: int,
    runtime_name: str | None,
    local_runtime: dict[str, Any],
    shared_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Read remote runtime state when the target is SSH-backed."""
    if target.kind == "local":
        return {}
    remote_runtime_path = _remote_training_runtime_path(target=target, phase=phase, runtime_name=runtime_name)
    if remote_runtime_path is None:
        return {}
    payload = _remote_runtime_file_snapshot(target=target, path=remote_runtime_path)
    if not payload:
        payload.update(shared_runtime or local_runtime)
    return payload


def _remote_runtime_file_snapshot(*, target: TrainingTarget, path: Path) -> dict[str, Any]:
    """Read one runtime payload from a remote SSH target."""
    command = _remote_python_here_doc(
        target,
        f"""
import json
from pathlib import Path
from trademl.data_node.training_control import read_training_runtime
payload = read_training_runtime(path=Path({path.as_posix()!r}))
print(json.dumps(payload))
""",
    )
    result = _run_ssh_command(target, command)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {}


def _write_remote_runtime_payload(target: TrainingTarget, path: Path, payload: dict[str, Any]) -> None:
    """Write one runtime payload onto a remote SSH target."""
    command = _remote_python_here_doc(
        target,
        f"""
import json
from pathlib import Path
from trademl.data_node.training_control import _write_runtime_payload
_write_runtime_payload(Path({path.as_posix()!r}), json.loads({json.dumps(_json_safe_payload(payload))!r}))
""",
    )
    _run_ssh_command(target, command, check=True)


def _training_log_tail(
    *,
    target: TrainingTarget,
    phase: int,
    runtime_name: str | None,
    tail_lines: int,
    runtime: dict[str, Any],
) -> str:
    """Read the latest training log lines from the correct execution target."""
    if target.kind == "local":
        log_path = Path(str(runtime.get("log_path") or _local_training_log_path(local_state=target.local_runtime_root, phase=phase, runtime_name=runtime_name, target_name=target.name)))
        if not log_path.exists():
            return ""
        return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail_lines:])
    remote_log_path = Path(str(runtime.get("remote_log_path") or _remote_training_log_path(target=target, phase=phase, runtime_name=runtime_name)))
    result = _run_ssh_command(target, f"tail -n {int(tail_lines)} {shlex.quote(str(remote_log_path))}")
    if result.returncode != 0:
        return result.stderr.strip() or ""
    return result.stdout


def _local_training_log_path(*, local_state: Path, phase: int, runtime_name: str | None, target_name: str) -> Path:
    """Return the controller-side training log path."""
    if runtime_name:
        return local_state / "logs" / "training_runs" / target_name / f"{runtime_name}.log"
    return local_state / "logs" / f"training_phase_{phase}.log"


def _remote_training_runtime_path(*, target: TrainingTarget, phase: int, runtime_name: str | None) -> Path | None:
    """Return the remote runtime path for a target-backed training run."""
    if target.kind == "local":
        return None
    if runtime_name:
        return target.repo_root / "control" / "training_runs" / f"{runtime_name}.json"
    return target.repo_root / "control" / f"training_phase_{phase}.json"


def _remote_training_log_path(*, target: TrainingTarget, phase: int, runtime_name: str | None) -> Path | None:
    """Return the remote log path for a target-backed training run."""
    if target.kind == "local":
        return None
    if runtime_name:
        return target.repo_root / "control" / "logs" / "training_runs" / f"{runtime_name}.log"
    return target.repo_root / "control" / "logs" / f"training_phase_{phase}.log"


def _remote_training_config_path(*, target: TrainingTarget, local_config_path: Path, runtime_name: str | None) -> Path:
    """Return the remote config path used for one target-backed run."""
    if runtime_name:
        return target.repo_root / "control" / "configs" / f"{runtime_name}.yml"
    return target.repo_root / "control" / "configs" / local_config_path.name


def _sync_remote_training_config(*, target: TrainingTarget, local_config_path: Path, runtime_name: str | None) -> Path:
    """Copy the exact local config file to the remote training target."""
    remote_config_path = _remote_training_config_path(
        target=target,
        local_config_path=local_config_path,
        runtime_name=runtime_name,
    )
    _copy_file_to_remote(target=target, local_path=local_config_path, remote_path=remote_config_path)
    return remote_config_path


def _remote_python_here_doc(target: TrainingTarget, body: str) -> str:
    """Wrap a Python here-doc for remote execution."""
    return " && ".join(
        [
            f"cd {shlex.quote(str(target.repo_root))}",
            f"PYTHONPATH=src {shlex.quote(target.python_executable)} - <<'PY'\n{body.strip()}\nPY",
        ]
    )


def _run_ssh_command(target: TrainingTarget, command: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one remote shell command against an SSH training target."""
    ssh_command = ["ssh", "-p", str(target.port)]
    if target.identity_file is not None:
        ssh_command.extend(["-i", str(target.identity_file)])
    ssh_command.append(f"{target.user}@{target.host}")
    ssh_command.append(command)
    result = subprocess.run(  # noqa: S603
        ssh_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"ssh command failed for target={target.name}")
    return result


def _copy_file_to_remote(*, target: TrainingTarget, local_path: Path, remote_path: Path) -> None:
    """Copy one local file onto the remote SSH target."""
    _run_ssh_command(target, f"mkdir -p {shlex.quote(str(remote_path.parent))}", check=True)
    scp_command = ["scp", "-P", str(target.port)]
    if target.identity_file is not None:
        scp_command.extend(["-i", str(target.identity_file)])
    scp_command.extend([str(local_path), f"{target.user}@{target.host}:{remote_path}"])
    result = subprocess.run(  # noqa: S603
        scp_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"scp failed for target={target.name}")


def _resolve_default_report_date(*, target: TrainingTarget, phase: int) -> str:
    """Return the default report date for one execution target."""
    if target.kind == "local":
        freeze_cutoff = recommended_training_cutoff(
            data_root=target.data_root,
            expected_symbol_count=_stage_symbol_count(target.data_root),
            phase=phase,
        )
        return str(freeze_cutoff.get("date") or date.today().isoformat())
    command = _remote_python_here_doc(
        target,
        f"""
import json
from pathlib import Path
from trademl.data_node.training_control import recommended_training_cutoff, _stage_symbol_count
data_root = Path({target.data_root.as_posix()!r})
payload = recommended_training_cutoff(
    data_root=data_root,
    expected_symbol_count=_stage_symbol_count(data_root),
    phase={phase},
)
print(json.dumps(payload))
""",
    )
    result = _run_ssh_command(target, command, check=True)
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid remote report-date payload for target={target.name}: {exc}") from exc
    return str(payload.get("date") or date.today().isoformat())


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
                  AND planner_tasks.planner_group = 'phase1_pinned_canonical'
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
                  AND planner_tasks.planner_group = 'phase1_pinned_canonical'
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


def _optional_target_value(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


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
    path.write_text(json.dumps(_json_safe_payload(payload), indent=2, sort_keys=True), encoding="utf-8")


def _hostname() -> str:
    return os.getenv("HOSTNAME") or os.uname().nodename


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(item) for item in value]
    return value


def _stage_symbol_count(data_root: Path) -> int:
    stage_path = data_root / "stage.yml"
    if not stage_path.exists():
        return 0
    payload = _read_yaml(stage_path)
    return len(payload.get("symbols", []))
