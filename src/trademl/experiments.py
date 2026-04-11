"""Experiment planning, supervision, evaluation, backtests, and proposal helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from trademl.data_node.training_control import (
    _remote_python_here_doc,
    _resolve_default_report_date,
    _run_ssh_command,
    launch_training_process,
    resolve_training_target,
    resolve_training_targets,
    shared_training_runtime_path,
    training_status_snapshot,
)

SPECIAL_MATRIX_DIMENSIONS = {"model_suite", "architecture_family", "feature_family", "data_profile"}

ARCHITECTURE_FAMILY_PRESETS: dict[str, dict[str, Any]] = {
    "linear_baseline": {"model_suite": "ridge_only", "config_overrides": {}},
    "tree_challenger": {"model_suite": "full", "config_overrides": {}},
}

FEATURE_FAMILY_PRESETS: dict[str, dict[str, Any]] = {
    "price_core": {
        "config_overrides": {
            "features.liquidity.adv_dollar": [],
            "features.liquidity.amihud": [],
        }
    },
    "price_liquidity": {"config_overrides": {}},
    "price_short_horizon": {
        "config_overrides": {
            "features.price.momentum": [5, 20, 60],
            "features.price.reversal": [1, 5],
            "features.price.drawdown": [20],
            "features.volatility.realized": [20],
            "features.liquidity.adv_dollar": [],
            "features.liquidity.amihud": [],
        }
    },
}

DATA_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "phase1_default": {"config_overrides": {}},
    "phase1_short_window": {"config_overrides": {"data.window_years": 3}},
    "phase1_long_window": {"config_overrides": {"data.window_years": 7}},
}


def plan_experiment(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Materialize deterministic run manifests for an experiment spec."""
    spec = _load_spec(spec_path)
    base_config_path = Path(str(spec.get("base_config") or (repo_root / "configs" / "equities_xs.yml"))).expanduser()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    experiment_id = str(spec["experiment_id"])
    phase = int(spec.get("phase", 1))
    available_targets = resolve_training_targets(
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    default_target = next((target.name for target in available_targets.values() if target.default), "local")
    target_name = str(spec.get("target") or default_target)
    target = resolve_training_target(
        target_name=target_name,
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    report_date = _resolve_report_date(spec, target=target)
    matrix = _realize_matrix(spec.get("matrix") or {})
    max_concurrent = int(spec.get("max_concurrent", 1) or 1)
    local_root = _local_experiment_root(local_state, experiment_id)
    local_root.mkdir(parents=True, exist_ok=True)
    runs_dir = local_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = local_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for row in matrix:
        run_id = _run_id(row)
        runtime_name = f"experiment_{experiment_id}__{run_id}"
        run_config = _apply_overrides(base_config, row.get("config_overrides", {}))
        config_path = configs_dir / f"{run_id}.yml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        output_root = target.data_root / "experiments" / experiment_id / "runs" / run_id
        base_manifest = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "runtime_name": runtime_name,
            "phase": phase,
            "target": target.name,
            "target_kind": target.kind,
            "report_date": report_date,
            "model_suite": row.get("model_suite") or spec.get("model_suite") or ("ridge_only" if phase == 1 else "full"),
            "matrix_values": row["matrix_values"],
            "config_overrides": row.get("config_overrides", {}),
            "config_path": str(config_path),
            "config_hash": hashlib.sha1(config_path.read_bytes()).hexdigest(),
            "local_runtime_path": str(target.local_runtime_root / "training_runs" / target.name / f"{runtime_name}.json"),
            "shared_runtime_path": str(shared_training_runtime_path(data_root=target.data_root, phase=phase, runtime_name=runtime_name)),
            "output_root": str(output_root),
            "report_path": str(output_root / "reports" / "daily" / f"{report_date}.json"),
            "status": "PLANNED",
            "assessment": {},
            "evaluation_stage": "PLANNED",
            "gate_failures": [],
            "evaluation_paths": {},
            "backtest_paths": {},
            "backtest_status": "NOT_STARTED",
            "shortlisted": False,
            "supervisor_history": [],
            "retry_count": 0,
            "failure_kind": None,
            "last_error": None,
        }
        manifest_path = runs_dir / f"{run_id}.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        merged = {**base_manifest, **_preserved_manifest_state(existing)}
        manifest_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        runs.append(merged)

    summary = {
        "experiment_id": experiment_id,
        "phase": phase,
        "target": target_name,
        "report_date": report_date,
        "spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
        "predictive_gate": _predictive_gate(spec),
        "backtest_gate": _backtest_gate(spec),
        "backtest_profile": dict(spec.get("backtest_profile") or {}),
        "proposal_policy": _proposal_policy(spec),
        "supervision": _supervision_policy(spec),
        "max_concurrent": max_concurrent,
        "run_count": len(runs),
        "runs": [_summary_run_row(run) for run in runs],
    }
    _write_experiment_summary(local_state=local_state, experiment_id=experiment_id, summary=summary)
    return summary


def launch_experiment(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Launch pending experiment runs up to the configured concurrency limit."""
    summary = plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    experiment_id = str(summary["experiment_id"])
    spec = _load_spec(spec_path)
    supervision = _supervision_policy(spec)
    status = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    active = int(status["counts"].get("RUNNING", 0) + status["counts"].get("STARTING", 0))
    available_slots = max(0, int(summary.get("max_concurrent", 1)) - active)
    launched: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    if available_slots <= 0:
        return {**status, "launched": launched, "failed": failed}
    for manifest in _load_run_manifests(local_state=local_state, experiment_id=experiment_id):
        if available_slots <= 0:
            break
        if not _run_ready_for_retry(manifest=manifest, supervision=supervision):
            continue
        try:
            payload = launch_training_process(
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                phase=int(manifest["phase"]),
                model_suite=str(manifest["model_suite"]),
                python_executable=python_executable,
                report_date=str(manifest["report_date"]),
                target=str(manifest["target"]),
                targets_config_path=targets_config_path,
                runtime_name=str(manifest["runtime_name"]),
                config_path=Path(str(manifest["config_path"])),
                output_root=Path(str(manifest["output_root"])),
            )
        except Exception as exc:  # noqa: BLE001
            manifest["status"] = "FAILED"
            manifest["failure_kind"] = _classify_failure(str(exc))
            manifest["last_error"] = str(exc)
            manifest["retry_count"] = int(manifest.get("retry_count", 0) or 0) + 1
            _append_supervisor_event(manifest, event="launch_failed", payload={"error": str(exc), "failure_kind": manifest["failure_kind"]})
            _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
            failed.append({"run_id": manifest["run_id"], "failure_kind": manifest["failure_kind"], "error": str(exc)})
            continue
        manifest["status"] = str(payload.get("status", "STARTING")).upper()
        manifest["runtime"] = payload
        manifest["last_error"] = None
        manifest["failure_kind"] = None
        _append_supervisor_event(manifest, event="launched", payload={"pid": payload.get("pid"), "target": manifest["target"]})
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        launched.append({"run_id": manifest["run_id"], "target": manifest["target"], "status": manifest["status"]})
        available_slots -= 1
    refreshed = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    return {**refreshed, "launched": launched, "failed": failed}


def experiment_status(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Refresh and return status for all runs in an experiment."""
    local_root = _local_experiment_root(local_state, experiment_id)
    runs: list[dict[str, Any]] = []
    summary = _read_experiment_summary(local_state=local_state, experiment_id=experiment_id)
    for path in sorted((local_root / "runs").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        snapshot = training_status_snapshot(
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            phase=int(manifest["phase"]),
            target=str(manifest["target"]),
            targets_config_path=targets_config_path,
            python_executable=python_executable,
            runtime_name=str(manifest["runtime_name"]),
            tail_lines=20,
        )
        runtime = snapshot.get("runtime") or {}
        status = str(runtime.get("status") or manifest.get("status") or "PLANNED").upper()
        report = _load_report_payload(
            manifest=manifest,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        if report is not None:
            status = "COMPLETED"
            manifest["assessment"] = report.get("assessment", {})
            manifest["report_preview"] = {
                "coverage": report.get("coverage"),
                "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
                "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
            }
        elif status == "FAILED":
            manifest["last_error"] = runtime.get("error") or manifest.get("last_error")
            if not manifest.get("failure_kind"):
                manifest["failure_kind"] = _classify_failure(str(manifest.get("last_error") or "training runtime failed"))
        manifest["status"] = status
        manifest["runtime"] = runtime
        manifest["log_tail"] = snapshot.get("log_tail", "")
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        runs.append(manifest)
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    summary = {
        **summary,
        "experiment_id": experiment_id,
        "counts": counts,
        "run_count": len(runs),
        "runs": runs,
    }
    return _refresh_experiment_summary(local_state=local_state, experiment_id=experiment_id, summary=summary)


def evaluate_experiment(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Evaluate all completed runs against deterministic predictive gates."""
    summary = _read_experiment_summary(local_state=local_state, experiment_id=experiment_id)
    spec = _load_spec(Path(str(summary["spec_path"])))
    gate = _predictive_gate(spec)
    evaluated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for manifest in _load_run_manifests(local_state=local_state, experiment_id=experiment_id):
        if str(manifest.get("status")) != "COMPLETED":
            continue
        report = _load_report_payload(
            manifest=manifest,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        if report is None:
            continue
        try:
            evaluation = _evaluate_report(manifest=manifest, report=report, gate=gate)
        except Exception as exc:  # noqa: BLE001
            manifest["evaluation_stage"] = "EVALUATION_ERROR"
            manifest["failure_kind"] = "evaluation"
            manifest["last_error"] = str(exc)
            manifest["gate_failures"] = [str(exc)]
            errors.append({"run_id": manifest["run_id"], "error": str(exc)})
            _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
            continue
        evaluation_paths = _write_evaluation_artifacts(
            local_state=local_state,
            experiment_id=experiment_id,
            run_id=str(manifest["run_id"]),
            evaluation=evaluation,
        )
        manifest["evaluation_stage"] = evaluation["evaluation_stage"]
        manifest["gate_failures"] = evaluation["gate_failures"]
        manifest["evaluation_paths"] = evaluation_paths
        manifest["survived_predictive"] = evaluation["survived_predictive"]
        if evaluation["evaluation_stage"] == "REJECTED_PREDICTIVE":
            manifest["shortlisted"] = False
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        evaluated.append(
            {
                "run_id": manifest["run_id"],
                "evaluation_stage": evaluation["evaluation_stage"],
                "gate_failures": evaluation["gate_failures"],
            }
        )
    refreshed = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    return {**refreshed, "evaluated": evaluated, "evaluation_errors": errors}


def backtest_experiment_survivors(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Run deterministic backtests for predictive survivors and update shortlist state."""
    summary = _read_experiment_summary(local_state=local_state, experiment_id=experiment_id)
    spec = _load_spec(Path(str(summary["spec_path"])))
    gate = _backtest_gate(spec)
    completed: list[dict[str, Any]] = []
    for manifest in _load_run_manifests(local_state=local_state, experiment_id=experiment_id):
        if str(manifest.get("evaluation_stage")) != "SURVIVES_PREDICTIVE":
            continue
        if str(manifest.get("backtest_status")) == "COMPLETED":
            continue
        manifest["evaluation_stage"] = "BACKTEST_RUNNING"
        manifest["backtest_status"] = "RUNNING"
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        summary_payload = _run_backtest_for_manifest(
            manifest=manifest,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        outcome = _evaluate_backtest(summary_payload=summary_payload, gate=gate)
        manifest["backtest_status"] = "COMPLETED"
        manifest["backtest_paths"] = summary_payload.get("paths", {})
        manifest["backtest_summary"] = summary_payload
        manifest["evaluation_stage"] = "SHORTLISTED" if outcome["shortlisted"] else "REJECTED_BACKTEST"
        manifest["shortlisted"] = bool(outcome["shortlisted"])
        manifest["gate_failures"] = outcome["gate_failures"]
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        completed.append({"run_id": manifest["run_id"], "evaluation_stage": manifest["evaluation_stage"]})
    refreshed = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    return {**refreshed, "backtested": completed}


def propose_next_experiment_family(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Write a bounded next-family proposal from completed experiment outcomes."""
    summary = _read_experiment_summary(local_state=local_state, experiment_id=experiment_id)
    spec = _load_spec(Path(str(summary["spec_path"])))
    comparison = compare_experiment(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    proposal = _build_next_family_proposal(
        experiment_id=experiment_id,
        base_spec=spec,
        comparison=comparison,
    )
    proposal_paths = _write_next_family_proposal(local_state=local_state, experiment_id=experiment_id, proposal=proposal)
    refreshed = _refresh_experiment_summary(
        local_state=local_state,
        experiment_id=experiment_id,
        summary={**summary, "proposal_summary": proposal | proposal_paths},
    )
    return {**refreshed, "proposal": proposal | proposal_paths}


def supervise_experiment(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int | None = None,
    detach: bool = False,
) -> dict[str, Any]:
    """Run or spawn the bounded experiment supervisor."""
    plan = plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    experiment_id = str(plan["experiment_id"])
    spec = _load_spec(spec_path)
    supervision = _supervision_policy(spec)
    proposal_policy = _proposal_policy(spec)
    resolved_poll = int(poll_seconds or supervision["poll_seconds"])
    if detach:
        return _spawn_supervisor_process(
            experiment_id=experiment_id,
            spec_path=spec_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            python_executable=python_executable,
            poll_seconds=resolved_poll,
        )

    state = _ensure_supervisor_state(
        local_state=local_state,
        experiment_id=experiment_id,
        payload={
            "experiment_id": experiment_id,
            "spec_path": str(spec_path),
            "status": "RUNNING",
            "started_at": datetime.now(tz=UTC).isoformat(),
            "heartbeat_at": datetime.now(tz=UTC).isoformat(),
            "pid": None,
            "poll_seconds": resolved_poll,
            "last_error": None,
            "last_error_kind": None,
            "active_run_ids": [],
            "queue_counts": {},
            "paused": False,
            "stop_requested": False,
        },
    )
    state["pid"] = state.get("pid") or None
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
    while True:
        state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
        now = datetime.now(tz=UTC).isoformat()
        if not state:
            state = {"experiment_id": experiment_id}
        state["heartbeat_at"] = now
        if state.get("stop_requested"):
            state["status"] = "STOPPED"
            _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
            return state
        if state.get("paused"):
            state["status"] = "PAUSED"
            _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue

        status = experiment_status(
            experiment_id=experiment_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        state["status"] = "RUNNING"
        state["queue_counts"] = status.get("counts", {})
        state["active_run_ids"] = [run["run_id"] for run in status["runs"] if run["status"] in {"RUNNING", "STARTING"}]
        _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)

        evaluate_experiment(
            experiment_id=experiment_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        if supervision["auto_backtest_survivors"]:
            backtest_experiment_survivors(
                experiment_id=experiment_id,
                local_state=local_state,
                repo_root=repo_root,
                data_root=data_root,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
            )
        status = experiment_status(
            experiment_id=experiment_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        counts = status.get("counts", {})
        planned = int(counts.get("PLANNED", 0) + counts.get("FAILED", 0))
        active = int(counts.get("RUNNING", 0) + counts.get("STARTING", 0))
        if planned > 0 and active < int(plan.get("max_concurrent", 1)):
            launch_result = launch_experiment(
                spec_path=spec_path,
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
            )
            launch_errors = launch_result.get("failed", [])
            if launch_errors:
                state["last_error"] = launch_errors[-1].get("error")
                state["last_error_kind"] = launch_errors[-1].get("failure_kind")
            state["last_launch_at"] = datetime.now(tz=UTC).isoformat()
            _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
            continue
        if planned <= 0 and active <= 0:
            chained = None
            if supervision["auto_propose_next_family"]:
                proposal_result = propose_next_experiment_family(
                    experiment_id=experiment_id,
                    local_state=local_state,
                    repo_root=repo_root,
                    data_root=data_root,
                    targets_config_path=targets_config_path,
                    python_executable=python_executable,
                )
                proposal = proposal_result.get("proposal") or {}
                if proposal_policy["auto_launch_next_family"] and bool(proposal.get("chain_allowed")):
                    chained = _spawn_supervisor_process(
                        experiment_id=str(proposal["recommended_experiment_id"]),
                        spec_path=Path(str(proposal["spec_path"])),
                        repo_root=repo_root,
                        data_root=data_root,
                        local_state=local_state,
                        env_path=env_path,
                        python_executable=python_executable,
                        poll_seconds=int(resolved_poll),
                    )
                    state["next_experiment_id"] = proposal["recommended_experiment_id"]
                    state["next_supervisor_pid"] = chained["pid"]
            state["status"] = "COMPLETED"
            state["completed_at"] = datetime.now(tz=UTC).isoformat()
            if chained is not None:
                state["chained"] = chained
            _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
            return state
        time.sleep(max(1, resolved_poll))


def read_experiment_supervisor_state(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    """Return one experiment supervisor state payload when present."""
    path = _supervisor_state_path(local_state=local_state, experiment_id=experiment_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pause_experiment_supervisor(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    """Pause an active experiment supervisor."""
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    if not state:
        raise ValueError(f"no supervisor state for experiment {experiment_id!r}")
    state["paused"] = True
    state["status"] = "PAUSED"
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
    return state


def stop_experiment_supervisor(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    """Request supervisor shutdown and terminate the local process when present."""
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    if not state:
        raise ValueError(f"no supervisor state for experiment {experiment_id!r}")
    state["stop_requested"] = True
    state["status"] = "STOPPING"
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
    pid = state.get("pid")
    if isinstance(pid, int):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    return state


def resume_experiment_supervisor(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Resume a paused supervisor, spawning a new detached process if needed."""
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    if not state:
        raise ValueError(f"no supervisor state for experiment {experiment_id!r}")
    state["paused"] = False
    state["stop_requested"] = False
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
    pid = state.get("pid")
    if isinstance(pid, int) and _is_local_process_running(pid):
        state["status"] = "RUNNING"
        _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
        return state
    return supervise_experiment(
        spec_path=Path(str(state["spec_path"])),
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=int(state.get("poll_seconds") or 30),
        detach=True,
    )


def compare_experiment(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path | None = None,
    data_root: Path | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Build a deterministic comparison table for completed experiment runs."""
    rows: list[dict[str, Any]] = []
    for manifest in _load_run_manifests(local_state=local_state, experiment_id=experiment_id):
        report = _load_report_payload(
            manifest=manifest,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        if report is None:
            continue
        primary_score = _primary_rank_ic(manifest=manifest, report=report)
        backtest_summary = manifest.get("backtest_summary") or _load_backtest_summary(
            manifest=manifest,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        rows.append(
            {
                "run_id": manifest["run_id"],
                "target": manifest["target"],
                "model_suite": manifest["model_suite"],
                "matrix_values": manifest["matrix_values"],
                "coverage": report.get("coverage"),
                "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
                "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
                "pbo": report.get("diagnostics", {}).get("pbo"),
                "dsr": report.get("diagnostics", {}).get("dsr"),
                "decision": report.get("assessment", {}).get("decision"),
                "decision_reason": report.get("assessment", {}).get("reason"),
                "evaluation_stage": manifest.get("evaluation_stage"),
                "gate_failures": manifest.get("gate_failures", []),
                "survived_predictive": bool(manifest.get("survived_predictive")),
                "backtest_status": manifest.get("backtest_status", "NOT_STARTED"),
                "shortlisted": bool(manifest.get("shortlisted")),
                "backtest_net_return": (backtest_summary or {}).get("net_return"),
                "backtest_turnover": (backtest_summary or {}).get("turnover"),
                "report_path": str(manifest.get("report_path")),
                "primary_score": primary_score,
            }
        )
    rows.sort(key=lambda item: (bool(item.get("shortlisted")), float(item.get("primary_score") or 0.0), str(item["run_id"])), reverse=True)
    best = rows[0] if rows else None
    return {"experiment_id": experiment_id, "rows": rows, "best": best}


def run_experiment_until_idle(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int = 30,
) -> dict[str, Any]:
    """Continuously refill the experiment queue until no runs remain active or planned."""
    return supervise_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=poll_seconds,
        detach=False,
    )


def render_experiment_report(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path | None = None,
    data_root: Path | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Write experiment comparison JSON and markdown reports."""
    comparison = compare_experiment(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    local_root = _local_experiment_root(local_state, experiment_id)
    json_path = local_root / "comparison.json"
    md_path = local_root / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [f"# Experiment {experiment_id}", ""]
    best = comparison.get("best")
    if best:
        md_lines.extend([f"Best run: `{best['run_id']}`", ""])
    for row in comparison["rows"]:
        md_lines.extend(
            [
                f"## {row['run_id']}",
                f"- target: {row['target']}",
                f"- model_suite: {row['model_suite']}",
                f"- coverage: {row['coverage']}",
                f"- ridge_mean_rank_ic: {row['ridge_mean_rank_ic']}",
                f"- lightgbm_mean_rank_ic: {row['lightgbm_mean_rank_ic']}",
                f"- pbo: {row['pbo']}",
                f"- dsr: {row['dsr']}",
                f"- decision: {row['decision']}",
                f"- decision_reason: {row['decision_reason']}",
                f"- evaluation_stage: {row['evaluation_stage']}",
                f"- gate_failures: {row['gate_failures']}",
                f"- backtest_status: {row['backtest_status']}",
                f"- backtest_net_return: {row['backtest_net_return']}",
                f"- shortlisted: {row['shortlisted']}",
                "",
            ]
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"experiment_id": experiment_id, "json_path": str(json_path), "markdown_path": str(md_path), "best": best}


def latest_experiment_summary(*, local_state: Path) -> dict[str, Any]:
    """Return the most recent experiment summary when present."""
    root = local_state / "experiments"
    summaries = sorted(root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summaries:
        return {}
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def latest_experiment_proposal(*, local_state: Path) -> dict[str, Any]:
    """Return the most recent experiment proposal summary when present."""
    proposals = sorted((local_state / "experiments").glob("*/next_family_proposal.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not proposals:
        return {}
    return json.loads(proposals[0].read_text(encoding="utf-8"))


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not payload.get("experiment_id"):
        raise ValueError("experiment spec requires experiment_id")
    return payload


def _resolve_report_date(spec: dict[str, Any], *, target: Any) -> str:
    report_date = spec.get("report_date")
    if report_date:
        return str(report_date)
    policy = str(spec.get("report_date_policy") or "phase1_freeze")
    if policy != "phase1_freeze":
        raise ValueError(f"unsupported report_date_policy: {policy}")
    resolved = _resolve_default_report_date(target=target, phase=int(spec.get("phase", 1) or 1))
    if not resolved:
        raise ValueError("unable to resolve phase1_freeze report_date from current canonical freeze state")
    return str(resolved)


def _realize_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if not matrix:
        return [{"matrix_values": {}, "config_overrides": {}}]
    keys = sorted(matrix)
    values = []
    for key in keys:
        raw = matrix.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"matrix dimension {key!r} must be a non-empty list")
        values.append(raw)
    rows: list[dict[str, Any]] = []
    for combination in itertools.product(*values):
        matrix_values = {key: value for key, value in zip(keys, combination, strict=True)}
        rows.append(_materialize_matrix_row(matrix_values))
    return rows


def _materialize_matrix_row(matrix_values: dict[str, Any]) -> dict[str, Any]:
    config_overrides: dict[str, Any] = {}
    model_suite = matrix_values.get("model_suite")
    architecture_family = matrix_values.get("architecture_family")
    if architecture_family is not None:
        preset = ARCHITECTURE_FAMILY_PRESETS.get(str(architecture_family))
        if preset is None:
            raise ValueError(f"unsupported architecture_family: {architecture_family}")
        preset_model_suite = str(preset["model_suite"])
        if model_suite is not None and str(model_suite) != preset_model_suite:
            raise ValueError(f"matrix row conflicts between model_suite={model_suite!r} and architecture_family={architecture_family!r}")
        model_suite = preset_model_suite
        config_overrides.update(dict(preset.get("config_overrides") or {}))
    feature_family = matrix_values.get("feature_family")
    if feature_family is not None:
        preset = FEATURE_FAMILY_PRESETS.get(str(feature_family))
        if preset is None:
            raise ValueError(f"unsupported feature_family: {feature_family}")
        config_overrides.update(dict(preset.get("config_overrides") or {}))
    data_profile = matrix_values.get("data_profile")
    if data_profile is not None:
        preset = DATA_PROFILE_PRESETS.get(str(data_profile))
        if preset is None:
            raise ValueError(f"unsupported data_profile: {data_profile}")
        config_overrides.update(dict(preset.get("config_overrides") or {}))
    config_overrides.update({key: value for key, value in matrix_values.items() if key not in SPECIAL_MATRIX_DIMENSIONS})
    return {
        "matrix_values": matrix_values,
        "config_overrides": config_overrides,
        "model_suite": model_suite,
    }


def _apply_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    for dotted_path, value in overrides.items():
        cursor = payload
        parts = str(dotted_path).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return payload


def _predictive_gate(spec: dict[str, Any]) -> dict[str, Any]:
    acceptance = dict(spec.get("acceptance") or {})
    gate = dict(spec.get("predictive_gate") or {})
    return {
        "require_go_decision": bool(gate.get("require_go_decision", acceptance.get("require_go_decision", False))),
        "min_rank_ic": float(gate.get("min_rank_ic", acceptance.get("min_rank_ic", 0.0)) or 0.0),
        "require_all_years_positive": bool(gate.get("require_all_years_positive", True)),
        "max_abs_placebo_ic": float(gate.get("max_abs_placebo_ic", 0.10) or 0.10),
        "min_cost_stress_net_return": float(gate.get("min_cost_stress_net_return", 0.0) or 0.0),
        "max_pbo": float(gate["max_pbo"]) if gate.get("max_pbo") is not None else None,
        "min_dsr": float(gate["min_dsr"]) if gate.get("min_dsr") is not None else None,
        "min_coverage": float(gate.get("min_coverage", 0.0) or 0.0),
    }


def _backtest_gate(spec: dict[str, Any]) -> dict[str, Any]:
    gate = dict(spec.get("backtest_gate") or {})
    return {
        "min_net_return": float(gate.get("min_net_return", 0.0) or 0.0),
        "require_cost_positive": bool(gate.get("require_cost_positive", True)),
        "max_turnover": float(gate["max_turnover"]) if gate.get("max_turnover") is not None else None,
    }


def _proposal_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policy = dict(spec.get("proposal_policy") or {})
    return {
        "family_size_cap": int(policy.get("family_size_cap", 6) or 6),
        "allowed_dimensions": list(
            policy.get("allowed_dimensions")
            or ["architecture_family", "feature_family", "validation.initial_train_years", "data_profile"]
        ),
        "max_generations": int(policy.get("max_generations", 1) or 1),
        "auto_launch_next_family": bool(policy.get("auto_launch_next_family", False)),
    }


def _supervision_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policy = dict(spec.get("supervision") or {})
    return {
        "poll_seconds": int(policy.get("poll_seconds", 30) or 30),
        "auto_backtest_survivors": bool(policy.get("auto_backtest_survivors", True)),
        "auto_propose_next_family": bool(policy.get("auto_propose_next_family", True)),
        "max_retry_count": int(policy.get("max_retry_count", 2) or 2),
    }


def _run_id(matrix_values: dict[str, Any]) -> str:
    serialized = json.dumps(matrix_values, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


def _local_experiment_root(local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiments" / experiment_id


def _summary_run_row(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "phase": manifest.get("phase"),
        "target": manifest.get("target"),
        "target_kind": manifest.get("target_kind"),
        "report_date": manifest.get("report_date"),
        "status": manifest["status"],
        "model_suite": manifest["model_suite"],
        "matrix_values": manifest["matrix_values"],
        "config_overrides": manifest.get("config_overrides", {}),
        "config_path": manifest.get("config_path"),
        "runtime_name": manifest.get("runtime_name"),
        "local_runtime_path": manifest.get("local_runtime_path"),
        "shared_runtime_path": manifest.get("shared_runtime_path"),
        "output_root": manifest.get("output_root"),
        "report_path": manifest.get("report_path"),
        "evaluation_stage": manifest.get("evaluation_stage"),
        "failure_kind": manifest.get("failure_kind"),
        "last_error": manifest.get("last_error"),
        "retry_count": manifest.get("retry_count", 0),
        "supervisor_history": manifest.get("supervisor_history", []),
        "shortlisted": bool(manifest.get("shortlisted")),
    }


def _preserved_manifest_state(existing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: existing.get(key)
        for key in [
            "status",
            "assessment",
            "runtime",
            "log_tail",
            "report_preview",
            "evaluation_stage",
            "gate_failures",
            "evaluation_paths",
            "backtest_paths",
            "backtest_summary",
            "backtest_status",
            "shortlisted",
            "survived_predictive",
            "supervisor_history",
            "retry_count",
            "failure_kind",
            "last_error",
        ]
        if key in existing
    }


def _write_run_manifest(*, local_state: Path, experiment_id: str, manifest: dict[str, Any]) -> None:
    path = _local_experiment_root(local_state, experiment_id) / "runs" / f"{manifest['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_experiment_summary(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    path = _local_experiment_root(local_state, experiment_id) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_experiment_summary(*, local_state: Path, experiment_id: str, summary: dict[str, Any]) -> None:
    path = _local_experiment_root(local_state, experiment_id) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _load_run_manifests(*, local_state: Path, experiment_id: str) -> list[dict[str, Any]]:
    root = _local_experiment_root(local_state, experiment_id) / "runs"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def _refresh_experiment_summary(*, local_state: Path, experiment_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    existing = _read_experiment_summary(local_state=local_state, experiment_id=experiment_id)
    runs = summary.get("runs") or _load_run_manifests(local_state=local_state, experiment_id=experiment_id)
    counts = dict(summary.get("counts") or {})
    evaluation_counts: dict[str, int] = {}
    gate_failures: dict[str, int] = {}
    shortlist = []
    best_run_id = None
    best_candidate = None
    best_primary_score = None
    best_backtest_net_return = None
    best_decision = None
    best_decision_reason = None
    best_score = -10.0
    for run in runs:
        stage = str(run.get("evaluation_stage") or "PLANNED")
        evaluation_counts[stage] = evaluation_counts.get(stage, 0) + 1
        if run.get("shortlisted"):
            shortlist.append(run["run_id"])
        for failure in run.get("gate_failures", []) or []:
            gate_failures[str(failure)] = gate_failures.get(str(failure), 0) + 1
        preview = run.get("report_preview") or {}
        score = preview.get("lightgbm_mean_rank_ic")
        if score is None:
            score = preview.get("ridge_mean_rank_ic")
        try:
            numeric = float(score or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric > best_score:
            best_score = numeric
            best_run_id = run.get("run_id")
            best_candidate = run.get("model_suite")
            best_primary_score = numeric
            best_backtest_net_return = (run.get("backtest_summary") or {}).get("net_return")
            best_decision = (run.get("assessment") or {}).get("decision")
            best_decision_reason = (run.get("assessment") or {}).get("reason")
    supervisor = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    proposal_path = _local_experiment_root(local_state, experiment_id) / "next_family_proposal.json"
    proposal_summary = json.loads(proposal_path.read_text(encoding="utf-8")) if proposal_path.exists() else {}
    merged = {
        **existing,
        **summary,
        "counts": counts,
        "run_count": len(runs),
        "runs": [_summary_run_row(run) for run in runs],
        "evaluation_counts": evaluation_counts,
        "shortlist": shortlist,
        "shortlist_count": len(shortlist),
        "best_run_id": best_run_id,
        "best_candidate": best_candidate,
        "best_primary_score": best_primary_score,
        "best_backtest_net_return": best_backtest_net_return,
        "best_decision": best_decision,
        "best_decision_reason": best_decision_reason,
        "top_gate_failures": sorted(gate_failures.items(), key=lambda item: (-item[1], item[0]))[:5],
        "supervisor": supervisor,
        "proposal_summary": summary.get("proposal_summary") or proposal_summary,
    }
    _write_experiment_summary(local_state=local_state, experiment_id=experiment_id, summary=merged)
    return merged


def _load_report_payload(
    *,
    manifest: dict[str, Any],
    local_state: Path,
    repo_root: Path | None,
    data_root: Path | None,
    targets_config_path: Path | None,
    python_executable: str,
) -> dict[str, Any] | None:
    report_path = Path(str(manifest["report_path"]))
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = _local_experiment_root(local_state, str(manifest["experiment_id"])) / "cache" / "reports" / f"{manifest['run_id']}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if repo_root is None or data_root is None or targets_config_path is None:
        return None
    target = resolve_training_target(
        target_name=str(manifest["target"]),
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    payload = _read_json_on_target(target=target, path=report_path)
    if payload is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _load_backtest_summary(
    *,
    manifest: dict[str, Any],
    local_state: Path,
    repo_root: Path | None,
    data_root: Path | None,
    targets_config_path: Path | None,
    python_executable: str,
) -> dict[str, Any] | None:
    summary_path = manifest.get("backtest_paths", {}).get("summary")
    if not summary_path:
        return None
    path = Path(str(summary_path))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if repo_root is None or data_root is None or targets_config_path is None:
        return None
    target = resolve_training_target(
        target_name=str(manifest["target"]),
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    return _read_json_on_target(target=target, path=path)


def _read_json_on_target(*, target: Any, path: Path) -> dict[str, Any] | None:
    if target.kind == "local":
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    command = _remote_python_here_doc(
        target,
        f"""
import json
from pathlib import Path
path = Path({path.as_posix()!r})
if not path.exists():
    print("")
else:
    print(path.read_text(encoding="utf-8"))
""",
    )
    result = _run_ssh_command(target, command)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def _primary_rank_ic(*, manifest: dict[str, Any], report: dict[str, Any]) -> float:
    if manifest.get("model_suite") == "full":
        lightgbm = report.get("lightgbm", {})
        if not lightgbm.get("skipped"):
            value = lightgbm.get("mean_rank_ic")
            if value is not None:
                return float(value)
    return float(report.get("ridge", {}).get("mean_rank_ic", 0.0) or 0.0)


def _evaluate_report(*, manifest: dict[str, Any], report: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    assessment = report.get("assessment")
    diagnostics = report.get("diagnostics")
    if not isinstance(assessment, dict) or not isinstance(diagnostics, dict):
        raise ValueError("report missing assessment or diagnostics payload")
    ic_by_year = diagnostics.get("ic_by_year")
    placebo = diagnostics.get("placebo")
    cost_stress = diagnostics.get("cost_stress")
    if not isinstance(ic_by_year, dict) or not isinstance(placebo, list) or not isinstance(cost_stress, dict):
        raise ValueError("report missing required predictive diagnostics")
    coverage = report.get("pinned_freeze_coverage", report.get("planner_window_coverage", report.get("coverage")))
    if coverage is None:
        raise ValueError("report missing coverage metric")
    rank_ic = _primary_rank_ic(manifest=manifest, report=report)
    years_positive = all(float(value) > 0 for value in ic_by_year.values()) if ic_by_year else False
    max_abs_placebo = max((abs(float(value)) for value in placebo), default=0.0)
    cost_stress_net_return = float(cost_stress.get("net_return", 0.0) or 0.0)
    pbo = diagnostics.get("pbo")
    dsr = diagnostics.get("dsr")
    failures: list[str] = []
    if gate["require_go_decision"] and str(assessment.get("decision")) != "GO":
        failures.append("assessment.decision != GO")
    if rank_ic < gate["min_rank_ic"]:
        failures.append(f"rank_ic<{gate['min_rank_ic']}")
    if gate["require_all_years_positive"] and not years_positive:
        failures.append("not all yearly IC values are positive")
    if max_abs_placebo > gate["max_abs_placebo_ic"]:
        failures.append(f"abs(placebo)>{gate['max_abs_placebo_ic']}")
    if cost_stress_net_return < gate["min_cost_stress_net_return"]:
        failures.append(f"cost_stress_net_return<{gate['min_cost_stress_net_return']}")
    if gate["max_pbo"] is not None and float(pbo or 0.0) > gate["max_pbo"]:
        failures.append(f"pbo>{gate['max_pbo']}")
    if gate["min_dsr"] is not None and float(dsr or 0.0) < gate["min_dsr"]:
        failures.append(f"dsr<{gate['min_dsr']}")
    if float(coverage or 0.0) < gate["min_coverage"]:
        failures.append(f"coverage<{gate['min_coverage']}")
    survived = not failures
    return {
        "run_id": manifest["run_id"],
        "evaluation_stage": "SURVIVES_PREDICTIVE" if survived else "REJECTED_PREDICTIVE",
        "survived_predictive": survived,
        "gate_failures": failures,
        "primary_rank_ic": rank_ic,
        "years_positive": years_positive,
        "max_abs_placebo": max_abs_placebo,
        "cost_stress_net_return": cost_stress_net_return,
        "pbo": pbo,
        "dsr": dsr,
        "coverage": coverage,
        "assessment": assessment,
    }


def _write_evaluation_artifacts(*, local_state: Path, experiment_id: str, run_id: str, evaluation: dict[str, Any]) -> dict[str, str]:
    root = _local_experiment_root(local_state, experiment_id) / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{run_id}.evaluation.json"
    md_path = root / f"{run_id}.evaluation.md"
    json_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# Evaluation {run_id}",
                "",
                f"- evaluation_stage: {evaluation['evaluation_stage']}",
                f"- primary_rank_ic: {evaluation['primary_rank_ic']}",
                f"- years_positive: {evaluation['years_positive']}",
                f"- max_abs_placebo: {evaluation['max_abs_placebo']}",
                f"- cost_stress_net_return: {evaluation['cost_stress_net_return']}",
                f"- gate_failures: {evaluation['gate_failures']}",
            ]
        ),
        encoding="utf-8",
    )
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def _run_backtest_for_manifest(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    report = _load_report_payload(
        manifest=manifest,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    if report is None:
        raise RuntimeError(f"report unavailable for run {manifest['run_id']}")
    artifacts = dict(report.get("artifacts") or {})
    predictions_key = "primary_predictions_path"
    targets_key = "primary_targets_path"
    prices_path = artifacts.get("prices_path")
    predictions_path = artifacts.get(predictions_key)
    targets_path = artifacts.get(targets_key)
    if not prices_path or not predictions_path or not targets_path:
        raise RuntimeError(f"missing backtest input artifacts for run {manifest['run_id']}")
    output_dir = Path(str(manifest["output_root"])) / "backtests" / "primary"
    target = resolve_training_target(
        target_name=str(manifest["target"]),
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    if target.kind == "local":
        command = [
            target.python_executable,
            str(repo_root / "src" / "scripts" / "backtest.py"),
            "--prices",
            str(prices_path),
            "--targets",
            str(targets_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_dir),
        ]
        result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    else:
        command = " && ".join(
            [
                f"mkdir -p {shlex.quote(str(output_dir))}",
                f"cd {shlex.quote(str(target.repo_root))}",
                " ".join(
                    [
                        shlex.quote(target.python_executable),
                        "src/scripts/backtest.py",
                        "--prices",
                        shlex.quote(str(prices_path)),
                        "--targets",
                        shlex.quote(str(targets_path)),
                        "--predictions",
                        shlex.quote(str(predictions_path)),
                        "--output",
                        shlex.quote(str(output_dir)),
                    ]
                ),
            ]
        )
        result = _run_ssh_command(target, command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "backtest execution failed")
    payload = json.loads(result.stdout.strip() or "{}")
    summary_path = payload.get("paths", {}).get("summary")
    if summary_path:
        manifest.setdefault("backtest_paths", {})["summary"] = summary_path
    return payload


def _evaluate_backtest(*, summary_payload: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if float(summary_payload.get("net_return", 0.0) or 0.0) < gate["min_net_return"]:
        failures.append(f"net_return<{gate['min_net_return']}")
    if gate["require_cost_positive"] and float(summary_payload.get("cost_total", 0.0) or 0.0) <= 0.0:
        failures.append("cost_total<=0")
    if gate["max_turnover"] is not None and float(summary_payload.get("turnover", 0.0) or 0.0) > gate["max_turnover"]:
        failures.append(f"turnover>{gate['max_turnover']}")
    return {"shortlisted": not failures, "gate_failures": failures}


def _build_next_family_proposal(*, experiment_id: str, base_spec: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    rows = list(comparison.get("rows") or [])
    shortlisted = [row for row in rows if row.get("shortlisted")]
    predictive_survivors = [row for row in rows if row.get("evaluation_stage") == "SURVIVES_PREDICTIVE"]
    full_rows = [row for row in rows if row.get("model_suite") == "full"]
    ridge_rows = [row for row in rows if row.get("model_suite") == "ridge_only"]
    policy = _proposal_policy(base_spec)
    allowed_dimensions = set(policy["allowed_dimensions"])
    current_generation = int(base_spec.get("generation", 0) or 0)
    family_root = str(base_spec.get("family_root") or experiment_id)
    next_generation = current_generation + 1
    next_spec = {
        "experiment_id": f"{family_root}-g{next_generation}",
        "phase": int(base_spec.get("phase", 1) or 1),
        "target": base_spec.get("target"),
        "report_date_policy": str(base_spec.get("report_date_policy") or "phase1_freeze"),
        "max_concurrent": int(base_spec.get("max_concurrent", 1) or 1),
        "supervision": dict(base_spec.get("supervision") or {}),
        "predictive_gate": dict(base_spec.get("predictive_gate") or {}),
        "backtest_gate": dict(base_spec.get("backtest_gate") or {}),
        "proposal_policy": dict(base_spec.get("proposal_policy") or {}),
        "family_root": family_root,
        "generation": next_generation,
        "parent_experiment_id": experiment_id,
    }
    rationale: list[str] = []
    next_matrix: dict[str, list[Any]]

    ridge_score = max((float(row.get("primary_score") or 0.0) for row in ridge_rows), default=-1.0)
    full_score = max((float(row.get("primary_score") or 0.0) for row in full_rows), default=-1.0)
    dominant_architecture = "linear_baseline" if ridge_score >= full_score else "tree_challenger"

    if not predictive_survivors:
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            dominant_architecture=dominant_architecture,
            initial_years=[2, 3],
            feature_families=["price_core", "price_liquidity", "price_short_horizon"],
            data_profiles=["phase1_default", "phase1_short_window", "phase1_long_window"],
            family_size_cap=policy["family_size_cap"],
        )
        if dominant_architecture == "linear_baseline":
            rationale.append("all runs failed predictive gates and ridge outperformed full, so the next family pivots to ridge-centered architecture and feature-family variants")
        else:
            rationale.append("all runs failed predictive gates and the tree challenger was best, so the next family keeps that architecture while rotating feature families and data windows")
    elif predictive_survivors and not shortlisted:
        best = predictive_survivors[0]
        best_architecture = _architecture_family_for_row(best)
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            dominant_architecture=best_architecture,
            initial_years=sorted({int(best["matrix_values"].get("validation.initial_train_years", 2) or 2), 2, 3}),
            feature_families=[_feature_family_for_row(best), "price_core"],
            data_profiles=["phase1_default"],
            family_size_cap=policy["family_size_cap"],
            extra_dimension=("portfolio.cost_stress_multiplier", [2.0, 3.0]),
        )
        rationale.append("a predictive survivor failed backtest gates, so the next family tightens around the winning architecture and stresses cost assumptions")
    else:
        best = shortlisted[0]
        best_architecture = _architecture_family_for_row(best)
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            dominant_architecture=best_architecture,
            initial_years=sorted({int(best["matrix_values"].get("validation.initial_train_years", 2) or 2), 2, 3}),
            feature_families=[_feature_family_for_row(best), "price_core"],
            data_profiles=["phase1_default", "phase1_long_window"],
            family_size_cap=policy["family_size_cap"],
        )
        rationale.append("a shortlist candidate exists, so the next family expands around that winner instead of reopening the whole search")
    next_spec["matrix"] = next_matrix
    chain_allowed = current_generation < policy["max_generations"]
    data_recommendations = _data_expansion_recommendations(
        predictive_survivors=predictive_survivors,
        shortlisted=shortlisted,
    )
    return {
        "experiment_id": experiment_id,
        "recommended_experiment_id": next_spec["experiment_id"],
        "rationale": rationale,
        "data_recommendations": data_recommendations,
        "next_spec": next_spec,
        "chain_allowed": chain_allowed,
        "generation": next_generation,
    }


def _bounded_matrix(
    *,
    allowed_dimensions: set[str],
    dominant_architecture: str,
    initial_years: list[int],
    feature_families: list[str],
    data_profiles: list[str],
    family_size_cap: int,
    extra_dimension: tuple[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    matrix: dict[str, list[Any]] = {}
    if "architecture_family" in allowed_dimensions:
        matrix["architecture_family"] = [dominant_architecture]
    else:
        matrix["model_suite"] = [ARCHITECTURE_FAMILY_PRESETS[dominant_architecture]["model_suite"]]
    if "feature_family" in allowed_dimensions:
        max_features = max(1, family_size_cap // max(1, len(initial_years)))
        matrix["feature_family"] = list(dict.fromkeys(feature_families))[:max_features]
    if "validation.initial_train_years" in allowed_dimensions:
        max_years = max(1, family_size_cap // max(1, len(matrix.get("feature_family", [1]))))
        matrix["validation.initial_train_years"] = list(dict.fromkeys(initial_years))[:max_years]
    if "data_profile" in allowed_dimensions:
        current_size = _matrix_combination_count(matrix)
        max_profiles = max(1, family_size_cap // max(1, current_size))
        matrix["data_profile"] = list(dict.fromkeys(data_profiles))[:max_profiles]
    if extra_dimension is not None:
        key, values = extra_dimension
        current_size = _matrix_combination_count(matrix)
        max_values = max(1, family_size_cap // max(1, current_size))
        matrix[key] = list(dict.fromkeys(values))[:max_values]
    return matrix


def _matrix_combination_count(matrix: dict[str, list[Any]]) -> int:
    size = 1
    for values in matrix.values():
        size *= max(1, len(values))
    return size


def _architecture_family_for_row(row: dict[str, Any]) -> str:
    matrix_values = dict(row.get("matrix_values") or {})
    family = matrix_values.get("architecture_family")
    if family:
        return str(family)
    return "tree_challenger" if str(row.get("model_suite")) == "full" else "linear_baseline"


def _feature_family_for_row(row: dict[str, Any]) -> str:
    matrix_values = dict(row.get("matrix_values") or {})
    family = matrix_values.get("feature_family")
    if family:
        return str(family)
    return "price_liquidity"


def _write_next_family_proposal(*, local_state: Path, experiment_id: str, proposal: dict[str, Any]) -> dict[str, str]:
    root = _local_experiment_root(local_state, experiment_id)
    json_path = root / "next_family_proposal.json"
    md_path = root / "next_family_proposal.md"
    spec_path = root / "next_family_proposal.yml"
    json_path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    spec_path.write_text(yaml.safe_dump(proposal["next_spec"], sort_keys=False), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# Next family for {experiment_id}",
                "",
                f"- recommended_experiment_id: {proposal['recommended_experiment_id']}",
                *[f"- rationale: {item}" for item in proposal["rationale"]],
                *[f"- data_recommendation: {item}" for item in proposal.get("data_recommendations", [])],
                "",
                "```yaml",
                yaml.safe_dump(proposal["next_spec"], sort_keys=False).rstrip(),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return {"json_path": str(json_path), "markdown_path": str(md_path), "spec_path": str(spec_path)}


def _spawn_supervisor_process(
    *,
    experiment_id: str,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    python_executable: str,
    poll_seconds: int,
) -> dict[str, Any]:
    log_path = _supervisor_log_path(local_state=local_state, experiment_id=experiment_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "trademl.cli",
        "experiments",
        "--data-root",
        str(data_root),
        "--local-state",
        str(local_state),
        "--env-file",
        str(env_path),
        "supervise",
        "--spec",
        str(spec_path),
        "--poll-seconds",
        str(poll_seconds),
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=repo_root, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
    payload = {
        "experiment_id": experiment_id,
        "spec_path": str(spec_path),
        "pid": process.pid,
        "status": "RUNNING",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "heartbeat_at": datetime.now(tz=UTC).isoformat(),
        "poll_seconds": poll_seconds,
        "paused": False,
        "stop_requested": False,
        "active_run_ids": [],
        "queue_counts": {},
        "log_path": str(log_path),
    }
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=payload)
    return payload


def _ensure_supervisor_state(*, local_state: Path, experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    return {**payload, **state}


def _supervisor_state_path(*, local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiment_supervisors" / f"{experiment_id}.json"


def _supervisor_log_path(*, local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiment_supervisors" / "logs" / f"{experiment_id}.log"


def _write_supervisor_state(*, local_state: Path, experiment_id: str, payload: dict[str, Any]) -> None:
    path = _supervisor_state_path(local_state=local_state, experiment_id=experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _run_ready_for_retry(*, manifest: dict[str, Any], supervision: dict[str, Any]) -> bool:
    status = str(manifest.get("status") or "PLANNED")
    if status == "PLANNED":
        return True
    if status != "FAILED":
        return False
    if str(manifest.get("failure_kind") or "") != "infra":
        return False
    return int(manifest.get("retry_count", 0) or 0) < int(supervision["max_retry_count"])


def _append_supervisor_event(manifest: dict[str, Any], *, event: str, payload: dict[str, Any]) -> None:
    history = list(manifest.get("supervisor_history") or [])
    history.append({"at": datetime.now(tz=UTC).isoformat(), "event": event, **payload})
    manifest["supervisor_history"] = history[-50:]


def _classify_failure(error: str) -> str:
    lowered = error.lower()
    if any(token in lowered for token in ["ssh", "permission denied", "connection refused", "connection reset", "host", "mount", "timed out", "no route"]):
        return "infra"
    if "preflight failed" in lowered:
        return "preflight"
    if any(token in lowered for token in ["missing qc parquet", "missing config", "no curated parquet"]):
        return "preflight"
    return "model"


def _data_expansion_recommendations(*, predictive_survivors: list[dict[str, Any]], shortlisted: list[dict[str, Any]]) -> list[str]:
    if shortlisted:
        return [
            "Add PIT-safe daily context next: FRED macro regime features and SEC filing-timing risk controls around the shortlisted family.",
            "Stage ticker-tagged news aggregates as a separate daily research lane before attempting minute-level modeling.",
        ]
    if predictive_survivors:
        return [
            "Backtest survivors suggest the next leverage point is daily event context: SEC filing dates, earnings-risk flags, and macro regime features.",
            "Keep minute bars and intraday/news alpha in a separate future lane until daily cost-adjusted winners exist.",
        ]
    return [
        "Before adding more model complexity, compare price-only vs liquidity-inclusive families because free IEX-volume features may still be noisy.",
        "The next data additions should be PIT-safe daily lanes: FRED macro pack, SEC filing dates, and news/event aggregates collected daily rather than minute-level alpha.",
    ]


def _is_local_process_running(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}")
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
