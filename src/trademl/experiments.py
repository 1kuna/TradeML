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
from datetime import UTC, datetime, timedelta
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
    stop_training_process,
    training_status_snapshot,
)
from trademl.modeling import DEFAULT_LABEL_VERSION, modeling_artifact_metadata
from trademl.research_architecture import (
    build_candidate_autopsy,
    build_objective_verdict,
    diagnostic_family_signature,
    gate_failures_by_objective,
    launchable_architecture_presets,
    primary_score_from_preview,
    primary_score_from_report,
    resolve_architecture_entry,
)
from trademl.validation.negative_controls import evaluate_negative_controls

SPECIAL_MATRIX_DIMENSIONS = {
    "model_suite",
    "architecture_family",
    "feature_family",
    "data_profile",
    "data_family",
    "feature_version",
    "label_horizon",
    "portfolio_profile",
}


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so concurrent readers never observe partial state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

ARCHITECTURE_FAMILY_PRESETS: dict[str, dict[str, Any]] = launchable_architecture_presets()

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

DATA_FAMILY_PRESETS: dict[str, dict[str, Any]] = {
    "price_only": {
        "config_overrides": {
            "features.liquidity.adv_dollar": [],
            "features.liquidity.amihud": [],
        },
        "modeling_ready": True,
        "required_datasets": ["equities_ohlcv_adj"],
    },
    "price_plus_liquidity": {
        "config_overrides": {},
        "modeling_ready": True,
        "required_datasets": ["equities_ohlcv_adj"],
    },
    "price_plus_macro": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "macros_fred_curated"],
    },
    "price_plus_events": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "sec_events_curated"],
    },
    "price_plus_news": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "ticker_news_curated"],
    },
    "minute_archive_derived": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "equities_minute_curated"],
    },
    "filings_derived": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "sec_filings_curated"],
    },
    "alt_reference_enriched": {
        "config_overrides": {},
        "modeling_ready": False,
        "required_datasets": ["equities_ohlcv_adj", "alt_reference_curated"],
    },
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
        modeling = _run_modeling_metadata(data_root=target.data_root, run_config=run_config)
        config_path = configs_dir / f"{run_id}.yml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        output_root = target.data_root / "experiments" / experiment_id / "runs" / run_id
        model_suite = row.get("model_suite") or spec.get("model_suite") or ("ridge_only" if phase == 1 else "full")
        architecture_entry = _architecture_entry_for_row(row={**row, "model_suite": model_suite})
        objective_policy = _objective_policy(spec)
        base_manifest = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_priority": _run_priority(row={**row, "model_suite": model_suite}, spec=spec),
            "runtime_name": runtime_name,
            "phase": phase,
            "target": target.name,
            "target_kind": target.kind,
            "report_date": report_date,
            "model_suite": model_suite,
            "matrix_values": row["matrix_values"],
            "architecture_registry_entry": architecture_entry,
            "architecture_lane": architecture_entry["family"],
            "objective_policy": objective_policy,
            "complexity_tier": int(architecture_entry["complexity_tier"]),
            "feature_set": modeling["feature_set"],
            "feature_version": modeling["feature_version"],
            "label_version": modeling["label_version"],
            "label_horizon": modeling["label_horizon"],
            "label_definition": modeling["label_definition"],
            "data_revision": modeling.get("data_revision"),
            "portfolio_profile": modeling["portfolio_profile"],
            "follow_up_of_run_id": spec.get("follow_up_of_run_id"),
            "follow_up_reason": spec.get("follow_up_reason"),
            "diagnostic_mode": spec.get("diagnostic_mode"),
            "diagnostic_family_signature": spec.get("diagnostic_family_signature"),
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
        _atomic_write_json(manifest_path, merged)
        runs.append(merged)

    summary = {
        "experiment_id": experiment_id,
        "phase": phase,
        "target": target_name,
        "report_date": report_date,
        "spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
        "predictive_gate": _predictive_gate(spec),
        "objective_policy": _objective_policy(spec),
        "backtest_gate": _backtest_gate(spec),
        "backtest_profile": dict(spec.get("backtest_profile") or {}),
        "proposal_policy": _proposal_policy(spec),
        "follow_up_of_run_id": spec.get("follow_up_of_run_id"),
        "follow_up_reason": spec.get("follow_up_reason"),
        "diagnostic_mode": spec.get("diagnostic_mode"),
        "diagnostic_family_signature": spec.get("diagnostic_family_signature"),
        "supervision": _supervision_policy(spec),
        "max_concurrent": max_concurrent,
        "run_count": len(runs),
        "runs": [_summary_run_row(run) for run in runs],
    }
    _write_experiment_summary(local_state=local_state, experiment_id=experiment_id, summary=summary, data_root=data_root)
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
        current_status = str(manifest.get("status") or "PLANNED").upper()
        if current_status == "PLANNED":
            manifest["status"] = "PLANNED"
            manifest.setdefault("runtime", {})
            manifest.setdefault("log_tail", "")
            _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
            runs.append(manifest)
            continue
        if not _manifest_requires_runtime_refresh(manifest):
            if current_status == "COMPLETED" and _manifest_needs_report_refresh(manifest):
                report = _load_report_payload(
                    manifest=manifest,
                    local_state=local_state,
                    repo_root=repo_root,
                    data_root=data_root,
                    targets_config_path=targets_config_path,
                    python_executable=python_executable,
                )
                if report is not None:
                    manifest["assessment"] = report.get("assessment", {})
                    manifest["report_preview"] = _report_preview_from_report(report)
                    _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
            runs.append(manifest)
            continue
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
            manifest["report_preview"] = _report_preview_from_report(report)
        elif status == "UNKNOWN":
            status = "FAILED"
            manifest["last_error"] = runtime.get("error") or manifest.get("last_error") or "training runtime ended unexpectedly"
            if not manifest.get("failure_kind") or str(manifest.get("failure_kind")) == "model":
                manifest["failure_kind"] = _classify_failure(
                    "\n".join(
                        part
                        for part in [
                            str(manifest.get("last_error") or ""),
                            str(snapshot.get("log_tail") or ""),
                        ]
                        if part
                    )
                    or "training runtime ended unexpectedly"
                )
        elif status == "FAILED":
            manifest["last_error"] = runtime.get("error") or manifest.get("last_error")
            failure_detail = "\n".join(
                part
                for part in [
                    str(manifest.get("last_error") or ""),
                    str(snapshot.get("log_tail") or ""),
                ]
                if part
            )
            if not manifest.get("failure_kind") or str(manifest.get("failure_kind")) == "model":
                manifest["failure_kind"] = _classify_failure(failure_detail or "training runtime failed")
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
    return _refresh_experiment_summary(
        local_state=local_state,
        experiment_id=experiment_id,
        summary=summary,
        data_root=data_root,
    )


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
        manifest["gate_failures_by_objective"] = evaluation["gate_failures_by_objective"]
        manifest["objective_verdict"] = evaluation["objective_verdict"]
        manifest["candidate_autopsy"] = evaluation["candidate_autopsy"]
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
                "candidate_autopsy": evaluation["candidate_autopsy"],
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
        objective_verdict = dict(manifest.get("objective_verdict") or {})
        objective_verdict["backtest"] = {
            "net_return": summary_payload.get("net_return"),
            "gross_return": summary_payload.get("gross_return"),
            "turnover": summary_payload.get("turnover"),
            "cost_total": summary_payload.get("cost_total"),
            "gate_failures": outcome["gate_failures"],
            "passed": bool(outcome["shortlisted"]),
        }
        objective_verdict["passed"] = bool(objective_verdict.get("passed", False) and outcome["shortlisted"])
        objective_verdict["gate_failures_by_objective"] = {
            **dict(objective_verdict.get("gate_failures_by_objective") or {}),
            **gate_failures_by_objective(outcome["gate_failures"]),
        }
        manifest["objective_verdict"] = objective_verdict
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
        data_root=data_root,
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
            "completed_at": None,
            "pid": None,
            "poll_seconds": resolved_poll,
            "last_error": None,
            "last_error_kind": None,
            "active_run_ids": [],
            "queue_counts": {},
            "paused": False,
            "stop_requested": False,
            "stop_reason": None,
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
        planned = _count_launchable_runs(runs=list(status.get("runs") or []), supervision=supervision)
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    pid = payload.get("pid")
    status = str(payload.get("status") or "").upper()
    active_statuses = {"RUNNING", "PAUSED", "STARTING", "STOPPING"}
    stopped_reason: str | None = None
    if isinstance(pid, int) and status in active_statuses and not _is_local_process_running(pid):
        stopped_reason = f"local experiment supervisor pid {pid} is not running"
    elif status in active_statuses and _supervisor_heartbeat_stale(payload):
        stopped_reason = "experiment supervisor heartbeat is stale"
    if stopped_reason:
        payload["status"] = "STOPPED"
        payload["active_run_ids"] = []
        payload["stop_reason"] = stopped_reason
        payload["completed_at"] = payload.get("completed_at") or datetime.now(tz=UTC).isoformat()
        _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=payload)
    return payload


def pause_experiment_supervisor(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    """Pause an active experiment supervisor."""
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    if not state:
        raise ValueError(f"no supervisor state for experiment {experiment_id!r}")
    state["paused"] = True
    state["status"] = "PAUSED"
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
    return state


def stop_experiment_supervisor(
    *,
    local_state: Path,
    experiment_id: str,
    repo_root: Path | None = None,
    data_root: Path | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
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
    state["stopped_training_runs"] = _stop_active_training_runs(
        local_state=local_state,
        experiment_id=experiment_id,
        run_ids=[str(run_id) for run_id in list(state.get("active_run_ids") or [])],
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=state)
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
                "experiment_id": manifest.get("experiment_id"),
                "target": manifest["target"],
                "model_suite": manifest["model_suite"],
                "matrix_values": manifest["matrix_values"],
                "architecture_lane": manifest.get("architecture_lane"),
                "architecture_registry_entry": manifest.get("architecture_registry_entry", {}),
                "objective_policy": manifest.get("objective_policy", {}),
                "complexity_tier": manifest.get("complexity_tier"),
                "feature_set": manifest.get("feature_set") or (report.get("modeling") or {}).get("feature_set"),
                "feature_version": manifest.get("feature_version") or (report.get("modeling") or {}).get("feature_version"),
                "label_version": manifest.get("label_version") or (report.get("modeling") or {}).get("label_version"),
                "label_horizon": manifest.get("label_horizon") or (report.get("modeling") or {}).get("label_horizon"),
                "label_definition": manifest.get("label_definition") or (report.get("modeling") or {}).get("label_definition"),
                "data_revision": manifest.get("data_revision") or (report.get("modeling") or {}).get("data_revision"),
                "portfolio_profile": manifest.get("portfolio_profile"),
                "coverage": report.get("coverage"),
                "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
                "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
                "pbo": report.get("diagnostics", {}).get("pbo"),
                "dsr": report.get("diagnostics", {}).get("dsr"),
                "years_positive": bool(report.get("diagnostics", {}).get("ic_by_year"))
                and all(float(value) > 0 for value in (report.get("diagnostics", {}).get("ic_by_year") or {}).values()),
                "decision": report.get("assessment", {}).get("decision"),
                "decision_reason": report.get("assessment", {}).get("reason"),
                "assessment": report.get("assessment", {}),
                "evaluation_stage": manifest.get("evaluation_stage"),
                "gate_failures": manifest.get("gate_failures", []),
                "survived_predictive": bool(manifest.get("survived_predictive")),
                "backtest_status": manifest.get("backtest_status", "NOT_STARTED"),
                "shortlisted": bool(manifest.get("shortlisted")),
                "backtest_net_return": (backtest_summary or {}).get("net_return"),
                "backtest_turnover": (backtest_summary or {}).get("turnover"),
                "artifacts": report.get("artifacts", {}),
                "report_path": str(manifest.get("report_path")),
                "primary_score": primary_score,
                "objective_verdict": manifest.get("objective_verdict", {}),
                "candidate_autopsy": manifest.get("candidate_autopsy") or report.get("candidate_autopsy") or {},
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
    _atomic_write_json(json_path, comparison)
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
    data_family = matrix_values.get("data_family")
    if data_family is not None:
        preset = DATA_FAMILY_PRESETS.get(str(data_family))
        if preset is None:
            raise ValueError(f"unsupported data_family: {data_family}")
        config_overrides.update(dict(preset.get("config_overrides") or {}))
    if matrix_values.get("feature_version") is not None:
        config_overrides["modeling.feature_version"] = str(matrix_values["feature_version"])
    if matrix_values.get("label_horizon") is not None:
        config_overrides["modeling.label_horizon"] = int(matrix_values["label_horizon"])
    if matrix_values.get("portfolio_profile") is not None:
        profile = str(matrix_values["portfolio_profile"])
        config_overrides["portfolio.portfolio_profile"] = profile
        config_overrides["portfolio.method"] = profile
        config_overrides["validation.portfolio_profile"] = profile
    config_overrides.update({key: value for key, value in matrix_values.items() if key not in SPECIAL_MATRIX_DIMENSIONS})
    return {
        "matrix_values": matrix_values,
        "config_overrides": config_overrides,
        "model_suite": model_suite,
    }


def _architecture_entry_for_row(*, row: dict[str, Any]) -> dict[str, Any]:
    family = dict(row.get("matrix_values") or {}).get("architecture_family")
    if family is None:
        model_suite = str(row.get("model_suite") or "")
        family = (
            "ensemble_meta"
            if model_suite == "ensemble"
            else "advanced_challenger"
            if model_suite == "advanced"
            else "tree_challenger"
            if model_suite == "full"
            else "linear_baseline"
        )
    return resolve_architecture_entry(str(family))


def _objective_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policy = dict(spec.get("objective_policy") or {})
    if not policy:
        return {
            "enabled": True,
            "primary": "research_profitability_v1",
            "complexity_penalty": {"enabled": True, "penalty_per_tier": 0.0005, "min_complexity_adjusted_improvement": 0.0},
        }
    policy.setdefault("enabled", True)
    policy.setdefault("primary", "research_profitability_v1")
    policy.setdefault("complexity_penalty", {"enabled": True, "penalty_per_tier": 0.0005, "min_complexity_adjusted_improvement": 0.0})
    return policy


def _run_modeling_metadata(*, data_root: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    modeling = dict(run_config.get("modeling") or {})
    portfolio = dict(run_config.get("portfolio") or {})
    registry = modeling_artifact_metadata(data_root=data_root)
    return {
        "feature_set": str(modeling.get("feature_set") or registry.get("feature_set") or "daily_price_liquidity_v1"),
        "feature_version": str(modeling.get("feature_version") or registry.get("feature_version") or "price_liquidity_v1"),
        "label_version": str(modeling.get("label_version") or registry.get("label_version") or DEFAULT_LABEL_VERSION),
        "label_horizon": int(modeling.get("label_horizon") or modeling.get("primary_label_horizon") or 5),
        "label_definition": str(modeling.get("label_definition") or registry.get("label_definition") or "universe_relative_forward_return"),
        "data_revision": registry.get("data_revision"),
        "portfolio_profile": str(portfolio.get("portfolio_profile") or portfolio.get("method") or "equal_weight_top_quintile"),
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
        "max_abs_negative_control_ic": float(gate.get("max_abs_negative_control_ic", 0.10) or 0.10),
        "max_abs_future_news_leak_ic": float(gate.get("max_abs_future_news_leak_ic", 0.10) or 0.10),
        "max_single_feature_score_drop": float(gate.get("max_single_feature_score_drop", 0.75) or 0.75),
        "min_feature_ablation_score_ratio": float(gate.get("min_feature_ablation_score_ratio", 0.25) or 0.25),
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


def _run_priority(*, row: dict[str, Any], spec: dict[str, Any]) -> int:
    """Order controlled frontier runs so advanced challengers launch first."""
    if not bool(spec.get("frontier_architecture", False)):
        return 100
    try:
        entry = _architecture_entry_for_row(row=row)
    except ValueError:
        return 100
    matrix_families = set(str(item) for item in (spec.get("matrix") or {}).get("architecture_family") or [])
    ordered_families = (
        ["advanced_challenger", "ensemble_meta", "tree_challenger", "linear_baseline"]
        if "ensemble_meta" in matrix_families
        else ["advanced_challenger", "tree_challenger", "linear_baseline"]
    )
    order = {family: idx for idx, family in enumerate(ordered_families)}
    return order.get(str(entry["family"]), 100)


def _local_experiment_root(local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiments" / experiment_id


def _shared_experiment_root(data_root: Path, experiment_id: str) -> Path:
    return data_root / "experiments" / experiment_id


def _dashboard_summary_payload(summary: dict[str, Any], *, recent_limit: int = 16) -> dict[str, Any]:
    runs = list(summary.get("runs") or [])
    recent_runs = sorted(
        (
            _summary_run_row(run) if "status" in run else dict(run)
            for run in runs
            if isinstance(run, dict)
        ),
        key=lambda row: (
            str(row.get("activity_at") or ""),
            str(row.get("report_date") or ""),
            str(row.get("run_id") or ""),
        ),
        reverse=True,
    )[:recent_limit]
    recent_runs = sorted(
        recent_runs,
        key=lambda row: (
            str(row.get("activity_at") or ""),
            str(row.get("report_date") or ""),
            str(row.get("run_id") or ""),
        ),
        reverse=True,
    )
    supervisor = summary.get("supervisor") if isinstance(summary.get("supervisor"), dict) else {}
    run_activity = max((str(row.get("activity_at") or "") for row in recent_runs), default="")
    activity_at = (
        supervisor.get("heartbeat_at")
        or supervisor.get("updated_at")
        or run_activity
        or summary.get("updated_at")
        or datetime.now(tz=UTC).isoformat()
    )
    return {
        "experiment_id": summary.get("experiment_id"),
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "activity_at": activity_at,
        "best_run_id": summary.get("best_run_id"),
        "best_candidate": summary.get("best_candidate"),
        "best_primary_score": summary.get("best_primary_score"),
        "best_backtest_net_return": summary.get("best_backtest_net_return"),
        "best_decision": summary.get("best_decision"),
        "best_decision_reason": summary.get("best_decision_reason"),
        "candidate_autopsy": summary.get("candidate_autopsy") or {},
        "candidate_classifications": summary.get("candidate_classifications") or {},
        "follow_up_of_run_id": summary.get("follow_up_of_run_id"),
        "diagnostic_mode": summary.get("diagnostic_mode"),
        "shortlist_count": summary.get("shortlist_count"),
        "recent_runs": recent_runs,
    }


def _summary_run_row(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "activity_at": _run_activity_at(manifest),
        "phase": manifest.get("phase"),
        "target": manifest.get("target"),
        "target_kind": manifest.get("target_kind"),
        "report_date": manifest.get("report_date"),
        "status": manifest["status"],
        "model_suite": manifest["model_suite"],
        "matrix_values": manifest["matrix_values"],
        "architecture_lane": manifest.get("architecture_lane"),
        "complexity_tier": manifest.get("complexity_tier"),
        "feature_set": manifest.get("feature_set"),
        "feature_version": manifest.get("feature_version"),
        "label_version": manifest.get("label_version"),
        "label_horizon": manifest.get("label_horizon"),
        "label_definition": manifest.get("label_definition"),
        "data_revision": manifest.get("data_revision"),
        "portfolio_profile": manifest.get("portfolio_profile"),
        "objective_verdict": manifest.get("objective_verdict", {}),
        "candidate_autopsy": manifest.get("candidate_autopsy", {}),
        "follow_up_of_run_id": manifest.get("follow_up_of_run_id"),
        "follow_up_reason": manifest.get("follow_up_reason"),
        "diagnostic_mode": manifest.get("diagnostic_mode"),
        "diagnostic_family_signature": manifest.get("diagnostic_family_signature"),
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
        "assessment": manifest.get("assessment", {}),
        "report_preview": manifest.get("report_preview", {}),
        "backtest_summary": manifest.get("backtest_summary", {}),
        "retry_count": manifest.get("retry_count", 0),
        "supervisor_history": manifest.get("supervisor_history", []),
        "shortlisted": bool(manifest.get("shortlisted")),
    }


def _run_activity_at(manifest: dict[str, Any]) -> str | None:
    history = manifest.get("supervisor_history")
    if isinstance(history, list):
        timestamps = [
            str(item.get("at"))
            for item in history
            if isinstance(item, dict) and item.get("at")
        ]
        if timestamps:
            return max(timestamps)
    runtime = manifest.get("runtime")
    if isinstance(runtime, dict):
        for key in ("finished_at", "started_at"):
            if runtime.get(key):
                return str(runtime[key])
    report_date = manifest.get("report_date")
    return str(report_date) if report_date else None


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
            "objective_verdict",
            "candidate_autopsy",
            "follow_up_of_run_id",
            "follow_up_reason",
            "diagnostic_mode",
            "diagnostic_family_signature",
        ]
        if key in existing
    }


def _write_run_manifest(*, local_state: Path, experiment_id: str, manifest: dict[str, Any]) -> None:
    path = _local_experiment_root(local_state, experiment_id) / "runs" / f"{manifest['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, manifest)


def _read_experiment_summary(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    path = _local_experiment_root(local_state, experiment_id) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_experiment_summary(
    *,
    local_state: Path,
    experiment_id: str,
    summary: dict[str, Any],
    data_root: Path | None = None,
) -> None:
    path = _local_experiment_root(local_state, experiment_id) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, summary)
    if data_root is not None:
        shared_root = _shared_experiment_root(data_root, experiment_id)
        shared_path = shared_root / "summary.json"
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(shared_path, summary)
        _atomic_write_json(shared_root / "dashboard_summary.json", _dashboard_summary_payload(summary))


def _load_run_manifests(*, local_state: Path, experiment_id: str) -> list[dict[str, Any]]:
    root = _local_experiment_root(local_state, experiment_id) / "runs"
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*.json")]
    return sorted(manifests, key=lambda item: (int(item["run_priority"]) if item.get("run_priority") is not None else 100, str(item.get("run_id") or "")))


def _refresh_experiment_summary(
    *,
    local_state: Path,
    experiment_id: str,
    summary: dict[str, Any],
    data_root: Path | None = None,
) -> dict[str, Any]:
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
    best_objective_verdict = {}
    best_candidate_autopsy = {}
    candidate_classifications: dict[str, int] = {}
    best_architecture_lane = None
    best_complexity_tier = None
    best_label_horizon = None
    best_feature_version = None
    best_portfolio_profile = None
    best_score = float("-inf")
    for run in runs:
        stage = str(run.get("evaluation_stage") or "PLANNED")
        evaluation_counts[stage] = evaluation_counts.get(stage, 0) + 1
        if run.get("shortlisted"):
            shortlist.append(run["run_id"])
        for failure in run.get("gate_failures", []) or []:
            gate_failures[str(failure)] = gate_failures.get(str(failure), 0) + 1
        autopsy = dict(run.get("candidate_autopsy") or {})
        classification = autopsy.get("classification")
        if classification:
            candidate_classifications[str(classification)] = candidate_classifications.get(str(classification), 0) + 1
        preview = run.get("report_preview") or {}
        score = _preview_primary_score(model_suite=str(run.get("model_suite") or ""), preview=preview)
        if score is None:
            continue
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            continue
        if numeric > best_score:
            best_score = numeric
            best_run_id = run.get("run_id")
            best_candidate = run.get("model_suite")
            best_primary_score = numeric
            best_backtest_net_return = (run.get("backtest_summary") or {}).get("net_return")
            best_decision = (run.get("assessment") or {}).get("decision")
            best_decision_reason = (run.get("assessment") or {}).get("reason")
            best_objective_verdict = dict(run.get("objective_verdict") or {})
            best_candidate_autopsy = dict(run.get("candidate_autopsy") or {})
            best_architecture_lane = run.get("architecture_lane")
            best_complexity_tier = run.get("complexity_tier")
            best_label_horizon = run.get("label_horizon")
            best_feature_version = run.get("feature_version")
            best_portfolio_profile = run.get("portfolio_profile")
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
        "objective_verdict": best_objective_verdict,
        "candidate_autopsy": best_candidate_autopsy,
        "candidate_classifications": candidate_classifications,
        "architecture_lane": best_architecture_lane,
        "complexity_tier": best_complexity_tier,
        "label_horizon": best_label_horizon,
        "feature_version": best_feature_version,
        "portfolio_profile": best_portfolio_profile,
        "follow_up_of_run_id": summary.get("follow_up_of_run_id") or existing.get("follow_up_of_run_id"),
        "follow_up_reason": summary.get("follow_up_reason") or existing.get("follow_up_reason"),
        "diagnostic_mode": summary.get("diagnostic_mode") or existing.get("diagnostic_mode"),
        "diagnostic_family_signature": summary.get("diagnostic_family_signature") or existing.get("diagnostic_family_signature"),
        "top_gate_failures": sorted(gate_failures.items(), key=lambda item: (-item[1], item[0]))[:5],
        "supervisor": supervisor,
        "proposal_summary": summary.get("proposal_summary") or proposal_summary,
    }
    _write_experiment_summary(local_state=local_state, experiment_id=experiment_id, summary=merged, data_root=data_root)
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
    _atomic_write_json(cache_path, payload)
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
    return primary_score_from_report(manifest=manifest, report=report)


def _report_preview_from_report(report: dict[str, Any]) -> dict[str, Any]:
    preview = {
        "coverage": report.get("coverage"),
        "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
        "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
    }
    catboost = report.get("catboost", {})
    if isinstance(catboost, dict) and not catboost.get("skipped"):
        preview["catboost_mean_rank_ic"] = catboost.get("mean_rank_ic")
    ensemble = report.get("ensemble", {})
    if isinstance(ensemble, dict) and not ensemble.get("skipped"):
        preview["ensemble_mean_rank_ic"] = ensemble.get("mean_rank_ic")
    return preview


def _manifest_needs_report_refresh(manifest: dict[str, Any]) -> bool:
    if not manifest.get("assessment"):
        return True
    preview = dict(manifest.get("report_preview") or {})
    model_suite = str(manifest.get("model_suite") or "")
    if model_suite == "advanced" and "catboost_mean_rank_ic" not in preview:
        return True
    if model_suite == "ensemble" and "ensemble_mean_rank_ic" not in preview:
        return True
    if model_suite in {"advanced", "full", "ensemble"} and "lightgbm_mean_rank_ic" not in preview:
        return True
    return False


def _preview_primary_score(*, model_suite: str, preview: dict[str, Any]) -> float | None:
    return primary_score_from_preview(model_suite=model_suite, preview=preview)


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
    negative_controls = evaluate_negative_controls(diagnostics, gate=gate)
    failures.extend(list(negative_controls["gate_failures"]))
    if cost_stress_net_return < gate["min_cost_stress_net_return"]:
        failures.append(f"cost_stress_net_return<{gate['min_cost_stress_net_return']}")
    if gate["max_pbo"] is not None and float(pbo or 0.0) > gate["max_pbo"]:
        failures.append(f"pbo>{gate['max_pbo']}")
    if gate["min_dsr"] is not None and float(dsr or 0.0) < gate["min_dsr"]:
        failures.append(f"dsr<{gate['min_dsr']}")
    if float(coverage or 0.0) < gate["min_coverage"]:
        failures.append(f"coverage<{gate['min_coverage']}")
    survived = not failures
    grouped_failures = gate_failures_by_objective(failures)
    objective_verdict = build_objective_verdict(
        manifest=manifest,
        primary_score=rank_ic,
        survived=survived,
        gate_failures_by_objective=grouped_failures,
        policy=manifest.get("objective_policy"),
    )
    candidate_autopsy = build_candidate_autopsy(
        manifest=manifest,
        report=report,
        gate_failures=failures,
        gate=gate,
    )
    return {
        "run_id": manifest["run_id"],
        "evaluation_stage": "SURVIVES_PREDICTIVE" if survived else "REJECTED_PREDICTIVE",
        "survived_predictive": survived,
        "gate_failures": failures,
        "gate_failures_by_objective": grouped_failures,
        "objective_verdict": objective_verdict,
        "primary_rank_ic": rank_ic,
        "years_positive": years_positive,
        "max_abs_placebo": max_abs_placebo,
        "cost_stress_net_return": cost_stress_net_return,
        "pbo": pbo,
        "dsr": dsr,
        "coverage": coverage,
        "assessment": assessment,
        "negative_controls": negative_controls,
        "candidate_autopsy": candidate_autopsy,
    }


def _write_evaluation_artifacts(*, local_state: Path, experiment_id: str, run_id: str, evaluation: dict[str, Any]) -> dict[str, str]:
    root = _local_experiment_root(local_state, experiment_id) / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{run_id}.evaluation.json"
    md_path = root / f"{run_id}.evaluation.md"
    _atomic_write_json(json_path, evaluation)
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
                f"- candidate_classification: {(evaluation.get('candidate_autopsy') or {}).get('classification')}",
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
    policy = _proposal_policy(base_spec)
    allowed_dimensions = set(policy["allowed_dimensions"])
    current_generation = int(base_spec.get("generation", 0) or 0)
    family_root = str(base_spec.get("family_root") or experiment_id)
    next_generation = current_generation + 1
    current_matrix = dict(base_spec.get("matrix") or {})
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
    best_row = rows[0] if rows else {}
    best_feature_family = _feature_family_for_row(best_row) if best_row else "price_core"
    best_data_family = _data_family_for_row(best_row) if best_row else "price_only"
    best_initial_year = int((best_row.get("matrix_values") or {}).get("validation.initial_train_years", 2) or 2) if best_row else 2
    current_feature_families = [str(item) for item in current_matrix.get("feature_family", [])]
    current_architecture_families = [str(item) for item in current_matrix.get("architecture_family", [])]
    current_data_profiles = [str(item) for item in current_matrix.get("data_profile", [])]
    current_data_families = [str(item) for item in current_matrix.get("data_family", [])]
    current_initial_years = [int(item) for item in current_matrix.get("validation.initial_train_years", [])]
    allowed_data_families = _proposal_data_families(base_spec=base_spec, current_data_families=current_data_families)
    allowed_architecture_families = _proposal_architecture_families(
        base_spec=base_spec,
        current_architecture_families=current_architecture_families,
    )
    strong_unstable = _best_classified_row(rows=rows, classification="strong_unstable")
    if strong_unstable is not None:
        horizon_values = list(dict.fromkeys([int((strong_unstable.get("matrix_values") or {}).get("label_horizon") or strong_unstable.get("label_horizon") or 5), 1, 5, 20]))
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            architecture_families=[
                family
                for family in ["advanced_challenger", "ensemble_meta", "tree_challenger", "linear_baseline"]
                if family in ARCHITECTURE_FAMILY_PRESETS
            ],
            initial_years=sorted({int((strong_unstable.get("matrix_values") or {}).get("validation.initial_train_years", 2) or 2), 2, 3, 4}),
            feature_families=[_feature_family_for_row(strong_unstable)],
            data_families=[_data_family_for_row(strong_unstable)],
            data_profiles=["phase1_short_window", "phase1_default", "phase1_long_window"],
            family_size_cap=policy["family_size_cap"],
        )
        if "label_horizon" in allowed_dimensions:
            next_matrix["label_horizon"] = horizon_values
        next_spec["matrix"] = next_matrix
        next_spec["diagnostic_mode"] = "strong_unstable"
        next_spec["follow_up_of_run_id"] = strong_unstable.get("run_id")
        next_spec["follow_up_reason"] = "strong rejected candidate failed stability gates"
        next_spec["diagnostic_family_signature"] = diagnostic_family_signature(
            {
                "run_id": strong_unstable.get("run_id"),
                "feature_version": strong_unstable.get("feature_version"),
                "label_version": strong_unstable.get("label_version"),
                "data_revision": strong_unstable.get("data_revision"),
                "objective_policy": base_spec.get("objective_policy"),
                "diagnostic_mode": "strong_unstable",
                "matrix": next_matrix,
            }
        )
        return {
            "experiment_id": experiment_id,
            "recommended_experiment_id": next_spec["experiment_id"],
            "rationale": [
                "a strong rejected candidate failed stability gates, so the next family runs horizon, architecture, and window diagnostics before broad search"
            ],
            "data_recommendations": [],
            "next_spec": next_spec,
            "chain_allowed": current_generation < policy["max_generations"],
            "generation": next_generation,
            "follow_up_of_run_id": strong_unstable.get("run_id"),
            "diagnostic_mode": "strong_unstable",
        }

    architecture_scores = {
        family: max((float(row.get("primary_score") or 0.0) for row in rows if _architecture_family_for_row(row) == family), default=-1.0)
        for family in allowed_architecture_families
    }
    dominant_architecture = max(architecture_scores, key=architecture_scores.get, default="linear_baseline")
    alternate_architectures = _alternate_architectures(dominant_architecture, allowed_architecture_families)
    if not predictive_survivors:
        if current_generation <= 0 or len(current_feature_families) > 1:
            next_matrix = _bounded_matrix(
                allowed_dimensions=allowed_dimensions,
                architecture_families=[dominant_architecture],
                initial_years=[2, 3],
                feature_families=["price_core", "price_liquidity", "price_short_horizon"],
                data_families=allowed_data_families,
                data_profiles=["phase1_default"],
                family_size_cap=policy["family_size_cap"],
            )
            rationale.append(
                f"all runs failed predictive gates, so the next family narrows to {dominant_architecture} while sweeping feature families across the currently modeling-ready data lanes"
            )
        elif len(current_architecture_families) > 1 and len(current_data_profiles) > 1 and len(current_feature_families) == 1:
            next_matrix = _bounded_matrix(
                allowed_dimensions=allowed_dimensions,
                architecture_families=[dominant_architecture, *alternate_architectures],
                initial_years=current_initial_years or [best_initial_year],
                feature_families=_next_feature_family_candidates(current_feature_families[0]),
                data_families=allowed_data_families,
                data_profiles=["phase1_default"],
                family_size_cap=policy["family_size_cap"],
            )
            next_matrix.pop("validation.initial_train_years", None)
            rationale.append(
                "architecture and window sweeps still failed, so the next family pivots to different feature and data-family lanes while keeping multiple architectures in play"
            )
        else:
            next_matrix = _bounded_matrix(
                allowed_dimensions=allowed_dimensions,
                architecture_families=[dominant_architecture, *alternate_architectures],
                initial_years=[best_initial_year],
                feature_families=[best_feature_family],
                data_families=allowed_data_families,
                data_profiles=["phase1_default", "phase1_short_window", "phase1_long_window"],
                family_size_cap=policy["family_size_cap"],
            )
            next_matrix.pop("validation.initial_train_years", None)
            rationale.append(
                "feature sweeps still failed predictive gates, so the next family compares architectures and data-window profiles around the best surviving feature and data lane"
            )
    elif predictive_survivors and not shortlisted:
        best = predictive_survivors[0]
        best_architecture = _architecture_family_for_row(best)
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            architecture_families=[best_architecture],
            initial_years=sorted({int(best["matrix_values"].get("validation.initial_train_years", 2) or 2), 2, 3}),
            feature_families=[_feature_family_for_row(best), "price_core"],
            data_families=[_data_family_for_row(best), best_data_family],
            data_profiles=["phase1_default"],
            family_size_cap=policy["family_size_cap"],
            extra_dimension=("portfolio.cost_stress_multiplier", [2.0, 3.0]),
        )
        rationale.append("a predictive survivor failed backtest gates, so the next family tightens around the strongest architecture/data lane and stresses cost assumptions")
    else:
        best = shortlisted[0]
        best_architecture = _architecture_family_for_row(best)
        next_matrix = _bounded_matrix(
            allowed_dimensions=allowed_dimensions,
            architecture_families=[best_architecture],
            initial_years=sorted({int(best["matrix_values"].get("validation.initial_train_years", 2) or 2), 2, 3}),
            feature_families=[_feature_family_for_row(best), "price_core"],
            data_families=[_data_family_for_row(best)],
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
    architecture_families: list[str],
    initial_years: list[int],
    feature_families: list[str],
    data_families: list[str],
    data_profiles: list[str],
    family_size_cap: int,
    extra_dimension: tuple[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    matrix: dict[str, list[Any]] = {}
    unique_architectures = list(dict.fromkeys(architecture_families))
    if "architecture_family" in allowed_dimensions:
        matrix["architecture_family"] = unique_architectures[: max(1, min(len(unique_architectures), family_size_cap))]
    else:
        matrix["model_suite"] = [ARCHITECTURE_FAMILY_PRESETS[unique_architectures[0]]["model_suite"]]
    if "feature_family" in allowed_dimensions:
        current_size = _matrix_combination_count(matrix)
        max_features = max(1, family_size_cap // max(1, current_size * max(1, len(initial_years))))
        matrix["feature_family"] = list(dict.fromkeys(feature_families))[:max_features]
    if "data_family" in allowed_dimensions and data_families:
        current_size = _matrix_combination_count(matrix)
        max_data_families = max(1, family_size_cap // max(1, current_size * max(1, len(initial_years))))
        matrix["data_family"] = list(dict.fromkeys(data_families))[:max_data_families]
    if "validation.initial_train_years" in allowed_dimensions:
        max_years = max(
            1,
            family_size_cap // max(1, len(matrix.get("feature_family", [1])) * len(matrix.get("data_family", [1]))),
        )
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
    model_suite = str(row.get("model_suite") or "")
    if model_suite == "advanced":
        return "advanced_challenger"
    return "tree_challenger" if model_suite == "full" else "linear_baseline"


def _alternate_architectures(family: str, allowed_families: list[str] | None = None) -> list[str]:
    candidates = allowed_families or list(ARCHITECTURE_FAMILY_PRESETS)
    return [candidate for candidate in candidates if candidate != family]


def _best_classified_row(*, rows: list[dict[str, Any]], classification: str) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if str((row.get("candidate_autopsy") or {}).get("classification") or "") == classification
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: float(row.get("primary_score") or 0.0))


def _feature_family_for_row(row: dict[str, Any]) -> str:
    matrix_values = dict(row.get("matrix_values") or {})
    family = matrix_values.get("feature_family")
    if family:
        return str(family)
    return "price_liquidity"


def _data_family_for_row(row: dict[str, Any]) -> str:
    matrix_values = dict(row.get("matrix_values") or {})
    family = matrix_values.get("data_family")
    if family:
        return str(family)
    return "price_plus_liquidity"


def _proposal_data_families(*, base_spec: dict[str, Any], current_data_families: list[str]) -> list[str]:
    phase = int(base_spec.get("phase", 1) or 1)
    allowed = [
        family
        for family, preset in DATA_FAMILY_PRESETS.items()
        if bool(preset.get("modeling_ready")) and (phase > 1 or family in {"price_only", "price_plus_liquidity"})
    ]
    ordered = list(dict.fromkeys(current_data_families + allowed))
    return ordered or ["price_only"]


def _proposal_architecture_families(*, base_spec: dict[str, Any], current_architecture_families: list[str]) -> list[str]:
    phase = int(base_spec.get("ssot_phase", base_spec.get("phase", 1)) or 1)
    allowed = ["linear_baseline", "tree_challenger"]
    if phase > 1:
        allowed.append("advanced_challenger")
    ordered = list(dict.fromkeys(current_architecture_families + allowed))
    return ordered or ["linear_baseline", "tree_challenger"]


def _next_feature_family_candidates(current_family: str) -> list[str]:
    ordered = ["price_core", "price_liquidity", "price_short_horizon"]
    return [family for family in ordered if family != current_family] or ordered


def _write_next_family_proposal(*, local_state: Path, experiment_id: str, proposal: dict[str, Any]) -> dict[str, str]:
    root = _local_experiment_root(local_state, experiment_id)
    json_path = root / "next_family_proposal.json"
    md_path = root / "next_family_proposal.md"
    spec_path = root / "next_family_proposal.yml"
    _atomic_write_json(json_path, proposal)
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
    payload = {
        "experiment_id": experiment_id,
        "spec_path": str(spec_path),
        "pid": None,
        "status": "STARTING",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "heartbeat_at": datetime.now(tz=UTC).isoformat(),
        "completed_at": None,
        "poll_seconds": poll_seconds,
        "paused": False,
        "stop_requested": False,
        "stop_reason": None,
        "active_run_ids": [],
        "queue_counts": {},
        "log_path": str(log_path),
    }
    existing = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    payload = {**existing, **payload}
    if existing.get("queue_counts") and not payload.get("queue_counts"):
        payload["queue_counts"] = existing["queue_counts"]
    if existing.get("active_run_ids") and not payload.get("active_run_ids"):
        payload["active_run_ids"] = existing["active_run_ids"]
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=payload)
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
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=repo_root, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
    except Exception as exc:
        payload["status"] = "STOPPED"
        payload["last_error"] = str(exc)
        payload["completed_at"] = datetime.now(tz=UTC).isoformat()
        _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=payload)
        raise
    payload["pid"] = process.pid
    payload["status"] = "RUNNING"
    payload["heartbeat_at"] = datetime.now(tz=UTC).isoformat()
    _write_supervisor_state(local_state=local_state, experiment_id=experiment_id, payload=payload)
    return payload


def _ensure_supervisor_state(*, local_state: Path, experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    merged = {**payload, **state}
    if str(payload.get("status") or "").upper() in {"RUNNING", "STARTING"}:
        for key in ("completed_at", "stop_reason", "stop_requested"):
            if key in payload:
                merged[key] = payload[key]
    return merged


def _stop_active_training_runs(
    *,
    local_state: Path,
    experiment_id: str,
    run_ids: list[str],
    repo_root: Path | None,
    data_root: Path | None,
    targets_config_path: Path | None,
    python_executable: str,
) -> list[dict[str, Any]]:
    """Stop active training processes owned by an experiment supervisor."""
    if not run_ids or repo_root is None or data_root is None:
        return []
    stopped: list[dict[str, Any]] = []
    for run_id in run_ids:
        manifest_path = local_state / "experiments" / experiment_id / "runs" / f"{run_id}.json"
        if not manifest_path.exists():
            stopped.append({"run_id": run_id, "stopped": False, "reason": "missing run manifest"})
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = stop_training_process(
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                phase=int(manifest.get("phase") or 1),
                target=str(manifest.get("target") or "local"),
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                runtime_name=str(manifest.get("runtime_name") or f"{experiment_id}-{run_id}"),
            )
            stopped.append({"run_id": run_id, **result})
        except Exception as exc:  # noqa: BLE001
            stopped.append({"run_id": run_id, "stopped": False, "reason": str(exc)})
    return stopped


def _supervisor_state_path(*, local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiment_supervisors" / f"{experiment_id}.json"


def _supervisor_log_path(*, local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiment_supervisors" / "logs" / f"{experiment_id}.log"


def _write_supervisor_state(*, local_state: Path, experiment_id: str, payload: dict[str, Any]) -> None:
    path = _supervisor_state_path(local_state=local_state, experiment_id=experiment_id)
    _atomic_write_json(path, payload)


def _supervisor_heartbeat_stale(payload: dict[str, Any]) -> bool:
    """Return whether a supervisor heartbeat is too old to trust as running."""
    heartbeat = payload.get("heartbeat_at")
    if not heartbeat:
        return False
    try:
        heartbeat_at = datetime.fromisoformat(str(heartbeat))
    except ValueError:
        return True
    if heartbeat_at.tzinfo is None:
        heartbeat_at = heartbeat_at.replace(tzinfo=UTC)
    poll_seconds = int(payload.get("poll_seconds") or 30)
    max_age = timedelta(seconds=max(300, poll_seconds * 6))
    return datetime.now(tz=UTC) - heartbeat_at > max_age


def _run_ready_for_retry(*, manifest: dict[str, Any], supervision: dict[str, Any]) -> bool:
    status = str(manifest.get("status") or "PLANNED")
    if status == "PLANNED":
        return True
    if status != "FAILED":
        return False
    if str(manifest.get("failure_kind") or "") != "infra":
        return False
    return int(manifest.get("retry_count", 0) or 0) < int(supervision["max_retry_count"])


def _count_launchable_runs(*, runs: list[dict[str, Any]], supervision: dict[str, Any]) -> int:
    """Return how many runs are still eligible for launch or retry."""
    return sum(1 for run in runs if _run_ready_for_retry(manifest=run, supervision=supervision))


def _manifest_requires_runtime_refresh(manifest: dict[str, Any]) -> bool:
    """Return whether this manifest still needs live runtime polling."""
    status = str(manifest.get("status") or "").upper()
    if status in {"RUNNING", "STARTING", "UNKNOWN"}:
        return True
    if status == "FAILED" and str(manifest.get("failure_kind") or "") == "infra":
        return True
    if status == "FAILED" and str(manifest.get("last_error") or "") == "remote process is not running":
        return True
    return False


def _append_supervisor_event(manifest: dict[str, Any], *, event: str, payload: dict[str, Any]) -> None:
    history = list(manifest.get("supervisor_history") or [])
    history.append({"at": datetime.now(tz=UTC).isoformat(), "event": event, **payload})
    manifest["supervisor_history"] = history[-50:]


def _classify_failure(error: str) -> str:
    lowered = error.lower()
    if any(
        token in lowered
        for token in [
            "ssh",
            "permission denied",
            "connection refused",
            "connection reset",
            "host",
            "mount",
            "timed out",
            "no route",
            "local process is not running",
            "remote process is not running",
        ]
    ):
        return "infra"
    if any(token in lowered for token in ["modulenotfounderror", "importerror", "no module named"]):
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
        os.kill(pid, 0)
    except OSError:
        return False
    return True
