"""Perpetual research-program supervision on top of experiment families."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from trademl.experiments import (
    ARCHITECTURE_FAMILY_PRESETS,
    _atomic_write_json,
    _count_launchable_runs,
    _load_run_manifests,
    _refresh_experiment_summary,
    _supervision_policy,
    compare_experiment,
    latest_experiment_summary,
    propose_next_experiment_family,
    read_experiment_supervisor_state,
    stop_experiment_supervisor,
    supervise_experiment,
)
from trademl.fleet.launchd import launch_agent_status
from trademl.data_node.training_control import (
    _resolve_default_report_date,
    resolve_training_target,
    training_preflight,
    training_status_snapshot,
)
from trademl.modeling import (
    DEFAULT_FEATURE_VERSION,
    DEFAULT_LABEL_VERSION,
    build_modeling_artifacts,
    feature_label_preflight,
    modeling_artifact_metadata,
)
from trademl.portfolio.build import build_portfolio
from trademl.research_architecture import (
    architecture_registry_payload,
    complexity_adjusted_score,
    objective_registry_payload,
    resolve_architecture_entry,
)
from trademl.research_alerts import (
    DEFAULT_ALERT_POLICY,
    send_research_alert_email as send_research_alert_email,
    write_research_alerts,
)
from trademl.broker.alpaca_paper import (
    build_alpaca_order_payloads_from_deltas,
    paper_account_smoke_check,
    submit_alpaca_paper_payloads,
)

DEFAULT_RESEARCH_REVIEW_HOURS = 12
DEFAULT_BUDGET_POLICY = {
    "max_total_runs": 500,
    "max_total_hours": 168,
    "max_consecutive_non_improving_families": 4,
    "max_repeat_rejection_reason": 4,
    "max_low_novelty_families": 3,
    "min_novelty_score": 0.25,
    "max_infra_failures": 3,
}
DEFAULT_STEERING = {
    "prefer_architecture_families": [],
    "avoid_architecture_families": [],
    "prefer_data_families": [],
    "avoid_data_families": [],
    "freeze_phase": None,
    "force_pivot": False,
    "exploration_breadth": "normal",
}
MODELING_READY_DATA_FAMILIES = {
    "price_only",
    "price_plus_liquidity",
}
DEFAULT_EXHAUSTION_MODE = "wait_for_new_data"
DEFAULT_EXPANSION_PROFILES = ["phase1_default", "phase1_short_window", "phase1_long_window"]
DEFAULT_EXPANSION_INITIAL_TRAIN_YEARS = [2, 3]
DEFAULT_INCUMBENT_POLICY = {
    "mode": "research",
    "manual_live_required": True,
    "max_pbo": 0.5,
    "min_net_return": 0.0,
    "min_rank_ic_improvement": 0.002,
    "min_net_return_improvement": 0.01,
}
DEFAULT_DRIFT_POLICY = {
    "min_coverage": 0.98,
    "max_infra_failures": 3,
    "max_hours_without_completed_run": 24,
    "mature_windows": 3,
    "min_relative_ic": 0.5,
    "psi_warn": 0.2,
}
DEFAULT_PAPER_POLICY = {
    "enabled": True,
    "rebalance_day": "FRI",
    "no_live_orders": True,
    "pnl_horizon_trading_days": 5,
    "portfolio_notional": 100_000.0,
    "broker": {
        "provider": "alpaca_paper",
        "base_url": "https://paper-api.alpaca.markets/v2",
        "submit_orders_enabled": False,
        "api_key_env": "ALPACA_API_KEY",
        "api_secret_env": "ALPACA_API_SECRET",
    },
}
DEFAULT_FRONTIER_ARCHITECTURE_POLICY = {
    "enabled": False,
    "allow_phase1_advanced": False,
    "trigger_min_completed_runs": 100,
    "advanced_first": True,
    "family_size_cap": 6,
    "sentinel_baseline_runs": 2,
    "max_advanced_failures_per_epoch": 12,
    "require_dependency_preflight": True,
    "auto_pivot_on_brake": True,
}
DEFAULT_OBJECTIVE_POLICY = {
    "enabled": True,
    "primary": "research_profitability_v1",
    "complexity_penalty": {"enabled": True, "penalty_per_tier": 0.0005, "min_complexity_adjusted_improvement": 0.0},
}
DEFAULT_ARCHITECTURE_REGISTRY_POLICY = {"enabled": True}
DEFAULT_AUTONOMOUS_PROGRESSION_POLICY = {
    "enabled": True,
    "disabled_future_lanes": ["rl_policy", "sequence_transformer", "gnn", "foundation_forecaster"],
}


def start_research_program(
    *,
    program_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int | None = None,
    detach: bool = False,
) -> dict[str, Any]:
    """Start or spawn the perpetual research supervisor."""
    spec = _load_research_program_spec(program_path)
    program_id = str(spec["program_id"])
    if detach:
        return _spawn_program_supervisor_process(
            program_id=program_id,
            program_path=program_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            python_executable=python_executable,
            poll_seconds=int(poll_seconds or spec.get("poll_seconds") or 30),
        )
    with _research_program_lock(local_state=local_state, program_id=program_id) as acquired:
        if not acquired:
            state = read_research_program_state(local_state=local_state, program_id=program_id)
            return {
                "program_id": program_id,
                "status": state.get("status") or "RUNNING",
                "duplicate": True,
                "reason": "research program lock is already held",
                "pid": state.get("pid"),
            }
        return supervise_research_program(
            program_path=program_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
            poll_seconds=poll_seconds,
        )


def supervise_research_program(
    *,
    program_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int | None = None,
) -> dict[str, Any]:
    """Run the perpetual research loop until paused, stopped, or exhausted."""
    spec = _load_research_program_spec(program_path)
    program_id = str(spec["program_id"])
    resolved_poll = int(poll_seconds or spec.get("poll_seconds") or 30)
    state = _ensure_program_state(
        local_state=local_state,
        program_id=program_id,
        payload=_initial_program_state(spec=spec, program_path=program_path, poll_seconds=resolved_poll),
    )
    state = _sync_family_budget_with_existing_experiments(
        local_state=local_state,
        program_id=program_id,
        state=state,
    )
    state = _bootstrap_research_state_from_training_history(
        local_state=local_state,
        state=state,
    )
    state["frontier_architecture"] = _frontier_architecture_status(
        spec=spec,
        state=state,
        frontier=dict(state.get("frontier") or _empty_frontier()),
        experiment_summary={},
    )
    state["status"] = "RUNNING"
    state["stop_reason"] = None
    state["completed_at"] = None
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)

    while True:
        state = read_research_program_state(local_state=local_state, program_id=program_id)
        if not state:
            state = _initial_program_state(spec=spec, program_path=program_path, poll_seconds=resolved_poll)
        state["heartbeat_at"] = datetime.now(tz=UTC).isoformat()
        state["stale_run_sweep"] = sweep_stale_experiment_runs(
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        if state.get("stop_requested"):
            state["status"] = "STOPPED"
            state["completed_at"] = datetime.now(tz=UTC).isoformat()
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            return state
        if state.get("paused"):
            state["status"] = "PAUSED"
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue
        if state.get("status") in {"WAITING_FOR_DATA", "INFRA_BLOCKED"}:
            resumed = _resume_if_data_changed(
                spec=spec,
                state=state,
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                poll_seconds=resolved_poll,
            )
            if resumed is not None:
                state.update(resumed)
                if resumed.get("status") == "INFRA_BLOCKED":
                    state["waiting_since"] = state.get("waiting_since") or datetime.now(tz=UTC).isoformat()
                else:
                    state["status"] = "RUNNING"
                    state["wait_reason"] = None
                    state["waiting_since"] = None
                _write_program_state(local_state=local_state, program_id=program_id, payload=state)
                time.sleep(max(1, resolved_poll))
                continue
            if _review_packet_due(state=state, spec=spec):
                packet = write_research_review_packet(
                    program_id=program_id,
                    local_state=local_state,
                    repo_root=repo_root,
                    data_root=data_root,
                    targets_config_path=targets_config_path,
                    python_executable=python_executable,
                )
                state["latest_review_packet"] = packet
                state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue

        active_experiment = str(state.get("current_experiment_id") or "")
        if not active_experiment:
            next_spec = _build_initial_phase_experiment_spec(program_state=state, spec=spec)
            if next_spec is None:
                state["status"] = "STOPPED"
                state["stop_reason"] = "no_initial_phase_spec"
                _write_program_state(local_state=local_state, program_id=program_id, payload=state)
                return state
            launched = _launch_program_family(
                program_id=program_id,
                program_state=state,
                next_spec=next_spec,
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                poll_seconds=resolved_poll,
            )
            state.update(launched)
            if launched.get("status") == "INFRA_BLOCKED":
                state["waiting_since"] = datetime.now(tz=UTC).isoformat()
            else:
                state["status"] = "RUNNING"
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue

        experiment_supervisor = read_experiment_supervisor_state(local_state=local_state, experiment_id=active_experiment)
        if experiment_supervisor and experiment_supervisor.get("status") not in {"COMPLETED", "STOPPED"}:
            state["status"] = "RUNNING"
            state["active_experiment_supervisor"] = experiment_supervisor
            if _review_packet_due(state=state, spec=spec):
                packet = write_research_review_packet(
                    program_id=program_id,
                    local_state=local_state,
                    repo_root=repo_root,
                    data_root=data_root,
                    targets_config_path=targets_config_path,
                    python_executable=python_executable,
                )
                state["latest_review_packet"] = packet
                state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue

        experiment_summary = latest_experiment_summary(local_state=local_state)
        if experiment_summary.get("experiment_id") != active_experiment:
            experiment_summary = _read_experiment_summary_direct(local_state=local_state, experiment_id=active_experiment)
        recovered = _recover_stalled_active_experiment(
            state=state,
            experiment_supervisor=experiment_supervisor,
            experiment_summary=experiment_summary,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
            poll_seconds=resolved_poll,
        )
        if recovered is not None:
            state["status"] = "RUNNING"
            state["stop_reason"] = None
            state["wait_reason"] = None
            state["active_experiment_supervisor"] = recovered
            state["last_transition"] = {
                "action": "recover_family",
                "reason": f"restarting stalled experiment supervisor for {active_experiment}",
            }
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            time.sleep(max(1, resolved_poll))
            continue
        proposal_result = propose_next_experiment_family(
            experiment_id=active_experiment,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        proposal = proposal_result.get("proposal") or {}
        frontier = _update_frontier_memory(state=state, experiment_summary=experiment_summary)
        decision = _determine_program_transition(spec=spec, state=state, frontier=frontier, experiment_summary=experiment_summary, proposal=proposal)
        state["frontier"] = frontier
        state["frontier_architecture"] = _frontier_architecture_status(
            spec=spec,
            state=state,
            frontier=frontier,
            experiment_summary=experiment_summary,
        )
        state["best_candidate_summary"] = _program_best_candidate(frontier=frontier, experiment_summary=experiment_summary)
        state["last_transition"] = decision
        packet = None
        if decision["action"] in {"advance_phase", "stop", "pause"} or _review_packet_due(state=state, spec=spec):
            packet = write_research_review_packet(
                program_id=program_id,
                local_state=local_state,
                repo_root=repo_root,
                data_root=data_root,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
            )
            state["latest_review_packet"] = packet
            state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()

        if decision["action"] == "launch_family":
            launched = _launch_program_family(
                program_id=program_id,
                program_state=state,
                next_spec=dict(decision["next_spec"]),
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                poll_seconds=resolved_poll,
            )
            state.update(launched)
            if launched.get("status") == "INFRA_BLOCKED":
                state["waiting_since"] = datetime.now(tz=UTC).isoformat()
            else:
                state["status"] = "RUNNING"
        elif decision["action"] == "advance_phase":
            state["current_phase"] = int(decision["next_phase"])
            state["current_ssot_phase"] = int(decision["next_phase"])
            state["current_campaign_track"] = str((decision.get("next_spec") or {}).get("campaign_track") or state.get("current_campaign_track") or "baseline")
            state["current_track"] = state["current_campaign_track"]
            launched = _launch_program_family(
                program_id=program_id,
                program_state=state,
                next_spec=dict(decision["next_spec"]),
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                env_path=env_path,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                poll_seconds=resolved_poll,
            )
            state.update(launched)
            if launched.get("status") == "INFRA_BLOCKED":
                state["waiting_since"] = datetime.now(tz=UTC).isoformat()
            else:
                state["status"] = "RUNNING"
        elif decision["action"] == "wait_for_data":
            state["status"] = "WAITING_FOR_DATA"
            state["wait_reason"] = decision["reason"]
            state["waiting_since"] = datetime.now(tz=UTC).isoformat()
            state["pending_next_spec"] = decision.get("next_spec")
            if state.get("last_seen_data_revision") is None:
                state["last_seen_data_revision"] = _current_data_revision(
                    spec=spec,
                    state=state,
                    repo_root=repo_root,
                    data_root=data_root,
                    local_state=local_state,
                    targets_config_path=targets_config_path,
                    python_executable=python_executable,
                )
        else:
            state["status"] = "STOPPED" if decision["action"] == "stop" else "PAUSED"
            state["stop_reason"] = decision["reason"]
            state["completed_at"] = datetime.now(tz=UTC).isoformat()
            _write_program_state(local_state=local_state, program_id=program_id, payload=state)
            return state

        _write_program_state(local_state=local_state, program_id=program_id, payload=state)
        time.sleep(max(1, resolved_poll))


def read_research_program_state(*, local_state: Path, program_id: str) -> dict[str, Any]:
    """Return the persisted research-program state when present."""
    path = _program_state_path(local_state=local_state, program_id=program_id)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    pid = payload.get("pid")
    status = str(payload.get("status") or "").upper()
    active_statuses = {"RUNNING", "PAUSED", "STARTING", "STOPPING"}
    stopped_reason: str | None = None
    if isinstance(pid, int) and status in active_statuses and not _is_local_process_running(pid):
        stopped_reason = f"local supervisor pid {pid} is not running"
    elif status in active_statuses and _program_heartbeat_stale(payload):
        stopped_reason = "research supervisor heartbeat is stale"
    if stopped_reason:
        payload["status"] = "STOPPED"
        payload["stop_reason"] = stopped_reason
        payload["completed_at"] = payload.get("completed_at") or datetime.now(tz=UTC).isoformat()
        _write_program_state(local_state=local_state, program_id=program_id, payload=payload)
    return payload


def latest_research_program_summary(*, local_state: Path) -> dict[str, Any]:
    """Return the most recently updated research-program state."""
    roots = sorted((local_state / "research_programs").glob("*/program_state.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not roots:
        return {}
    return json.loads(roots[0].read_text(encoding="utf-8"))


def sweep_stale_experiment_runs(
    *,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Reconcile stale RUNNING/STARTING experiment manifests across all families."""
    active_statuses = {"RUNNING", "STARTING"}
    checked = 0
    reconciled = 0
    touched: set[str] = set()
    errors: list[dict[str, Any]] = []
    for path in sorted((local_state / "experiments").glob("*/runs/*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        status = str(manifest.get("status") or "").upper()
        if status not in active_statuses:
            continue
        checked += 1
        experiment_id = str(manifest.get("experiment_id") or path.parent.parent.name)
        try:
            snapshot = training_status_snapshot(
                repo_root=repo_root,
                data_root=data_root,
                local_state=local_state,
                phase=int(manifest.get("phase") or 1),
                target=str(manifest.get("target") or "local"),
                targets_config_path=targets_config_path,
                python_executable=python_executable,
                runtime_name=str(manifest.get("runtime_name") or f"{experiment_id}-{manifest.get('run_id')}"),
                tail_lines=20,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "run_id": manifest.get("run_id"), "error": str(exc)})
            continue
        runtime = dict(snapshot.get("runtime") or {})
        runtime_status = str(runtime.get("status") or "").lower()
        running = bool(runtime.get("running"))
        report_path_value = manifest.get("report_path")
        report_exists = bool(report_path_value) and Path(str(report_path_value)).exists()
        updated = dict(manifest)
        updated["runtime"] = runtime
        updated["log_tail"] = snapshot.get("log_tail", [])
        if runtime_status in {"completed", "succeeded", "success"} or report_exists:
            updated["status"] = "COMPLETED"
            updated["completed_at"] = runtime.get("finished_at") or datetime.now(tz=UTC).isoformat()
        elif not running:
            updated["status"] = "FAILED"
            updated["failure_kind"] = "infra"
            updated["last_error"] = str(runtime.get("error") or f"stale {status.lower()} manifest; runtime is not running")
            updated["completed_at"] = runtime.get("finished_at") or datetime.now(tz=UTC).isoformat()
        else:
            _atomic_write_json(path, updated)
            continue
        history = list(updated.get("supervisor_history") or [])
        history.append({"at": datetime.now(tz=UTC).isoformat(), "event": "stale_run_sweep", "status": updated["status"]})
        updated["supervisor_history"] = history[-50:]
        _atomic_write_json(path, updated)
        reconciled += 1
        touched.add(experiment_id)
    refreshed = []
    for experiment_id in sorted(touched):
        runs = _load_run_manifests(local_state=local_state, experiment_id=experiment_id)
        counts: dict[str, int] = {}
        for run in runs:
            key = str(run.get("status") or "UNKNOWN").upper()
            counts[key] = counts.get(key, 0) + 1
        summary = _refresh_experiment_summary(
            local_state=local_state,
            experiment_id=experiment_id,
            summary={"experiment_id": experiment_id, "runs": runs, "counts": counts},
            data_root=data_root,
        )
        refreshed.append({"experiment_id": experiment_id, "counts": summary.get("counts", {})})
    return {
        "checked": checked,
        "reconciled": reconciled,
        "refreshed": refreshed,
        "errors": errors,
        "ran_at": datetime.now(tz=UTC).isoformat(),
    }


def read_research_incumbent(*, local_state: Path, program_id: str) -> dict[str, Any]:
    """Return the canonical local research incumbent record when present."""
    path = _incumbent_path(local_state=local_state, program_id=program_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_research_incumbent(
    *,
    program_id: str,
    local_state: Path,
    data_root: Path,
    candidates: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Promote a research incumbent only when strict predictive/backtest gates pass."""
    merged_policy = {**DEFAULT_INCUMBENT_POLICY, **dict(policy or {})}
    current = read_research_incumbent(local_state=local_state, program_id=program_id)
    rejections: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (float(item.get("primary_score") or 0.0), float(item.get("backtest_net_return") or 0.0)), reverse=True):
        reasons = _incumbent_rejection_reasons(candidate=candidate, incumbent=current, policy=merged_policy)
        if reasons:
            rejections.append({"run_id": candidate.get("run_id"), "experiment_id": candidate.get("experiment_id"), "reasons": reasons})
            continue
        payload = {
            **candidate,
            "program_id": program_id,
            "mode": "research",
            "manual_live_required": bool(merged_policy.get("manual_live_required", True)),
            "promoted_at": datetime.now(tz=UTC).isoformat(),
            "previous_incumbent": {
                key: current.get(key)
                for key in ("experiment_id", "run_id", "primary_score", "backtest_net_return", "promoted_at")
                if current.get(key) is not None
            },
            "last_rejections": rejections,
        }
        _write_incumbent_payload(local_state=local_state, data_root=data_root, program_id=program_id, payload=payload)
        return {"promoted": True, "incumbent": payload, "rejections": rejections}
    if current:
        current["last_rejections"] = rejections
        current["last_reviewed_at"] = datetime.now(tz=UTC).isoformat()
        _write_incumbent_payload(local_state=local_state, data_root=data_root, program_id=program_id, payload=current)
    return {"promoted": False, "incumbent": current, "rejections": rejections}


def write_paper_outputs_for_incumbent(
    *,
    incumbent: dict[str, Any],
    data_root: Path,
    local_state: Path,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic paper-ready signals, weights, and order deltas for the incumbent."""
    return _write_paper_outputs(
        source=incumbent,
        data_root=data_root,
        local_state=local_state,
        policy=policy,
        shared_family="paper",
        local_family="paper",
        path_prefix="",
        non_incumbent=False,
    )


def write_shadow_paper_outputs_for_candidate(
    *,
    candidate: dict[str, Any],
    data_root: Path,
    local_state: Path,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic non-tradable shadow paper outputs for a non-incumbent candidate."""
    return _write_paper_outputs(
        source=candidate,
        data_root=data_root,
        local_state=local_state,
        policy=policy,
        shared_family="shadow_paper",
        local_family="shadow_paper",
        path_prefix="shadow_",
        non_incumbent=True,
    )


def _write_paper_outputs(
    *,
    source: dict[str, Any],
    data_root: Path,
    local_state: Path,
    policy: dict[str, Any] | None,
    shared_family: str,
    local_family: str,
    path_prefix: str,
    non_incumbent: bool,
) -> dict[str, Any]:
    """Write deterministic paper-style signals, weights, and order deltas."""
    merged_policy = _paper_policy(policy)
    if not bool(merged_policy.get("enabled", True)):
        return {"status": "skipped", "reason": "paper policy disabled"}
    if not bool(merged_policy.get("no_live_orders", True)):
        raise ValueError("paper_policy.no_live_orders must remain true in research autopilot")
    artifacts = dict(source.get("artifacts") or {})
    predictions_path = Path(str(artifacts.get("primary_predictions_path") or source.get("primary_predictions_path") or ""))
    if not predictions_path.exists():
        return {"status": "skipped", "reason": f"missing predictions parquet: {predictions_path}"}
    predictions = pd.read_parquet(predictions_path)
    required = {"date", "symbol", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions parquet missing required columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest_available_date = frame["date"].max()
    latest_date = _latest_paper_signal_date(
        predictions=frame,
        rebalance_day=str(merged_policy.get("rebalance_day", "FRI")),
    )
    latest = frame.loc[frame["date"] == latest_date, ["date", "symbol", "prediction"]].copy()
    latest = latest.sort_values(["prediction", "symbol"], ascending=[False, True]).reset_index(drop=True)
    latest["score"] = latest["prediction"].astype(float)
    latest["rank"] = range(1, len(latest) + 1)
    latest["source_run_id"] = source.get("run_id")
    signals = latest[["date", "symbol", "score", "rank", "source_run_id"]].copy()
    targets = build_portfolio(
        signals[["date", "symbol", "score"]],
        {"rebalance_day": _rebalance_day_for_signal_date(latest_date)},
    )
    previous = _latest_prior_paper_targets(data_root=data_root, local_state=local_state, current_date=latest_date.date().isoformat())
    orders = _paper_order_deltas(targets=targets, previous=previous)
    root = _shared_research_root(data_root=data_root) / shared_family / latest_date.date().isoformat()
    root.mkdir(parents=True, exist_ok=True)
    signals_path = root / f"{path_prefix}signals.parquet"
    targets_path = root / f"{path_prefix}target_weights.parquet"
    orders_path = root / f"{path_prefix}paper_orders.parquet"
    signals.to_parquet(signals_path, index=False)
    targets.to_parquet(targets_path, index=False)
    orders.to_parquet(orders_path, index=False)
    local_root = _local_research_root(local_state=local_state) / local_family / latest_date.date().isoformat()
    local_root.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(local_root / f"{path_prefix}signals.parquet", index=False)
    targets.to_parquet(local_root / f"{path_prefix}target_weights.parquet", index=False)
    orders.to_parquet(local_root / f"{path_prefix}paper_orders.parquet", index=False)
    payloads = write_alpaca_paper_order_payloads(
        paper_orders_path=orders_path,
        output_dir=root,
        policy=merged_policy,
        prefix=path_prefix,
        source_run_id=str(source.get("run_id") or "unknown"),
    )
    payload = {
        "status": "written",
        "date": latest_date.date().isoformat(),
        "latest_prediction_date": latest_available_date.date().isoformat(),
        "signals_path": str(signals_path),
        "target_weights_path": str(targets_path),
        "paper_orders_path": str(orders_path),
        "local_root": str(local_root),
        "no_live_orders": True,
        "paper_order_payloads_path": payloads.get("payloads_path"),
        "paper_broker": payloads.get("broker"),
    }
    if non_incumbent:
        payload.update(
            {
                "non_incumbent": True,
                "not_trade_approved": True,
                "shadow_signals_path": str(signals_path),
                "shadow_target_weights_path": str(targets_path),
                "shadow_orders_path": str(orders_path),
                "shadow_order_payloads_path": payloads.get("payloads_path"),
            }
        )
    return payload


def _latest_paper_signal_date(*, predictions: pd.DataFrame, rebalance_day: str) -> pd.Timestamp:
    """Return the latest prediction date eligible for the paper rebalance policy."""
    dates = pd.to_datetime(predictions["date"].dropna().unique())
    if len(dates) == 0:
        raise ValueError("predictions parquet has no usable dates")
    normalized = pd.Series(dates).dt.normalize().sort_values()
    weekday_lookup = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
    weekday = weekday_lookup.get(str(rebalance_day).upper()[:3], 4)
    eligible = normalized.loc[normalized.dt.weekday == weekday]
    if not eligible.empty:
        return pd.Timestamp(eligible.iloc[-1])
    return pd.Timestamp(normalized.iloc[-1])


def _rebalance_day_for_signal_date(signal_date: pd.Timestamp) -> str:
    """Return a rebalance-day token that preserves the selected signal date."""
    codes = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    return codes[int(pd.Timestamp(signal_date).weekday())]


def write_alpaca_paper_order_payloads(
    *,
    paper_orders_path: Path,
    output_dir: Path,
    policy: dict[str, Any] | None = None,
    prefix: str = "",
    source_run_id: str = "unknown",
) -> dict[str, Any]:
    """Write Alpaca-compatible paper order payloads without submitting them."""
    merged_policy = _paper_policy(policy)
    if not bool(merged_policy.get("no_live_orders", True)):
        raise ValueError("paper payload generation requires no_live_orders=true")
    broker = dict(merged_policy.get("broker") or {})
    if str(broker.get("provider") or "alpaca_paper") != "alpaca_paper":
        return {"status": "skipped", "reason": "unsupported paper broker", "broker": broker.get("provider")}
    client_prefix = f"trademl-{Path(str(paper_orders_path)).parent.name}-{source_run_id}".replace("_", "-")[:40]
    payloads = build_alpaca_order_payloads_from_deltas(
        orders_path=paper_orders_path,
        portfolio_notional=float(merged_policy.get("portfolio_notional") or 100_000.0),
        client_order_prefix=client_prefix,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads_path = output_dir / f"{prefix}paper_order_payloads.json"
    result = {
        "status": "written",
        "broker": "alpaca_paper",
        "payloads_path": str(payloads_path),
        "payload_count": len(payloads),
        "submit_orders_enabled": bool(broker.get("submit_orders_enabled", False)),
        "no_live_orders": True,
        "payloads": payloads,
    }
    _atomic_write_json(payloads_path, result)
    return {key: value for key, value in result.items() if key != "payloads"}


def paper_account_smoke(
    *,
    policy: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a read-only Alpaca paper smoke check when credentials are configured."""
    merged_policy = _paper_policy(policy)
    return paper_account_smoke_check(policy=merged_policy, environ=environ)


def run_and_persist_paper_account_smoke(
    *,
    program_path: Path,
    local_state: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a read-only paper account smoke check and persist the safe result."""
    spec = _load_research_program_spec(program_path)
    program_id = str(spec.get("program_id") or program_path.stem)
    payload = paper_account_smoke(
        policy=dict(spec.get("paper_policy") or {}),
        environ=environ,
    )
    payload = {
        **payload,
        "checked_at": datetime.now(tz=UTC).isoformat(),
        "read_only": True,
    }
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    state["program_id"] = program_id
    state["latest_paper_account_smoke"] = payload
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return payload


def submit_paper_orders(
    *,
    payloads_path: Path,
    policy: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Submit previously generated Alpaca paper order payloads with explicit guards."""
    merged_policy = _paper_policy(policy)
    payload = json.loads(payloads_path.read_text(encoding="utf-8"))
    if not bool(payload.get("no_live_orders", True)):
        raise ValueError("paper payload file must declare no_live_orders=true")
    if str(payload.get("broker") or "") != "alpaca_paper":
        raise ValueError("only alpaca_paper payload files are supported")
    submissions = submit_alpaca_paper_payloads(
        payloads=list(payload.get("payloads") or []),
        policy=merged_policy,
        environ=environ,
    )
    result = {
        **submissions,
        "payloads_path": str(payloads_path),
        "no_live_orders": True,
        "submitted_at": datetime.now(tz=UTC).isoformat(),
    }
    submissions_path = payloads_path.with_name(payloads_path.stem.replace("payloads", "submissions") + ".json")
    _atomic_write_json(submissions_path, result)
    result["submissions_path"] = str(submissions_path)
    return result


def evaluate_paper_pnl(
    *,
    paper_outputs: dict[str, Any],
    data_root: Path,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate mature paper/shadow portfolio PnL from curated PIT-safe prices."""
    merged_policy = _paper_policy(policy)
    if not paper_outputs or paper_outputs.get("status") != "written":
        return {"status": "pending", "reason": "no paper outputs written", "no_live_orders": True}
    targets_path = Path(
        str(
            paper_outputs.get("target_weights_path")
            or paper_outputs.get("shadow_target_weights_path")
            or ""
        )
    )
    if not targets_path.exists():
        return {"status": "pending", "reason": f"missing target weights parquet: {targets_path}", "no_live_orders": True}
    targets = pd.read_parquet(targets_path)
    if targets.empty:
        return {"status": "pending", "reason": "target weights are empty", "no_live_orders": True}
    signal_date = pd.to_datetime(targets["date"].max()).normalize()
    horizon = int(merged_policy.get("pnl_horizon_trading_days") or 5)
    symbols = sorted({str(symbol).upper() for symbol in targets["symbol"].dropna().tolist()})
    benchmark_symbol = str(merged_policy.get("benchmark_symbol") or "SPY").upper()
    prices = _load_curated_prices_for_pnl(data_root=data_root, symbols=sorted(set(symbols + [benchmark_symbol])), start_date=signal_date)
    available_dates = sorted(pd.to_datetime(prices["date"].dropna().unique())) if not prices.empty else []
    future_dates = [pd.Timestamp(value).normalize() for value in available_dates if pd.Timestamp(value).normalize() > signal_date]
    if len(future_dates) < horizon:
        return {
            "status": "pending_labels",
            "signal_date": signal_date.date().isoformat(),
            "available_future_dates": len(future_dates),
            "required_future_dates": horizon,
            "no_live_orders": True,
            "source": "shadow" if paper_outputs.get("non_incumbent") else "paper",
        }
    mature_date = future_dates[horizon - 1]
    returns = _symbol_returns_from_prices(prices=prices, start_date=signal_date, end_date=mature_date)
    weighted = targets[["symbol", "target_weight"]].copy()
    weighted["symbol"] = weighted["symbol"].astype(str).str.upper()
    weighted = weighted.merge(returns.rename("return"), left_on="symbol", right_index=True, how="left")
    weighted["return"] = pd.to_numeric(weighted["return"], errors="coerce").fillna(0.0)
    weighted["target_weight"] = pd.to_numeric(weighted["target_weight"], errors="coerce").fillna(0.0)
    gross_return = float((weighted["target_weight"] * weighted["return"]).sum())
    turnover = float(pd.to_numeric(targets.get("target_weight", pd.Series(dtype=float)), errors="coerce").abs().sum())
    cost_bps = float(merged_policy.get("cost_spread_bps", 5.0) or 5.0)
    cost_drag = turnover * cost_bps / 10_000.0
    benchmark_return = _safe_float(returns.get(benchmark_symbol))
    payload = {
        "status": "available",
        "source": "shadow" if paper_outputs.get("non_incumbent") else "paper",
        "signal_date": signal_date.date().isoformat(),
        "mature_date": mature_date.date().isoformat(),
        "horizon_trading_days": horizon,
        "gross_return": gross_return,
        "cost_drag": cost_drag,
        "net_return": gross_return - cost_drag,
        "turnover": turnover,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_return": benchmark_return,
        "excess_return": (gross_return - cost_drag - benchmark_return) if benchmark_return is not None else None,
        "positions": int(len(weighted)),
        "no_live_orders": True,
        "non_incumbent": bool(paper_outputs.get("non_incumbent", False)),
    }
    pnl_path = targets_path.parent / ("shadow_paper_pnl.json" if paper_outputs.get("non_incumbent") else "paper_pnl.json")
    _atomic_write_json(pnl_path, payload)
    payload["path"] = str(pnl_path)
    return payload


def _paper_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_PAPER_POLICY)
    incoming = dict(policy or {})
    broker = {**dict(merged.get("broker") or {}), **dict(incoming.get("broker") or {})}
    merged.update(incoming)
    merged["broker"] = broker
    return merged


def evaluate_research_drift(
    *,
    program_id: str,
    state: dict[str, Any],
    incumbent: dict[str, Any],
    latest_summary: dict[str, Any],
    policy: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return alert records for coverage, infra, and mature-performance drift."""
    merged_policy = {**DEFAULT_DRIFT_POLICY, **dict(policy or {})}
    now = now or datetime.now(tz=UTC)
    alerts: list[dict[str, Any]] = []
    coverage = _latest_summary_coverage(latest_summary)
    if coverage is not None and coverage < float(merged_policy["min_coverage"]):
        alerts.append(
            {
                "program_id": program_id,
                "kind": "coverage",
                "severity": "warning",
                "message": f"latest research coverage {coverage:.3f} is below {float(merged_policy['min_coverage']):.3f}",
                "value": coverage,
            }
        )
    failed = int((latest_summary.get("counts") or {}).get("FAILED", 0) or 0)
    if failed >= int(merged_policy["max_infra_failures"]):
        alerts.append(
            {
                "program_id": program_id,
                "kind": "infra_failures",
                "severity": "critical",
                "message": f"{failed} failed runs in latest experiment summary",
                "value": failed,
            }
        )
    if str(state.get("status") or "").upper() == "RUNNING":
        last_completed = state.get("last_completed_run_at")
        if last_completed:
            with contextlib.suppress(ValueError):
                elapsed = now - datetime.fromisoformat(str(last_completed))
                if elapsed >= timedelta(hours=float(merged_policy["max_hours_without_completed_run"])):
                    alerts.append(
                        {
                            "program_id": program_id,
                            "kind": "stalled_research",
                            "severity": "warning",
                            "message": f"no completed runs for {elapsed.total_seconds() / 3600:.1f} hours while runnable",
                            "value": elapsed.total_seconds() / 3600,
                        }
                    )
    mature_ics = [float(value) for value in state.get("mature_window_rank_ics", [])[-int(merged_policy["mature_windows"]):]]
    incumbent_ic = float(incumbent.get("primary_score") or incumbent.get("rank_ic") or 0.0)
    if len(mature_ics) >= int(merged_policy["mature_windows"]) and incumbent_ic > 0:
        threshold = incumbent_ic * float(merged_policy["min_relative_ic"])
        if all(value <= 0.0 or value <= threshold for value in mature_ics):
            alerts.append(
                {
                    "program_id": program_id,
                    "kind": "mature_ic_drift",
                    "severity": "critical",
                    "message": "mature label IC degraded across consecutive windows",
                    "value": mature_ics,
                }
            )
    for alert in alerts:
        alert.setdefault("created_at", now.isoformat())
    return alerts


def research_health(
    *,
    program_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Return the operator health surface for a research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    spec = _load_research_program_spec(Path(str(state.get("spec_path")))) if state.get("spec_path") else {}
    state = _refresh_last_canary_status(local_state=local_state, state=state)
    current_experiment_id = str(state.get("current_experiment_id") or "")
    incumbent = read_research_incumbent(local_state=local_state, program_id=program_id)
    sweep = sweep_stale_experiment_runs(
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    latest_summary = _read_experiment_summary_direct(local_state=local_state, experiment_id=current_experiment_id) if current_experiment_id else {}
    active_run = _active_experiment_run_summary(latest_summary)
    modeling_metadata = modeling_artifact_metadata(data_root=data_root)
    alerts = evaluate_research_drift(
        program_id=program_id,
        state=state,
        incumbent=incumbent,
        latest_summary=latest_summary,
        policy={},
    )
    return {
        "program_id": program_id,
        "status": state.get("status"),
        "wait_reason": state.get("wait_reason"),
        "current_experiment_id": current_experiment_id,
        "stale_run_sweep": sweep,
        "incumbent": incumbent,
        "alerts": alerts,
        "paper": state.get("latest_paper_outputs", {}),
        "latest_paper_outputs": state.get("latest_paper_outputs", {}),
        "latest_shadow_paper_outputs": state.get("latest_shadow_paper_outputs", {}),
        "latest_paper_pnl": state.get("latest_paper_pnl", {}),
        "latest_paper_account_smoke": state.get("latest_paper_account_smoke", {}),
        "latest_paper_submission": state.get("latest_paper_submission", {}),
        "last_canary": state.get("last_canary", {}),
        "infra_blocker": state.get("last_infra_preflight", {}),
        "launchd": _research_launchd_status(program_id),
        "frontier_architecture": _frontier_architecture_status(spec=spec, state=state, frontier=dict(state.get("frontier") or {}), experiment_summary=latest_summary) if spec else {},
        "architecture_lane": state.get("architecture_lane"),
        "complexity_tier": state.get("complexity_tier"),
        "objective_verdict": state.get("objective_verdict", {}),
        "pivot_reason": (state.get("last_transition") or {}).get("reason"),
        "next_lane": state.get("next_lane") or (state.get("frontier") or {}).get("next_lane"),
        "sentinel_delta": state.get("sentinel_delta") or (state.get("frontier") or {}).get("sentinel_delta"),
        "autonomous_progression": state.get("autonomous_progression", {}),
        "dependency_preflight": (state.get("last_infra_preflight") or {}).get("dependencies", {}),
        "feature_label_readiness": (state.get("last_infra_preflight") or {}).get("feature_label", {}),
        "modeling": {
            "feature_version": modeling_metadata.get("feature_version"),
            "feature_set": modeling_metadata.get("feature_set"),
            "label_version": modeling_metadata.get("label_version"),
            "label_horizons": modeling_metadata.get("label_horizons"),
            "data_revision": modeling_metadata.get("data_revision"),
            "current_label_horizon": latest_summary.get("label_horizon") or active_run.get("label_horizon"),
            "current_feature_version": latest_summary.get("feature_version") or active_run.get("feature_version"),
            "current_portfolio_profile": latest_summary.get("portfolio_profile") or active_run.get("portfolio_profile"),
        },
        "paths": _research_path_summary(local_state=local_state, data_root=data_root),
    }


def _active_experiment_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the most relevant current run row from an experiment summary."""
    status_order = {"RUNNING": 0, "STARTING": 1, "PLANNED": 2, "COMPLETED": 3}
    rows = [dict(row) for row in list(summary.get("runs") or []) if isinstance(row, dict)]
    if not rows:
        return {}
    rows.sort(
        key=lambda row: (
            status_order.get(str(row.get("status") or ""), 99),
            int(row.get("phase") or 0),
            str(row.get("run_id") or ""),
        )
    )
    return rows[0]


def build_research_features(
    *,
    program_path: Path,
    data_root: Path,
    feature_version: str | None = None,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Build modeling-ready feature and label artifacts for a research program."""
    spec = _load_research_program_spec(program_path)
    modeling = dict(spec.get("modeling") or {})
    config_path = Path(__file__).resolve().parents[2] / "configs" / "equities_xs.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return build_modeling_artifacts(
        data_root=data_root,
        feature_config=dict(config.get("features") or {}),
        feature_set=str(modeling.get("feature_set") or "daily_price_liquidity_v1"),
        feature_version=str(feature_version or modeling.get("feature_version") or DEFAULT_FEATURE_VERSION),
        label_definition=str(modeling.get("label_definition") or "universe_relative_forward_return"),
        label_version=str(modeling.get("label_version") or DEFAULT_LABEL_VERSION),
        label_horizons=[int(item) for item in list(modeling.get("label_horizons") or [1, 5, 20])],
        report_date=report_date,
    )


def run_research_canary(
    *,
    program_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int | None = None,
    detach: bool = False,
    feature_version: str | None = None,
    label_horizon: int | None = None,
) -> dict[str, Any]:
    """Run one bounded architecture canary family for the research autopilot."""
    spec = _load_research_program_spec(program_path)
    program_id = str(spec["program_id"])
    resolved_poll = int(poll_seconds or spec.get("poll_seconds") or 30)
    state = _ensure_program_state(
        local_state=local_state,
        program_id=program_id,
        payload=_initial_program_state(spec=spec, program_path=program_path, poll_seconds=resolved_poll),
    )
    state = _sync_family_budget_with_existing_experiments(
        local_state=local_state,
        program_id=program_id,
        state=state,
    )
    phase_policy = _phase_policy(spec=spec, phase=_current_ssot_phase(state=state, spec=spec))
    if phase_policy is None:
        payload = {"status": "INFRA_BLOCKED", "reason": "missing phase policy for canary", "program_id": program_id}
        state.update(payload)
        _write_program_state(local_state=local_state, program_id=program_id, payload=state)
        return payload
    next_spec = _canary_experiment_spec(
        spec=spec,
        state=state,
        phase_policy=phase_policy,
        feature_version=feature_version,
        label_horizon=label_horizon,
    )
    launched = _launch_program_family(
        program_id=program_id,
        program_state=state,
        next_spec=next_spec,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=resolved_poll,
        detach=detach,
    )
    state.update(launched)
    state["last_canary"] = {
        "experiment_id": next_spec["experiment_id"],
        "status": launched.get("status") or launched.get("active_experiment_supervisor", {}).get("status"),
        "ran_at": datetime.now(tz=UTC).isoformat(),
        "architecture_families": list(next_spec["matrix"].get("architecture_family") or []),
    }
    if launched.get("status") == "INFRA_BLOCKED":
        _write_program_state(local_state=local_state, program_id=program_id, payload=state)
        return {**launched, "program_id": program_id, "canary_spec": next_spec}
    paper_smoke = paper_account_smoke(policy=dict(spec.get("paper_policy") or {}))
    review_packet = {}
    if not detach:
        review_packet = write_research_review_packet(
            program_id=program_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
        state["latest_review_packet"] = review_packet
        state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()
    state["latest_paper_account_smoke"] = paper_smoke
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return {
        **launched,
        "program_id": program_id,
        "canary_spec": next_spec,
        "paper_account_smoke": paper_smoke,
        "review_packet": review_packet,
    }


def list_research_alerts(*, local_state: Path, program_id: str, limit: int = 20) -> dict[str, Any]:
    """Return recent research alert files."""
    root = _local_research_root(local_state=local_state) / "alerts"
    paths = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]
    return {
        "program_id": program_id,
        "alerts": [json.loads(path.read_text(encoding="utf-8")) for path in paths],
        "paths": [str(path) for path in paths],
    }


def _research_launchd_status(program_id: str) -> dict[str, Any]:
    """Return launchd status for the conventional research LaunchAgent label."""
    status = dict(launch_agent_status(f"com.trademl.research.{program_id}"))
    status.pop("stdout", None)
    status.pop("stderr", None)
    return status


def _refresh_last_canary_status(*, local_state: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Refresh persisted canary status from its supervisor and summary artifacts."""
    canary = dict(state.get("last_canary") or {})
    experiment_id = str(canary.get("experiment_id") or "")
    if not experiment_id:
        return state
    supervisor = read_experiment_supervisor_state(local_state=local_state, experiment_id=experiment_id)
    summary = _read_experiment_summary_direct(local_state=local_state, experiment_id=experiment_id)
    counts = dict(summary.get("counts") or {})
    status = str((supervisor or {}).get("status") or canary.get("status") or "").upper()
    active_count = sum(int(counts.get(key, 0) or 0) for key in ("RUNNING", "STARTING", "PLANNED"))
    terminal_count = sum(int(counts.get(key, 0) or 0) for key in ("COMPLETED", "FAILED", "STOPPED", "PERMANENT_FAILED"))
    if active_count == 0 and terminal_count > 0 and status in {"", "RUNNING", "STARTING", "STOPPING"}:
        status = "COMPLETED" if int(counts.get("COMPLETED", 0) or 0) == terminal_count else "COMPLETED_WITH_FAILURES"
    if not status:
        return state
    refreshed = {**canary, "status": status}
    if counts:
        refreshed["counts"] = counts
    completed_at = (supervisor or {}).get("completed_at")
    if completed_at:
        refreshed["completed_at"] = completed_at
    if refreshed == canary:
        return state
    updated = deepcopy(state)
    updated["last_canary"] = refreshed
    program_id = str(updated.get("program_id") or "")
    if program_id:
        _write_program_state(local_state=local_state, program_id=program_id, payload=updated)
    return updated


def _research_alert_signature(alerts: list[dict[str, Any]]) -> str:
    """Return a stable signature for the current alert set."""
    if not alerts:
        return ""
    normalized = [
        {
            "kind": alert.get("kind"),
            "severity": alert.get("severity"),
            "message": alert.get("message"),
            "value": alert.get("value"),
        }
        for alert in alerts
    ]
    return hashlib.sha1(json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _research_alerts_due(*, state: dict[str, Any], policy: dict[str, Any], alert_signature: str) -> bool:
    """Return whether the current alert set should write/send now."""
    if not alert_signature:
        return True
    merged_policy = {**DEFAULT_ALERT_POLICY, **dict(policy or {})}
    if state.get("latest_alert_signature") != alert_signature:
        return True
    last = state.get("last_alert_at")
    if not last:
        return True
    with contextlib.suppress(ValueError):
        return (datetime.now(tz=UTC) - datetime.fromisoformat(str(last))) >= timedelta(hours=float(merged_policy["cadence_hours"]))
    return True


def pause_research_program(*, local_state: Path, program_id: str) -> dict[str, Any]:
    """Pause one perpetual research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    state["paused"] = True
    state["status"] = "PAUSED"
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return state


def stop_research_program(
    *,
    local_state: Path,
    program_id: str,
    repo_root: Path | None = None,
    data_root: Path | None = None,
    targets_config_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Stop one perpetual research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    state["stop_requested"] = True
    state["status"] = "STOPPING"
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    pid = state.get("pid")
    if isinstance(pid, int):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    current_experiment_id = state.get("current_experiment_id")
    if current_experiment_id:
        with contextlib.suppress(Exception):
            stop_experiment_supervisor(
                local_state=local_state,
                experiment_id=str(current_experiment_id),
                repo_root=repo_root,
                data_root=data_root,
                targets_config_path=targets_config_path,
                python_executable=python_executable,
            )
    return state


def resume_research_program(
    *,
    program_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Resume a paused research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    state["paused"] = False
    state["stop_requested"] = False
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    pid = state.get("pid")
    if isinstance(pid, int) and _is_local_process_running(pid):
        state["status"] = "RUNNING"
        _write_program_state(local_state=local_state, program_id=program_id, payload=state)
        return state
    return start_research_program(
        program_path=Path(str(state["spec_path"])),
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=int(state.get("poll_seconds") or 30),
        detach=True,
    )


def steer_research_program(
    *,
    local_state: Path,
    program_id: str,
    prefer_architecture_families: list[str] | None = None,
    avoid_architecture_families: list[str] | None = None,
    prefer_data_families: list[str] | None = None,
    avoid_data_families: list[str] | None = None,
    freeze_phase: int | None = None,
    force_pivot: bool | None = None,
    exploration_breadth: str | None = None,
) -> dict[str, Any]:
    """Persist manual steering hints for the research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    steering = {**DEFAULT_STEERING, **dict(state.get("steering") or {})}
    if prefer_architecture_families is not None:
        steering["prefer_architecture_families"] = [value for value in prefer_architecture_families if value]
    if avoid_architecture_families is not None:
        steering["avoid_architecture_families"] = [value for value in avoid_architecture_families if value]
    if prefer_data_families is not None:
        steering["prefer_data_families"] = [value for value in prefer_data_families if value]
    if avoid_data_families is not None:
        steering["avoid_data_families"] = [value for value in avoid_data_families if value]
    if freeze_phase is not None:
        steering["freeze_phase"] = freeze_phase
    if force_pivot is not None:
        steering["force_pivot"] = bool(force_pivot)
    if exploration_breadth is not None:
        steering["exploration_breadth"] = str(exploration_breadth)
    state["steering"] = steering
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return state


def write_research_review_packet(
    *,
    program_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Write a review packet for one research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    state = _refresh_last_canary_status(local_state=local_state, state=state)
    current_experiment_id = str(state.get("current_experiment_id") or "")
    experiment_summary = _read_experiment_summary_direct(local_state=local_state, experiment_id=current_experiment_id) if current_experiment_id else {}
    completed_runs = int(((experiment_summary.get("counts") or {}).get("COMPLETED", 0)) or 0)
    comparison = {"rows": [], "best": None}
    if current_experiment_id and completed_runs > 0:
        comparison = compare_experiment(
            experiment_id=current_experiment_id,
            local_state=local_state,
            repo_root=repo_root,
            data_root=data_root,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
        )
    spec = {}
    if state.get("spec_path"):
        with contextlib.suppress(Exception):
            spec = _load_research_program_spec(Path(str(state["spec_path"])))
    incumbent_update = update_research_incumbent(
        program_id=program_id,
        local_state=local_state,
        data_root=data_root,
        candidates=list(comparison.get("rows") or []),
        policy=dict(spec.get("incumbent_policy") or {}),
    )
    incumbent = incumbent_update.get("incumbent") or read_research_incumbent(local_state=local_state, program_id=program_id)
    paper_outputs = {}
    if incumbent:
        paper_outputs = write_paper_outputs_for_incumbent(
            incumbent=incumbent,
            data_root=data_root,
            local_state=local_state,
            policy=dict(spec.get("paper_policy") or {}),
        )
    shadow_paper_outputs = {}
    if not incumbent:
        shadow_candidate = _best_candidate_with_predictions(list(comparison.get("rows") or []))
        if shadow_candidate:
            shadow_paper_outputs = write_shadow_paper_outputs_for_candidate(
                candidate=shadow_candidate,
                data_root=data_root,
                local_state=local_state,
                policy=dict(spec.get("paper_policy") or {}),
            )
    paper_pnl = {}
    if paper_outputs.get("status") == "written":
        paper_pnl = evaluate_paper_pnl(
            paper_outputs=paper_outputs,
            data_root=data_root,
            policy=dict(spec.get("paper_policy") or {}),
        )
    elif shadow_paper_outputs.get("status") == "written":
        paper_pnl = evaluate_paper_pnl(
            paper_outputs=shadow_paper_outputs,
            data_root=data_root,
            policy=dict(spec.get("paper_policy") or {}),
        )
    drift_alerts = evaluate_research_drift(
        program_id=program_id,
        state=state,
        incumbent=incumbent,
        latest_summary=experiment_summary,
        policy=dict(spec.get("drift_policy") or {}),
    )
    alert_policy = dict(spec.get("alert_policy") or {})
    alert_signature = _research_alert_signature(drift_alerts)
    if _research_alerts_due(state=state, policy=alert_policy, alert_signature=alert_signature):
        alert_result = write_research_alerts(
            program_id=program_id,
            alerts=drift_alerts,
            local_state=local_state,
            data_root=data_root,
            policy=alert_policy,
        )
    else:
        alert_result = {
            "status": "skipped",
            "reason": "alert cadence not due for unchanged alert set",
            "alert_signature": alert_signature,
        }
    payload = {
        "program_id": program_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": state.get("status"),
        "current_phase": state.get("current_phase"),
        "current_ssot_phase": state.get("current_ssot_phase", state.get("current_phase")),
        "current_campaign_track": state.get("current_campaign_track", state.get("current_track")),
        "current_experiment_id": current_experiment_id,
        "best_candidate_summary": state.get("best_candidate_summary", {}),
        "architecture_lane": state.get("architecture_lane"),
        "complexity_tier": state.get("complexity_tier"),
        "objective_verdict": state.get("objective_verdict", {}),
        "pivot_reason": (state.get("last_transition") or {}).get("reason"),
        "next_lane": state.get("next_lane") or (state.get("frontier") or {}).get("next_lane"),
        "sentinel_delta": state.get("sentinel_delta") or (state.get("frontier") or {}).get("sentinel_delta"),
        "autonomous_progression": state.get("autonomous_progression", {}),
        "frontier": state.get("frontier", {}),
        "budgets": state.get("budgets", {}),
        "last_transition": state.get("last_transition", {}),
        "steering": state.get("steering", {}),
        "experiment_summary": experiment_summary,
        "comparison_best": comparison.get("best"),
        "incumbent": incumbent,
        "incumbent_update": incumbent_update,
        "best_rejected_candidate": _best_rejected_candidate(comparison.get("rows") or [], incumbent_update.get("rejections") or []),
        "paper_outputs": paper_outputs,
        "shadow_paper_outputs": shadow_paper_outputs,
        "paper_pnl": paper_pnl,
        "drift_alerts": drift_alerts,
        "alert_result": alert_result,
        "infra_blocker": state.get("last_infra_preflight", {}),
        "launchd": _research_launchd_status(program_id),
        "frontier_architecture": _frontier_architecture_status(
            spec=spec,
            state=state,
            frontier=dict(state.get("frontier") or {}),
            experiment_summary=experiment_summary,
        ),
        "dependency_preflight": (state.get("last_infra_preflight") or {}).get("dependencies", {}),
        "paths": _research_path_summary(local_state=local_state, data_root=data_root),
        "top_rejections": experiment_summary.get("top_gate_failures", []),
        "next_direction": state.get("last_transition", {}).get("reason"),
    }
    root = _program_root(local_state=local_state, program_id=program_id) / "review_packets"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"{stamp}.json"
    md_path = root / f"{stamp}.md"
    _atomic_write_json(json_path, payload)
    md_lines = [
        f"# Research review packet: {program_id}",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- status: {payload['status']}",
        f"- current_phase: {payload['current_phase']}",
        f"- current_ssot_phase: {payload['current_ssot_phase']}",
        f"- current_campaign_track: {payload['current_campaign_track'] or '-'}",
        f"- current_experiment_id: {payload['current_experiment_id'] or '-'}",
        f"- architecture_lane: {payload['architecture_lane'] or '-'}",
        f"- complexity_tier: {payload['complexity_tier']}",
        f"- next_lane: {payload['next_lane'] or '-'}",
        f"- sentinel_delta: {payload['sentinel_delta']}",
        f"- best_candidate: {(payload['best_candidate_summary'] or {}).get('best_candidate') or '-'}",
        f"- best_primary_score: {(payload['best_candidate_summary'] or {}).get('best_primary_score')}",
        f"- incumbent_run_id: {(payload['incumbent'] or {}).get('run_id') or '-'}",
        f"- paper_outputs: {(payload['paper_outputs'] or {}).get('paper_orders_path') or '-'}",
        f"- shadow_paper_outputs: {(payload['shadow_paper_outputs'] or {}).get('shadow_orders_path') or '-'}",
        f"- paper_pnl: {(payload['paper_pnl'] or {}).get('status') or '-'}",
        f"- alert_count: {len(payload['drift_alerts'])}",
        f"- last_transition: {(payload['last_transition'] or {}).get('reason') or '-'}",
        "",
        "## Top rejections",
        *[f"- {reason}: {count}" for reason, count in payload["top_rejections"]],
        "",
        "## Best comparison row",
        f"- {json.dumps(payload['comparison_best'], default=str)}",
        "",
        "## Best rejected candidate",
        f"- {json.dumps(payload['best_rejected_candidate'], default=str)}",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    result = {"json_path": str(json_path), "markdown_path": str(md_path)}
    state["latest_review_packet"] = result
    state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()
    state["incumbent"] = incumbent
    state["latest_paper_outputs"] = paper_outputs
    state["latest_shadow_paper_outputs"] = shadow_paper_outputs
    state["latest_paper_pnl"] = paper_pnl
    state["latest_drift_alerts"] = drift_alerts
    if alert_result.get("status") == "written":
        state["last_alert_at"] = datetime.now(tz=UTC).isoformat()
        state["latest_alert_signature"] = alert_signature
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return result


def _incumbent_rejection_reasons(*, candidate: dict[str, Any], incumbent: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    decision = candidate.get("decision") or (candidate.get("assessment") or {}).get("decision")
    primary_score = _safe_float(candidate.get("primary_score") or candidate.get("rank_ic"))
    net_return = _safe_float(candidate.get("backtest_net_return") or candidate.get("net_return"))
    pbo = _safe_float(candidate.get("pbo"))
    if not bool(candidate.get("shortlisted")):
        reasons.append("candidate was not shortlisted")
    if not bool(candidate.get("survived_predictive", candidate.get("shortlisted", False))):
        reasons.append("predictive gates did not pass")
    if str(decision) != "GO":
        reasons.append("assessment.decision != GO")
    if not bool(candidate.get("years_positive")):
        reasons.append("not all years positive")
    if str(candidate.get("backtest_status") or "COMPLETED").upper() != "COMPLETED":
        reasons.append("backtest did not complete")
    if net_return is None or net_return < float(policy["min_net_return"]):
        reasons.append("cost-positive backtest gate failed")
    if pbo is None or pbo > float(policy["max_pbo"]):
        reasons.append(f"pbo>{policy['max_pbo']}")
    if primary_score is None:
        reasons.append("missing rank IC")
    if incumbent and primary_score is not None and net_return is not None:
        incumbent_score = _safe_float(incumbent.get("primary_score") or incumbent.get("rank_ic")) or 0.0
        incumbent_return = _safe_float(incumbent.get("backtest_net_return") or incumbent.get("net_return")) or 0.0
        score_improved = primary_score >= incumbent_score + float(policy["min_rank_ic_improvement"])
        return_improved = net_return >= incumbent_return + float(policy["min_net_return_improvement"])
        if not (score_improved or return_improved):
            reasons.append("does not improve incumbent")
        candidate_adjusted = complexity_adjusted_score(candidate, policy=policy)
        incumbent_adjusted = complexity_adjusted_score(incumbent, policy=policy)
        min_adjusted = float((policy.get("complexity_penalty") or {}).get("min_complexity_adjusted_improvement") or 0.0)
        candidate_tier = int(candidate.get("complexity_tier") or 0)
        incumbent_tier = int(incumbent.get("complexity_tier") or 0)
        if (
            bool((policy.get("complexity_penalty") or {}).get("enabled", False))
            and candidate_tier > incumbent_tier
            and candidate_adjusted is not None
            and incumbent_adjusted is not None
            and candidate_adjusted <= incumbent_adjusted + min_adjusted
        ):
            reasons.append("does not beat incumbent after complexity penalty")
    return reasons


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_candidate_with_predictions(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the strongest candidate that exposes a primary predictions artifact."""
    eligible = []
    for candidate in candidates:
        artifacts = dict(candidate.get("artifacts") or {})
        if artifacts.get("primary_predictions_path") or candidate.get("primary_predictions_path"):
            eligible.append(candidate)
    if not eligible:
        return {}
    return sorted(
        eligible,
        key=lambda item: (
            float(item.get("primary_score") or item.get("rank_ic") or 0.0),
            float(item.get("backtest_net_return") or item.get("net_return") or 0.0),
        ),
        reverse=True,
    )[0]


def _incumbent_path(*, local_state: Path, program_id: str) -> Path:
    return _local_research_root(local_state=local_state) / "incumbents" / f"{program_id}.json"


def _write_incumbent_payload(*, local_state: Path, data_root: Path, program_id: str, payload: dict[str, Any]) -> None:
    local_path = _incumbent_path(local_state=local_state, program_id=program_id)
    _atomic_write_json(local_path, payload)
    shared_path = _shared_research_root(data_root=data_root) / "incumbents" / f"{program_id}.json"
    _atomic_write_json(shared_path, payload)


def _local_research_root(*, local_state: Path) -> Path:
    return local_state / "research"


def _shared_research_root(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research"


def _latest_prior_paper_targets(*, data_root: Path, local_state: Path, current_date: str) -> pd.DataFrame:
    roots = [
        _shared_research_root(data_root=data_root) / "paper",
        _local_research_root(local_state=local_state) / "paper",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/target_weights.parquet"):
            if path.parent.name < current_date:
                candidates.append(path)
    if not candidates:
        return pd.DataFrame(columns=["date", "symbol", "score", "target_weight"])
    latest = max(candidates, key=lambda path: path.parent.name)
    return pd.read_parquet(latest)


def _paper_order_deltas(*, targets: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current = targets[["symbol", "target_weight"]].copy() if not targets.empty else pd.DataFrame(columns=["symbol", "target_weight"])
    current = current.rename(columns={"target_weight": "target_weight"})
    previous_frame = previous[["symbol", "target_weight"]].copy() if not previous.empty else pd.DataFrame(columns=["symbol", "target_weight"])
    previous_frame = previous_frame.rename(columns={"target_weight": "previous_weight"})
    merged = current.merge(previous_frame, on="symbol", how="outer")
    merged["target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)
    merged["previous_weight"] = pd.to_numeric(merged["previous_weight"], errors="coerce").fillna(0.0)
    merged["order_delta"] = merged["target_weight"] - merged["previous_weight"]
    merged["action"] = "HOLD"
    merged.loc[merged["order_delta"] > 0, "action"] = "BUY"
    merged.loc[merged["order_delta"] < 0, "action"] = "SELL"
    if not targets.empty and "date" in targets.columns:
        merged["date"] = pd.to_datetime(targets["date"].max())
    return merged.sort_values(["action", "symbol"]).reset_index(drop=True)


def _load_curated_prices_for_pnl(*, data_root: Path, symbols: list[str], start_date: pd.Timestamp) -> pd.DataFrame:
    curated_root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    if not curated_root.exists():
        return pd.DataFrame(columns=["date", "symbol", "close"])
    frames: list[pd.DataFrame] = []
    symbol_set = set(symbols)
    for path in sorted(curated_root.glob("date=*/data.parquet")):
        with contextlib.suppress(ValueError, IndexError):
            partition_date = pd.Timestamp(path.parent.name.split("=", 1)[1]).normalize()
            if partition_date < start_date:
                continue
        try:
            frame = pd.read_parquet(path, columns=["date", "symbol", "close"])
        except Exception:
            continue
        if frame.empty:
            continue
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.loc[frame["symbol"].isin(symbol_set)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return prices.dropna(subset=["date", "symbol", "close"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def _symbol_returns_from_prices(*, prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    start = frame.loc[frame["date"] == start_date, ["symbol", "close"]].drop_duplicates("symbol", keep="last")
    end = frame.loc[frame["date"] == end_date, ["symbol", "close"]].drop_duplicates("symbol", keep="last")
    merged = start.merge(end, on="symbol", suffixes=("_start", "_end"))
    if merged.empty:
        return pd.Series(dtype=float)
    merged["return"] = pd.to_numeric(merged["close_end"], errors="coerce") / pd.to_numeric(merged["close_start"], errors="coerce") - 1.0
    return merged.set_index("symbol")["return"].dropna()


def _latest_summary_coverage(summary: dict[str, Any]) -> float | None:
    coverages: list[float] = []
    for run in list(summary.get("runs") or []):
        preview = dict(run.get("report_preview") or {})
        value = preview.get("coverage")
        if value is not None:
            numeric = _safe_float(value)
            if numeric is not None:
                coverages.append(numeric)
    return min(coverages) if coverages else None


def _research_path_summary(*, local_state: Path, data_root: Path) -> dict[str, str]:
    return {
        "local_research_root": str(_local_research_root(local_state=local_state)),
        "shared_research_root": str(_shared_research_root(data_root=data_root)),
        "local_state": str(local_state),
        "data_root": str(data_root),
    }


def _best_rejected_candidate(rows: list[dict[str, Any]], rejections: list[dict[str, Any]]) -> dict[str, Any] | None:
    rejected_ids = {item.get("run_id"): item.get("reasons", []) for item in rejections}
    for row in sorted(rows, key=lambda item: float(item.get("primary_score") or 0.0), reverse=True):
        if row.get("run_id") in rejected_ids:
            return {**row, "rejection_reasons": rejected_ids[row.get("run_id")]}
    return None


def _load_research_program_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("ssot_phase_order") and not payload.get("phase_order"):
        payload["phase_order"] = list(payload["ssot_phase_order"])
    if payload.get("phase_order") and not payload.get("ssot_phase_order"):
        payload["ssot_phase_order"] = list(payload["phase_order"])
    if payload.get("default_ssot_phase") is not None and payload.get("default_phase") is None:
        payload["default_phase"] = payload["default_ssot_phase"]
    if payload.get("default_phase") is not None and payload.get("default_ssot_phase") is None:
        payload["default_ssot_phase"] = payload["default_phase"]
    if payload.get("ssot_phase_policies") and not payload.get("phase_policies"):
        payload["phase_policies"] = deepcopy(payload["ssot_phase_policies"])
    if payload.get("phase_policies") and not payload.get("ssot_phase_policies"):
        payload["ssot_phase_policies"] = deepcopy(payload["phase_policies"])
    raw_policies = payload.get("phase_policies") or {}
    if isinstance(raw_policies, dict):
        normalized: dict[str, Any] = {}
        for key, item in raw_policies.items():
            if isinstance(item, dict):
                phase_value = item.get("ssot_phase", item.get("phase", key))
                normalized_item = dict(item)
                normalized_item["ssot_phase"] = int(phase_value)
                normalized_item["phase"] = int(normalized_item.get("phase", phase_value))
                normalized_item.setdefault("campaign_track", "baseline")
                normalized[str(phase_value)] = normalized_item
            else:
                normalized[str(key)] = item
        raw_policies = normalized
    payload["phase_policies"] = raw_policies
    payload["ssot_phase_policies"] = deepcopy(raw_policies)
    if not payload.get("program_id"):
        raise ValueError("research program spec requires program_id")
    if not payload.get("phase_order"):
        raise ValueError("research program spec requires phase_order")
    payload.setdefault("default_campaign_track", "baseline")
    return payload


def _initial_program_state(*, spec: dict[str, Any], program_path: Path, poll_seconds: int) -> dict[str, Any]:
    default_phase = int(spec.get("default_phase") or spec["phase_order"][0])
    return {
        "program_id": str(spec["program_id"]),
        "spec_path": str(program_path),
        "status": "RUNNING",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "heartbeat_at": datetime.now(tz=UTC).isoformat(),
        "completed_at": None,
        "pid": os.getpid(),
        "poll_seconds": poll_seconds,
        "paused": False,
        "stop_requested": False,
        "stop_reason": None,
        "current_phase": default_phase,
        "current_ssot_phase": default_phase,
        "current_track": str(spec.get("default_campaign_track") or "baseline"),
        "current_campaign_track": str(spec.get("default_campaign_track") or "baseline"),
        "current_generation": 0,
        "search_epoch": 0,
        "current_experiment_id": None,
        "current_family_signature": None,
        "active_experiment_supervisor": {},
        "frontier": _empty_frontier(),
        "architecture_registry_policy": _architecture_registry_policy(spec),
        "objective_policy": _objective_policy(spec),
        "autonomous_progression": _autonomous_progression_policy(spec),
        "architecture_registry": architecture_registry_payload(),
        "objective_registry": objective_registry_payload(),
        "frontier_architecture": _frontier_architecture_status(
            spec=spec,
            state={"budgets": {"runs_completed": 0}},
            frontier=_empty_frontier(),
            experiment_summary={},
        ),
        "budgets": {
            "families_started": 0,
            "runs_completed": 0,
            "max_total_runs": int(_budget_policy(spec)["max_total_runs"]),
            "max_total_hours": int(_budget_policy(spec)["max_total_hours"]),
            "phase_family_counts": {},
            "phase_run_counts": {},
            "non_improving_families": 0,
            "low_novelty_families": 0,
            "infra_failures": 0,
        },
        "last_transition": {},
        "latest_review_packet": {},
        "last_review_packet_at": None,
        "best_candidate_summary": {},
        "steering": {**DEFAULT_STEERING, **dict(spec.get("steering_defaults") or {})},
        "pending_next_spec": None,
        "last_seen_data_revision": None,
        "wait_reason": None,
        "waiting_since": None,
    }


def _empty_frontier() -> dict[str, Any]:
    return {
        "tried_family_signatures": [],
        "tried_run_signatures": [],
        "lane_stats": {
            "architecture_family": {},
            "feature_family": {},
            "data_family": {},
            "phase": {},
        },
        "best_by_phase": {},
        "best_by_architecture_family": {},
        "best_by_feature_family": {},
        "best_by_data_family": {},
        "exhausted_lanes": {},
        "next_lane": None,
        "sentinel_delta": None,
        "recent_rejection_reasons": [],
        "family_history": [],
        "processed_experiment_ids": [],
    }


def _bootstrap_research_state_from_training_history(
    *, local_state: Path, state: dict[str, Any]
) -> dict[str, Any]:
    """Seed a fresh research state from legacy training-run manifests."""
    budgets = dict(state.get("budgets") or {})
    frontier = dict(state.get("frontier") or _empty_frontier())
    if int(budgets.get("runs_completed", 0) or 0) > 0 or frontier.get("family_history"):
        return state
    completed = [
        manifest
        for manifest in _legacy_training_history_manifests(local_state=local_state)
        if str(manifest.get("status") or "").lower()
        in {"completed", "succeeded", "success"}
    ]
    if not completed:
        return state
    phase_run_counts: dict[str, int] = {}
    experiment_ids: set[str] = set()
    max_family = 0
    for manifest in completed:
        phase_key = str(int(manifest.get("phase") or 1))
        phase_run_counts[phase_key] = phase_run_counts.get(phase_key, 0) + 1
        experiment_id = _legacy_training_experiment_id(manifest)
        if experiment_id:
            experiment_ids.add(experiment_id)
            max_family = max(max_family, _family_sequence(experiment_id) or 0)
    budgets["runs_completed"] = len(completed)
    budgets["phase_run_counts"] = {
        **dict(budgets.get("phase_run_counts") or {}),
        **phase_run_counts,
    }
    budgets["families_started"] = max(
        int(budgets.get("families_started", 0) or 0),
        max_family,
        len(experiment_ids),
    )
    state["budgets"] = budgets
    state["frontier"] = frontier
    state["last_transition"] = {
        "action": "bootstrap_training_history",
        "reason": "seeded fresh research supervisor from existing training run manifests",
        "runs_completed": len(completed),
        "families_started": budgets["families_started"],
    }
    return state


def _legacy_training_history_manifests(*, local_state: Path) -> list[dict[str, Any]]:
    roots = [local_state / "training_runs", local_state / "control" / "training_runs"]
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        paths.update(path for path in root.glob("*.json") if path.is_file())
        paths.update(path for path in root.glob("*/*.json") if path.is_file())
    manifests: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload.setdefault("_manifest_path", str(path))
        manifests.append(payload)
    return manifests


def _legacy_training_experiment_id(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("experiment_id")
    if value:
        return str(value)
    path_value = manifest.get("_manifest_path")
    if path_value:
        name = Path(str(path_value)).name
        if name.startswith("experiment_") and "__" in name:
            return name.removeprefix("experiment_").split("__", 1)[0]
    report_path = manifest.get("report_path")
    if report_path:
        parts = Path(str(report_path)).parts
        if "experiments" in parts:
            index = parts.index("experiments")
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def _build_initial_phase_experiment_spec(*, program_state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    phase_policy = _phase_policy(spec=spec, phase=_current_ssot_phase(state=program_state, spec=spec))
    if phase_policy is None:
        return None
    frontier_spec = _frontier_architecture_spec(
        spec=spec,
        state=program_state,
        phase_policy=phase_policy,
        frontier=dict(program_state.get("frontier") or _empty_frontier()),
        experiment_summary={},
    )
    if frontier_spec is not None:
        return frontier_spec
    return _make_experiment_spec_from_phase_policy(
        spec=spec,
        program_state=program_state,
        phase_policy=phase_policy,
        experiment_id=_next_experiment_id(
            program_state=program_state,
            spec=spec,
            phase=_current_ssot_phase(state=program_state, spec=spec),
        ),
        generation=int(program_state.get("current_generation", 0)),
    )


def _phase_policy(*, spec: dict[str, Any], phase: int) -> dict[str, Any] | None:
    raw = spec.get("phase_policies") or {}
    if isinstance(raw, list):
        for item in raw:
            item_phase = item.get("ssot_phase", item.get("phase", 0))
            if int(item_phase or 0) == phase:
                data = dict(item)
                data.setdefault("ssot_phase", phase)
                data.setdefault("phase", phase)
                data.setdefault("campaign_track", "baseline")
                return data
        return None
    payload = raw.get(str(phase)) if isinstance(raw, dict) else None
    if payload is None and isinstance(raw, dict):
        payload = raw.get(phase)
    if payload is None:
        return None
    data = dict(payload)
    data.setdefault("phase", phase)
    data.setdefault("ssot_phase", phase)
    data.setdefault("campaign_track", "baseline")
    return data


def _make_experiment_spec_from_phase_policy(
    *,
    spec: dict[str, Any],
    program_state: dict[str, Any],
    phase_policy: dict[str, Any],
    experiment_id: str,
    generation: int,
) -> dict[str, Any]:
    allowed_data_families = [
        family
        for family in list(phase_policy.get("allowed_data_families") or ["price_only", "price_plus_liquidity"])
        if family in MODELING_READY_DATA_FAMILIES
    ]
    initial_matrix = deepcopy(phase_policy.get("initial_matrix") or {})
    if allowed_data_families and "data_family" not in initial_matrix:
        initial_matrix["data_family"] = allowed_data_families[: max(1, min(len(allowed_data_families), 2))]
    matrix = _apply_program_steering_to_matrix(matrix=initial_matrix, steering=dict(program_state.get("steering") or {}))
    family_size_cap = int(phase_policy.get("family_size_cap") or 6)
    breadth = _exploration_multiplier(str((program_state.get("steering") or {}).get("exploration_breadth") or "normal"))
    family_size_cap = max(1, int(round(family_size_cap * breadth)))
    next_spec = {
        "experiment_id": experiment_id,
        "family_root": str(spec["program_id"]),
        "generation": generation,
        "search_epoch": int(program_state.get("search_epoch", 0) or 0),
        "phase": int(phase_policy["phase"]),
        "ssot_phase": int(phase_policy.get("ssot_phase", phase_policy["phase"])),
        "campaign_track": str(
            program_state.get("current_campaign_track")
            or program_state.get("current_track")
            or phase_policy.get("campaign_track")
            or spec.get("default_campaign_track")
            or "baseline"
        ),
        "target": str(spec.get("target") or "local"),
        "modeling": deepcopy(spec.get("modeling") or {}),
        "report_date_policy": str(phase_policy.get("report_date_policy") or spec.get("report_date_policy") or "phase1_freeze"),
        "max_concurrent": int(phase_policy.get("max_concurrent") or spec.get("max_concurrent") or 1),
        "supervision": {
            **dict(spec.get("supervision") or {}),
            "auto_backtest_survivors": True,
            "auto_propose_next_family": False,
        },
        "matrix": matrix,
        "predictive_gate": deepcopy(phase_policy.get("predictive_gate") or spec.get("predictive_gate") or {}),
        "backtest_gate": deepcopy(phase_policy.get("backtest_gate") or spec.get("backtest_gate") or {}),
        "backtest_profile": deepcopy(phase_policy.get("backtest_profile") or spec.get("backtest_profile") or {}),
        "proposal_policy": {
            "family_size_cap": family_size_cap,
            "allowed_dimensions": list(phase_policy.get("allowed_dimensions") or spec.get("search_policy", {}).get("allowed_dimensions") or []),
            "max_generations": int(phase_policy.get("max_generations") or 1),
            "auto_launch_next_family": False,
        },
    }
    return _decorate_autonomous_experiment_spec(next_spec=next_spec, spec=spec)


def _determine_program_transition(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    frontier: dict[str, Any],
    experiment_summary: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    current_phase = _current_ssot_phase(state=state, spec=spec)
    phase_policy = _phase_policy(spec=spec, phase=current_phase)
    if phase_policy is None:
        return {"action": "stop", "reason": f"missing phase policy for phase {current_phase}"}

    next_phase = _next_phase(spec=spec, phase=current_phase)
    if _phase_unlock_allowed(spec=spec, state=state, phase_policy=phase_policy, experiment_summary=experiment_summary) and next_phase is not None:
        next_policy = _phase_policy(spec=spec, phase=next_phase)
        if next_policy is not None:
            next_spec = _make_experiment_spec_from_phase_policy(
                spec=spec,
                program_state={**state, "current_phase": next_phase, "current_ssot_phase": next_phase, "current_generation": 0},
                phase_policy=next_policy,
                experiment_id=_next_experiment_id(
                    program_state={**state, "current_phase": next_phase, "current_ssot_phase": next_phase, "current_generation": 0},
                    spec=spec,
                    phase=next_phase,
                ),
                generation=0,
            )
            signature = _family_signature(next_spec)
            novelty = _novelty_score(frontier=frontier, next_spec=next_spec)
            return {
                "action": "advance_phase",
                "reason": f"phase {current_phase} produced a shortlist-worthy candidate; advancing to phase {next_phase}",
                "next_phase": next_phase,
                "next_spec": next_spec,
                "family_signature": signature,
                "novelty_score": novelty,
            }

    frontier_spec = _frontier_architecture_spec(spec=spec, state=state, phase_policy=phase_policy, frontier=frontier, experiment_summary=experiment_summary)
    if frontier_spec is not None:
        return {
            "action": "launch_family",
            "reason": "frontier architecture policy is testing advanced challengers before more baseline churn",
            "next_spec": frontier_spec,
            "family_signature": _family_signature(frontier_spec),
            "novelty_score": _novelty_score(frontier=frontier, next_spec=frontier_spec),
            "frontier_architecture": _frontier_architecture_status(spec=spec, state=state, frontier=frontier, experiment_summary=experiment_summary),
        }

    pivot_spec = _frontier_architecture_brake_pivot_spec(
        spec=spec,
        state=state,
        phase_policy=phase_policy,
        frontier=frontier,
        experiment_summary=experiment_summary,
    )
    if pivot_spec is not None:
        return {
            "action": "launch_family",
            "reason": f"frontier architecture brake reached; pivoting to search epoch {pivot_spec['search_epoch']}",
            "next_spec": pivot_spec,
            "family_signature": _family_signature(pivot_spec),
            "novelty_score": _novelty_score(frontier=frontier, next_spec=pivot_spec),
            "frontier_architecture": _frontier_architecture_status(
                spec=spec,
                state={**state, "search_epoch": pivot_spec["search_epoch"]},
                frontier=frontier,
                experiment_summary=experiment_summary,
            ),
        }

    stop_reason = _program_stop_reason(spec=spec, state=state, phase_policy=phase_policy, frontier=frontier, experiment_summary=experiment_summary)
    if stop_reason:
        if _stop_reason_waits_for_data(spec=spec, stop_reason=stop_reason):
            return {"action": "wait_for_data", "reason": f"{stop_reason}; waiting for a fresh search frontier or environment recovery"}
        return {"action": "stop", "reason": stop_reason}

    next_spec = _program_next_spec_from_proposal(spec=spec, state=state, phase_policy=phase_policy, proposal=proposal, frontier=frontier)
    if next_spec is None:
        expanded_spec = _expanded_search_spec(spec=spec, state=state, phase_policy=phase_policy, frontier=frontier)
        if expanded_spec is not None:
            return {
                "action": "launch_family",
                "reason": "proposal space exhausted, broadening the bounded search frontier",
                "next_spec": expanded_spec,
                "family_signature": _family_signature(expanded_spec),
                "novelty_score": _novelty_score(frontier=frontier, next_spec=expanded_spec),
            }
        if _exhaustion_mode(spec) == "wait_for_new_data":
            return {"action": "wait_for_data", "reason": "no valid next-family proposal remained; waiting for new data revision"}
        return {"action": "stop", "reason": "no valid next-family proposal remained after novelty and budget checks"}
    signature = _family_signature(next_spec)
    novelty = _novelty_score(frontier=frontier, next_spec=next_spec)
    if signature in set(frontier.get("tried_family_signatures") or []) and novelty < float(_budget_policy(spec)["min_novelty_score"]):
        expanded_spec = _expanded_search_spec(spec=spec, state=state, phase_policy=phase_policy, frontier=frontier)
        if expanded_spec is not None:
            return {
                "action": "launch_family",
                "reason": "next-family proposal was low-novelty, broadening the bounded search frontier",
                "next_spec": expanded_spec,
                "family_signature": _family_signature(expanded_spec),
                "novelty_score": _novelty_score(frontier=frontier, next_spec=expanded_spec),
            }
        if _exhaustion_mode(spec) == "wait_for_new_data":
            return {"action": "wait_for_data", "reason": "next-family proposal was a duplicate with insufficient novelty; waiting for new data revision"}
        return {"action": "stop", "reason": "next-family proposal was a duplicate with insufficient novelty"}
    return {
        "action": "launch_family",
        "reason": proposal.get("rationale", ["continuing bounded search"])[0] if isinstance(proposal.get("rationale"), list) and proposal.get("rationale") else "continuing bounded search",
        "next_spec": next_spec,
        "family_signature": signature,
        "novelty_score": novelty,
    }


def _program_next_spec_from_proposal(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    proposal: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any] | None:
    next_spec = deepcopy(proposal.get("next_spec") or {})
    if not next_spec:
        return None
    next_spec["target"] = str(spec.get("target") or next_spec.get("target") or "local")
    next_spec["supervision"] = {
        **dict(next_spec.get("supervision") or {}),
        "auto_backtest_survivors": True,
        "auto_propose_next_family": False,
    }
    next_spec["proposal_policy"] = {
        **dict(next_spec.get("proposal_policy") or {}),
        "auto_launch_next_family": False,
    }
    next_spec["phase"] = _current_ssot_phase(state=state, spec=spec)
    next_spec["ssot_phase"] = int(next_spec["phase"])
    next_spec["campaign_track"] = str(
        next_spec.get("campaign_track")
        or state.get("current_campaign_track")
        or state.get("current_track")
        or phase_policy.get("campaign_track")
        or spec.get("default_campaign_track")
        or "baseline"
    )
    next_spec["experiment_id"] = _next_experiment_id(program_state=state, spec=spec, phase=int(next_spec["phase"]))
    next_spec["generation"] = int(state.get("current_generation", 0) or 0) + 1
    next_spec["search_epoch"] = int(state.get("search_epoch", 0) or 0)
    next_spec["family_root"] = str(spec["program_id"])
    if "data_family" not in dict(next_spec.get("matrix") or {}):
        allowed = [
            family
            for family in list(phase_policy.get("allowed_data_families") or [])
            if family in MODELING_READY_DATA_FAMILIES
        ]
        if allowed:
            next_spec.setdefault("matrix", {})["data_family"] = allowed[:1]
    next_spec["matrix"] = _apply_program_steering_to_matrix(
        matrix=dict(next_spec.get("matrix") or {}),
        steering=dict(state.get("steering") or {}),
    )
    return next_spec


def _frontier_architecture_policy(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the controlled advanced-first frontier policy."""
    return {**DEFAULT_FRONTIER_ARCHITECTURE_POLICY, **dict(spec.get("frontier_architecture_policy") or {})}


def _objective_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policy = deepcopy(DEFAULT_OBJECTIVE_POLICY)
    incoming = dict(spec.get("objective_policy") or {})
    penalty = {**dict(policy.get("complexity_penalty") or {}), **dict(incoming.get("complexity_penalty") or {})}
    policy.update(incoming)
    policy["complexity_penalty"] = penalty
    return policy


def _architecture_registry_policy(spec: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_ARCHITECTURE_REGISTRY_POLICY, **dict(spec.get("architecture_registry_policy") or {})}


def _autonomous_progression_policy(spec: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_AUTONOMOUS_PROGRESSION_POLICY, **dict(spec.get("autonomous_progression_policy") or {})}


def _decorate_autonomous_experiment_spec(*, next_spec: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    matrix = dict(next_spec.get("matrix") or {})
    architectures = [str(item) for item in matrix.get("architecture_family") or []]
    suites = [str(item) for item in matrix.get("model_suite") or []]
    suite_lane = (
        "ensemble_meta"
        if "ensemble" in suites
        else "advanced_challenger"
        if "advanced" in suites
        else "tree_challenger"
        if "full" in suites
        else "linear_baseline"
    )
    primary_lane = architectures[0] if architectures else suite_lane
    entry = resolve_architecture_entry(primary_lane)
    next_spec["architecture_registry_policy"] = _architecture_registry_policy(spec)
    next_spec["architecture_registry"] = architecture_registry_payload()
    next_spec["objective_registry"] = objective_registry_payload()
    next_spec["objective_policy"] = _objective_policy(spec)
    next_spec["architecture_lane"] = primary_lane
    next_spec["complexity_tier"] = int(entry["complexity_tier"])
    next_spec["autonomous_progression"] = _autonomous_progression_policy(spec)
    next_spec["next_lane"] = primary_lane
    return next_spec


def _frontier_architecture_status(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    frontier: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    policy = _frontier_architecture_policy(spec)
    advanced_count = _frontier_advanced_count_for_epoch(state=state, frontier=frontier)
    return {
        "policy": policy,
        "enabled": bool(policy.get("enabled", False)),
        "runs_completed": int((state.get("budgets") or {}).get("runs_completed", 0) or 0),
        "search_epoch": int(state.get("search_epoch", 0) or 0),
        "trigger_min_completed_runs": int(policy.get("trigger_min_completed_runs") or 0),
        "advanced_failures_this_epoch": advanced_count,
        "max_advanced_failures_per_epoch": int(policy.get("max_advanced_failures_per_epoch") or 0),
        "brake_active": bool(int(policy.get("max_advanced_failures_per_epoch") or 0) > 0 and advanced_count >= int(policy.get("max_advanced_failures_per_epoch") or 0)),
        "shortlist_count": int(experiment_summary.get("shortlist_count", 0) or 0),
    }


def _frontier_architecture_spec(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    frontier: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any] | None:
    policy = _frontier_architecture_policy(spec)
    if not bool(policy.get("enabled", False)):
        return None
    phase = _current_ssot_phase(state=state, spec=spec)
    if phase <= 1 and not bool(policy.get("allow_phase1_advanced", False)):
        return None
    if int((state.get("budgets") or {}).get("runs_completed", 0) or 0) < int(policy.get("trigger_min_completed_runs") or 0):
        return None
    best_summary = dict(state.get("best_candidate_summary") or {})
    if int(experiment_summary.get("shortlist_count", 0) or best_summary.get("shortlist_count", 0) or 0) > 0:
        return None
    advanced_count = _frontier_advanced_count_for_epoch(state=state, frontier=frontier)
    max_advanced = int(policy.get("max_advanced_failures_per_epoch") or 0)
    if max_advanced > 0 and advanced_count >= max_advanced:
        return None
    allowed_data_families = [
        family
        for family in list(phase_policy.get("allowed_data_families") or [])
        if family in MODELING_READY_DATA_FAMILIES
    ]
    if not allowed_data_families:
        allowed_data_families = ["price_only"]
    epoch = int(state.get("search_epoch", 0) or 0)
    best_feature = _frontier_lane_for_epoch(
        frontier=frontier,
        lane="feature_family",
        allowed=["price_liquidity", "price_core", "price_short_horizon"],
        default="price_liquidity",
        epoch=epoch,
    )
    best_data = _frontier_lane_for_epoch(
        frontier=frontier,
        lane="data_family",
        allowed=allowed_data_families,
        default=allowed_data_families[0],
        epoch=epoch,
    )
    progression = _architecture_progression_plan(policy=policy, frontier=frontier, epoch=epoch)
    architectures = progression["architectures"]
    next_spec = _make_experiment_spec_from_phase_policy(
        spec=spec,
        program_state=state,
        phase_policy=phase_policy,
        experiment_id=_next_experiment_id(program_state=state, spec=spec, phase=phase),
        generation=int(state.get("current_generation", 0) or 0) + 1,
    )
    next_spec["matrix"] = _apply_program_steering_to_matrix(
        matrix={
            "architecture_family": architectures,
            "feature_family": [best_feature],
            "data_family": [best_data],
            "label_horizon": [_frontier_label_horizon_for_epoch(spec=spec, epoch=epoch)],
            "portfolio_profile": ["cost_aware_long_only_v1"],
            "data_profile": [_frontier_profile_for_epoch(epoch)],
            "validation.initial_train_years": [_frontier_train_years_for_epoch(epoch)],
        },
        steering=dict(state.get("steering") or {}),
    )
    architecture_values = list(next_spec["matrix"].get("architecture_family") or [])
    current_lane = str(progression.get("current_lane") or "")
    if dict(state.get("steering") or {}).get("force_pivot"):
        next_spec["matrix"]["architecture_family"] = [value for value in architectures if value in architecture_values]
    elif current_lane in architecture_values:
        next_spec["matrix"]["architecture_family"] = [current_lane] + [value for value in architecture_values if value != current_lane]
    next_spec["proposal_policy"]["family_size_cap"] = int(policy.get("family_size_cap") or phase_policy.get("family_size_cap") or 6)
    next_spec["frontier_architecture_policy"] = policy
    next_spec["frontier_architecture"] = True
    next_spec["frontier_iteration"] = advanced_count + 1
    next_spec["search_epoch"] = int(state.get("search_epoch", 0) or 0)
    next_spec["progression"] = progression
    next_spec = _decorate_autonomous_experiment_spec(next_spec=next_spec, spec=spec)
    next_spec["autonomous_progression"] = {
        **dict(next_spec.get("autonomous_progression") or {}),
        **progression,
    }
    next_spec["architecture_lane"] = str(progression.get("current_lane") or next_spec.get("architecture_lane"))
    next_spec["next_lane"] = str(progression.get("next_lane") or next_spec.get("next_lane"))
    if _family_signature(next_spec) in set(frontier.get("tried_family_signatures") or []):
        return None
    return next_spec


def _architecture_progression_plan(
    *,
    policy: dict[str, Any],
    frontier: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Return deterministic architecture lanes for the next frontier epoch."""
    sentinel_count = max(0, int(policy.get("sentinel_baseline_runs") or 0))
    max_advanced = int(policy.get("max_advanced_failures_per_epoch") or 0)
    lane_stats = dict((frontier.get("lane_stats") or {}).get("architecture_family") or {})
    advanced_exhausted = _prior_epoch_exhausted_architecture(frontier=frontier, family="advanced_challenger") or (
        int(epoch or 0) > 0
        and max_advanced > 0
        and int(lane_stats.get("advanced_challenger") or 0) >= max_advanced
    )
    ensemble_exhausted = _prior_epoch_exhausted_architecture(frontier=frontier, family="ensemble_meta")
    if int(epoch or 0) > 0 and advanced_exhausted and not ensemble_exhausted:
        primary = "ensemble_meta"
        pivot_reason = "advanced_non_promotable_pivot_to_ensemble"
    elif int(epoch or 0) > 1 and advanced_exhausted and ensemble_exhausted:
        primary = "advanced_challenger"
        pivot_reason = "ensemble_not_better_pivot_window_profile"
    else:
        primary = "advanced_challenger"
        pivot_reason = "advanced_first"
    architectures = [primary]
    for family in ("advanced_challenger", "ensemble_meta", "tree_challenger", "linear_baseline"):
        if family == primary:
            continue
        if family == "ensemble_meta" and int(epoch or 0) <= 0:
            continue
        if family == "tree_challenger" and sentinel_count < 1:
            continue
        if family == "linear_baseline" and sentinel_count < 2:
            continue
        architectures.append(family)
    if not bool(policy.get("advanced_first", True)) and primary == "advanced_challenger":
        architectures = list(reversed(architectures))
    exhausted = [family for family, exhausted in {"advanced_challenger": advanced_exhausted, "ensemble_meta": ensemble_exhausted}.items() if exhausted]
    return {
        "current_lane": primary,
        "architectures": architectures,
        "exhausted_lanes": exhausted,
        "pivot_reason": pivot_reason,
        "next_lane": architectures[0],
        "epoch": int(epoch or 0),
        "last_successful_run": _last_successful_frontier_run(frontier),
    }


def _prior_epoch_exhausted_architecture(*, frontier: dict[str, Any], family: str) -> bool:
    history = [item for item in list(frontier.get("family_history") or []) if isinstance(item, dict)]
    return any(
        family in set(item.get("architecture_families") or [])
        and int(item.get("shortlist_count") or 0) <= 0
        for item in history
    )


def _last_successful_frontier_run(frontier: dict[str, Any]) -> str | None:
    history = [item for item in list(frontier.get("family_history") or []) if isinstance(item, dict)]
    for item in reversed(history):
        if item.get("best_primary_score") is not None:
            return str(item.get("experiment_id"))
    return None


def _canary_experiment_spec(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    feature_version: str | None = None,
    label_horizon: int | None = None,
) -> dict[str, Any]:
    """Build the fixed bounded architecture canary family."""
    phase = _current_ssot_phase(state=state, spec=spec)
    canary_state = {**state, "current_generation": int(state.get("current_generation", 0) or 0) + 1}
    next_spec = _make_experiment_spec_from_phase_policy(
        spec=spec,
        program_state=canary_state,
        phase_policy=phase_policy,
        experiment_id=f"{spec['program_id']}-canary-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}",
        generation=int(canary_state["current_generation"]),
    )
    allowed_data = [
        family
        for family in list(phase_policy.get("allowed_data_families") or ["price_plus_liquidity", "price_only"])
        if family in MODELING_READY_DATA_FAMILIES
    ]
    data_family = "price_plus_liquidity" if "price_plus_liquidity" in allowed_data else allowed_data[0] if allowed_data else "price_only"
    next_spec["phase"] = phase
    next_spec["ssot_phase"] = phase
    next_spec["max_concurrent"] = 1
    next_spec["matrix"] = {
        "architecture_family": ["advanced_challenger", "ensemble_meta", "tree_challenger", "linear_baseline"],
        "feature_family": ["price_liquidity"],
        "data_family": [data_family],
        "label_horizon": [int(label_horizon or (spec.get("modeling") or {}).get("primary_label_horizon") or 5)],
        "portfolio_profile": ["cost_aware_long_only_v1"],
        "data_profile": ["phase1_default"],
        "validation.initial_train_years": [2],
    }
    if feature_version:
        next_spec["matrix"]["feature_version"] = [str(feature_version)]
    next_spec["proposal_policy"]["family_size_cap"] = 4
    next_spec["frontier_architecture"] = True
    next_spec["canary"] = True
    next_spec["supervision"] = {
        **dict(next_spec.get("supervision") or {}),
        "auto_backtest_survivors": True,
        "auto_propose_next_family": False,
    }
    return _decorate_autonomous_experiment_spec(next_spec=next_spec, spec=spec)


def _frontier_label_horizon_for_epoch(*, spec: dict[str, Any], epoch: int) -> int:
    modeling = dict(spec.get("modeling") or {})
    horizons = [int(item) for item in list(modeling.get("label_horizons") or [modeling.get("primary_label_horizon") or 5])]
    primary = int(modeling.get("primary_label_horizon") or 5)
    ordered = [primary] + [horizon for horizon in horizons if horizon != primary]
    return ordered[int(epoch or 0) % len(ordered)]


def _frontier_advanced_count_for_epoch(*, state: dict[str, Any], frontier: dict[str, Any]) -> int:
    """Return advanced challenger families completed in the current search epoch."""
    epoch = int(state.get("search_epoch", 0) or 0)
    history = [item for item in list(frontier.get("family_history") or []) if isinstance(item, dict)]
    with_epoch = [item for item in history if item.get("search_epoch") is not None]
    if with_epoch:
        return sum(
            1
            for item in with_epoch
            if int(item.get("search_epoch", 0) or 0) == epoch
            and "advanced_challenger" in set(item.get("architecture_families") or [])
        )
    if epoch == 0:
        return int(((frontier.get("lane_stats") or {}).get("architecture_family") or {}).get("advanced_challenger", 0) or 0)
    return 0


def _frontier_architecture_brake_pivot_spec(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    frontier: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Open a new search epoch when the advanced lane brake fires."""
    policy = _frontier_architecture_policy(spec)
    if not bool(policy.get("enabled", False)) or not bool(policy.get("auto_pivot_on_brake", True)):
        return None
    max_advanced = int(policy.get("max_advanced_failures_per_epoch") or 0)
    if max_advanced <= 0:
        return None
    if _frontier_advanced_count_for_epoch(state=state, frontier=frontier) < max_advanced:
        return None
    next_state = deepcopy(state)
    next_state["search_epoch"] = int(state.get("search_epoch", 0) or 0) + 1
    return _frontier_architecture_spec(
        spec=spec,
        state=next_state,
        phase_policy=phase_policy,
        frontier=frontier,
        experiment_summary=experiment_summary,
    )


def _best_frontier_lane(*, frontier: dict[str, Any], lane: str, allowed: list[str], default: str) -> str:
    mapping = {
        "feature_family": "best_by_feature_family",
        "data_family": "best_by_data_family",
        "architecture_family": "best_by_architecture_family",
    }
    rows = dict(frontier.get(mapping[lane]) or {})
    ranked = sorted(
        (
            (name, float((payload or {}).get("best_primary_score") or float("-inf")))
            for name, payload in rows.items()
            if name in set(allowed)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if ranked:
        return ranked[0][0]
    return default if default in set(allowed) else allowed[0]


def _frontier_lane_for_epoch(*, frontier: dict[str, Any], lane: str, allowed: list[str], default: str, epoch: int) -> str:
    """Cycle through scored lanes across search epochs, best lane first."""
    ordered = _ranked_frontier_lanes(frontier=frontier, lane=lane, allowed=allowed)
    if not ordered:
        ordered = [default if default in set(allowed) else allowed[0]]
    return ordered[int(epoch or 0) % len(ordered)]


def _ranked_frontier_lanes(*, frontier: dict[str, Any], lane: str, allowed: list[str]) -> list[str]:
    mapping = {
        "feature_family": "best_by_feature_family",
        "data_family": "best_by_data_family",
        "architecture_family": "best_by_architecture_family",
    }
    scored = dict(frontier.get(mapping[lane]) or {})
    ranked = [
        name
        for name, _score in sorted(
            (
                (name, float((payload or {}).get("best_primary_score") or float("-inf")))
                for name, payload in scored.items()
                if name in set(allowed)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    ranked.extend(name for name in allowed if name not in set(ranked))
    return ranked


def _frontier_profile_for_epoch(epoch: int) -> str:
    """Return the data profile for a frontier search epoch."""
    return DEFAULT_EXPANSION_PROFILES[int(epoch or 0) % len(DEFAULT_EXPANSION_PROFILES)]


def _frontier_train_years_for_epoch(epoch: int) -> int:
    """Return the initial training window for a frontier search epoch."""
    values = [2, 3, 4]
    return values[int(epoch or 0) % len(values)]


def _expanded_search_spec(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any] | None:
    allowed_data_families = [
        family
        for family in list(phase_policy.get("allowed_data_families") or [])
        if family in MODELING_READY_DATA_FAMILIES
    ]
    feature_values = ["price_core", "price_liquidity", "price_short_horizon"]
    phase_value = int(phase_policy.get("ssot_phase", phase_policy.get("phase", 1)) or 1)
    architecture_values = ["linear_baseline", "tree_challenger", "ensemble_meta"]
    if phase_value > 1:
        architecture_values.append("advanced_challenger")
    profile_values = list(DEFAULT_EXPANSION_PROFILES)
    initial_train_values = list(DEFAULT_EXPANSION_INITIAL_TRAIN_YEARS)
    candidate_matrices: list[dict[str, Any]] = [
        {
            "architecture_family": architecture_values,
            "feature_family": feature_values,
            "data_family": allowed_data_families,
            "data_profile": profile_values,
            "validation.initial_train_years": initial_train_values,
        },
        {
            "architecture_family": ["linear_baseline"],
            "feature_family": feature_values,
            "data_family": allowed_data_families,
            "data_profile": profile_values,
            "validation.initial_train_years": initial_train_values,
        },
        {
            "architecture_family": ["tree_challenger"],
            "feature_family": feature_values,
            "data_family": allowed_data_families,
            "data_profile": profile_values,
            "validation.initial_train_years": initial_train_values,
        },
        {
            "architecture_family": architecture_values,
            "feature_family": ["price_core"],
            "data_family": allowed_data_families,
            "data_profile": profile_values,
            "validation.initial_train_years": [2, 3, 4],
        },
        {
            "architecture_family": architecture_values,
            "feature_family": ["price_short_horizon"],
            "data_family": allowed_data_families,
            "data_profile": profile_values,
            "validation.initial_train_years": [2, 3, 4],
        },
    ]
    family_size_cap = int(phase_policy.get("family_size_cap") or 6)
    for matrix in candidate_matrices:
        candidate = _make_experiment_spec_from_phase_policy(
            spec=spec,
            program_state=state,
            phase_policy=phase_policy,
            experiment_id=_next_experiment_id(program_state=state, spec=spec, phase=_current_ssot_phase(state=state, spec=spec)),
            generation=int(state.get("current_generation", 0) or 0) + 1,
        )
        candidate["matrix"] = _apply_program_steering_to_matrix(matrix=matrix, steering=dict(state.get("steering") or {}))
        candidate["proposal_policy"]["family_size_cap"] = family_size_cap
        candidate["search_epoch"] = int(state.get("search_epoch", 0) or 0)
        candidate = _decorate_autonomous_experiment_spec(next_spec=candidate, spec=spec)
        signature = _family_signature(candidate)
        if signature not in set(frontier.get("tried_family_signatures") or []):
            return candidate
    return None


def _apply_program_steering_to_matrix(*, matrix: dict[str, Any], steering: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(matrix)
    if "architecture_family" in updated:
        values = list(updated["architecture_family"])
        values = [value for value in values if value not in set(steering.get("avoid_architecture_families") or [])]
        prefer = [value for value in steering.get("prefer_architecture_families") or [] if value in values]
        if prefer:
            remainder = [value for value in values if value not in prefer]
            values = prefer + remainder
        if values:
            updated["architecture_family"] = values
    if "data_family" in updated:
        values = [value for value in list(updated["data_family"]) if value not in set(steering.get("avoid_data_families") or [])]
        prefer = [value for value in steering.get("prefer_data_families") or [] if value in values]
        if prefer:
            remainder = [value for value in values if value not in prefer]
            values = prefer + remainder
        if values:
            updated["data_family"] = values
    if steering.get("force_pivot"):
        if "architecture_family" in updated and len(updated["architecture_family"]) > 1:
            updated["architecture_family"] = list(reversed(updated["architecture_family"]))
        if "data_family" in updated and len(updated["data_family"]) > 1:
            updated["data_family"] = list(reversed(updated["data_family"]))
    return updated


def _update_frontier_memory(*, state: dict[str, Any], experiment_summary: dict[str, Any]) -> dict[str, Any]:
    frontier = deepcopy(state.get("frontier") or _empty_frontier())
    experiment_id = str(experiment_summary.get("experiment_id") or state.get("current_experiment_id") or "")
    if not experiment_id:
        return frontier
    signature = str(state.get("current_family_signature") or "")
    if not signature:
        return frontier
    tried = set(frontier.get("tried_family_signatures") or [])
    if signature not in tried:
        tried.add(signature)
        frontier["tried_family_signatures"] = sorted(tried)
    processed_ids = set(frontier.get("processed_experiment_ids") or [])
    family_history = list(frontier.get("family_history") or [])
    already_processed = experiment_id in processed_ids or any(
        isinstance(item, dict)
        and (item.get("experiment_id") == experiment_id or item.get("family_signature") == signature)
        for item in family_history
    )
    if already_processed:
        if experiment_id:
            processed_ids.add(experiment_id)
            frontier["processed_experiment_ids"] = sorted(processed_ids)[-200:]
        return frontier
    current_phase = _current_ssot_phase(state=state, spec={"phase_order": [state.get("current_ssot_phase") or state.get("current_phase") or 1]})
    best_score = experiment_summary.get("best_primary_score")
    if best_score is not None:
        phase_key = str(current_phase)
        frontier.setdefault("best_by_phase", {})
        current_best = frontier["best_by_phase"].get(phase_key)
        if current_best is None or float(best_score or 0.0) > float(current_best.get("best_primary_score", -10.0)):
            frontier["best_by_phase"][phase_key] = {
                "experiment_id": experiment_id,
                "best_candidate": experiment_summary.get("best_candidate"),
                "best_primary_score": best_score,
                "best_decision": experiment_summary.get("best_decision"),
            }
    lane_stats = dict(frontier.get("lane_stats") or {})
    runs = list(experiment_summary.get("runs") or [])
    for run in runs:
        matrix_values = dict(run.get("matrix_values") or {})
        score = _run_primary_score(run)
        for lane_name in ("architecture_family", "feature_family", "data_family"):
            lane_value = matrix_values.get(lane_name)
            if lane_value is None:
                continue
            counts = dict(lane_stats.get(lane_name) or {})
            counts[str(lane_value)] = int(counts.get(str(lane_value), 0) or 0) + 1
            lane_stats[lane_name] = counts
            _update_best_lane(frontier, lane_name=lane_name, lane_value=str(lane_value), run=run, score=score)
    phase_counts = dict(lane_stats.get("phase") or {})
    phase_key = str(current_phase)
    phase_counts[phase_key] = int(phase_counts.get(phase_key, 0) or 0) + 1
    lane_stats["phase"] = phase_counts
    frontier["lane_stats"] = lane_stats
    top_failures = list(experiment_summary.get("top_gate_failures") or [])
    if top_failures:
        frontier["recent_rejection_reasons"] = [item[0] for item in top_failures[:5]]
    family_history.append(
        {
            "experiment_id": experiment_id,
            "family_signature": signature,
            "phase": current_phase,
            "search_epoch": int(state.get("search_epoch", 0) or 0),
            "architecture_families": sorted(
                {
                    str((run.get("matrix_values") or {}).get("architecture_family"))
                    for run in runs
                    if isinstance(run, dict)
                    and (run.get("matrix_values") or {}).get("architecture_family") is not None
                }
            ),
            "feature_families": sorted(
                {
                    str((run.get("matrix_values") or {}).get("feature_family"))
                    for run in runs
                    if isinstance(run, dict)
                    and (run.get("matrix_values") or {}).get("feature_family") is not None
                }
            ),
            "data_families": sorted(
                {
                    str((run.get("matrix_values") or {}).get("data_family"))
                    for run in runs
                    if isinstance(run, dict)
                    and (run.get("matrix_values") or {}).get("data_family") is not None
                }
            ),
            "best_primary_score": experiment_summary.get("best_primary_score"),
            "shortlist_count": experiment_summary.get("shortlist_count"),
            "top_gate_failures": top_failures,
        }
    )
    frontier["family_history"] = family_history[-50:]
    frontier["sentinel_delta"] = _sentinel_delta(best_by_architecture=dict(frontier.get("best_by_architecture_family") or {}))
    frontier["next_lane"] = _next_frontier_lane(frontier=frontier)
    processed_ids.add(experiment_id)
    frontier["processed_experiment_ids"] = sorted(processed_ids)[-200:]
    budgets = state.setdefault("budgets", {})
    budgets["families_started"] = max(
        int(budgets.get("families_started", 0) or 0),
        len(frontier["family_history"]),
        _family_sequence(experiment_id) or 0,
    )
    budgets["runs_completed"] = int(budgets.get("runs_completed", 0) or 0) + int(experiment_summary.get("run_count", 0) or 0)
    phase_family_counts = dict(budgets.get("phase_family_counts") or {})
    phase_family_counts[phase_key] = int(phase_family_counts.get(phase_key, 0) or 0) + 1
    budgets["phase_family_counts"] = phase_family_counts
    phase_run_counts = dict(budgets.get("phase_run_counts") or {})
    phase_run_counts[phase_key] = int(phase_run_counts.get(phase_key, 0) or 0) + int(experiment_summary.get("run_count", 0) or 0)
    budgets["phase_run_counts"] = phase_run_counts
    previous_best = state.get("best_candidate_summary", {}).get("best_primary_score")
    new_best = experiment_summary.get("best_primary_score")
    if new_best is None or previous_best is None or float(new_best) <= float(previous_best):
        budgets["non_improving_families"] = int(budgets.get("non_improving_families", 0) or 0) + 1
    else:
        budgets["non_improving_families"] = 0
    if top_failures:
        top_reason = top_failures[0][0]
        previous_reason = state.get("last_transition", {}).get("top_rejection")
        if previous_reason == top_reason:
            budgets["repeat_rejection_reason_count"] = int(budgets.get("repeat_rejection_reason_count", 0) or 0) + 1
        else:
            budgets["repeat_rejection_reason_count"] = 1
    return frontier


def _program_best_candidate(*, frontier: dict[str, Any], experiment_summary: dict[str, Any]) -> dict[str, Any]:
    best_by_architecture = dict(frontier.get("best_by_architecture_family", {}) or {})
    sentinel_delta = _sentinel_delta(best_by_architecture=best_by_architecture)
    return {
        "best_candidate": experiment_summary.get("best_candidate"),
        "best_primary_score": experiment_summary.get("best_primary_score"),
        "best_backtest_net_return": experiment_summary.get("best_backtest_net_return"),
        "best_decision": experiment_summary.get("best_decision"),
        "best_decision_reason": experiment_summary.get("best_decision_reason"),
        "objective_verdict": experiment_summary.get("objective_verdict", {}),
        "architecture_lane": experiment_summary.get("architecture_lane"),
        "complexity_tier": experiment_summary.get("complexity_tier"),
        "sentinel_delta": sentinel_delta,
        "experiment_id": experiment_summary.get("experiment_id"),
        "shortlist_count": experiment_summary.get("shortlist_count"),
        "best_by_phase": frontier.get("best_by_phase", {}),
        "best_by_architecture_family": frontier.get("best_by_architecture_family", {}),
        "best_by_feature_family": frontier.get("best_by_feature_family", {}),
        "best_by_data_family": frontier.get("best_by_data_family", {}),
        "best_baseline": best_by_architecture.get("linear_baseline"),
        "best_advanced": best_by_architecture.get("advanced_challenger"),
        "best_ensemble": best_by_architecture.get("ensemble_meta"),
    }


def _sentinel_delta(*, best_by_architecture: dict[str, Any]) -> float | None:
    advanced = _safe_float((best_by_architecture.get("advanced_challenger") or {}).get("best_primary_score"))
    tree = _safe_float((best_by_architecture.get("tree_challenger") or {}).get("best_primary_score"))
    linear = _safe_float((best_by_architecture.get("linear_baseline") or {}).get("best_primary_score"))
    sentinel = max([value for value in (tree, linear) if value is not None], default=None)
    if advanced is None or sentinel is None:
        return None
    return advanced - sentinel


def _next_frontier_lane(*, frontier: dict[str, Any]) -> str:
    recent = list(frontier.get("recent_rejection_reasons") or [])
    if any("cost" in str(reason) or "net_return" in str(reason) for reason in recent):
        return "tree_challenger"
    if any("rank_ic" in str(reason) or "assessment" in str(reason) for reason in recent):
        return "ensemble_meta"
    return "advanced_challenger"


def _program_stop_reason(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    phase_policy: dict[str, Any],
    frontier: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> str | None:
    budgets = state.get("budgets") or {}
    policy = _budget_policy(spec)
    phase_key = str(_current_ssot_phase(state=state, spec=spec))
    if int(budgets.get("runs_completed", 0) or 0) >= int(policy["max_total_runs"]):
        return "total run budget exhausted"
    started_at = state.get("started_at")
    if started_at:
        with contextlib.suppress(ValueError):
            elapsed = datetime.now(tz=UTC) - datetime.fromisoformat(str(started_at))
            if elapsed >= timedelta(hours=int(policy["max_total_hours"])):
                return "total wall-clock budget exhausted"
    if int((budgets.get("phase_family_counts") or {}).get(phase_key, 0) or 0) >= int(phase_policy.get("max_generations", 999)):
        return f"phase {phase_key} generation budget exhausted"
    if int((budgets.get("phase_run_counts") or {}).get(phase_key, 0) or 0) >= int(phase_policy.get("max_phase_runs", 9999)):
        return f"phase {phase_key} run budget exhausted"
    if int(budgets.get("non_improving_families", 0) or 0) >= int(policy["max_consecutive_non_improving_families"]):
        return "too many consecutive non-improving families"
    if int(budgets.get("repeat_rejection_reason_count", 0) or 0) >= int(policy["max_repeat_rejection_reason"]):
        return "same rejection reason dominated too many consecutive families"
    if int(budgets.get("infra_failures", 0) or 0) >= int(policy["max_infra_failures"]):
        return "infra failures exceeded guardrail"
    return None


def _stop_reason_waits_for_data(*, spec: dict[str, Any], stop_reason: str) -> bool:
    if _exhaustion_mode(spec) != "wait_for_new_data":
        return False
    return any(
        fragment in stop_reason
        for fragment in (
            "phase ",
            "non-improving families",
            "same rejection reason",
            "infra failures exceeded guardrail",
        )
    )


def _phase_unlock_allowed(*, spec: dict[str, Any], state: dict[str, Any], phase_policy: dict[str, Any], experiment_summary: dict[str, Any]) -> bool:
    if not bool(phase_policy.get("auto_unlock", False)):
        return False
    freeze_phase = (state.get("steering") or {}).get("freeze_phase")
    if freeze_phase is not None and int(freeze_phase) == _current_ssot_phase(state=state, spec=spec):
        return False
    unlock_gate = dict(phase_policy.get("unlock_gate") or {})
    require_shortlist = bool(unlock_gate.get("require_shortlist", True))
    require_cost_positive = bool(unlock_gate.get("require_cost_positive", True))
    if require_shortlist and int(experiment_summary.get("shortlist_count", 0) or 0) <= 0:
        return False
    if require_cost_positive:
        best_backtest = experiment_summary.get("best_backtest_net_return")
        if best_backtest is None or float(best_backtest or 0.0) < 0.0:
            return False
    return True


def _family_signature(next_spec: dict[str, Any]) -> str:
    payload = {
        "phase": next_spec.get("phase"),
        "ssot_phase": next_spec.get("ssot_phase", next_spec.get("phase")),
        "campaign_track": next_spec.get("campaign_track"),
        "search_epoch": next_spec.get("search_epoch"),
        "frontier_iteration": next_spec.get("frontier_iteration"),
        "modeling": next_spec.get("modeling"),
        "matrix": next_spec.get("matrix"),
        "predictive_gate": next_spec.get("predictive_gate"),
        "backtest_gate": next_spec.get("backtest_gate"),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _novelty_score(*, frontier: dict[str, Any], next_spec: dict[str, Any]) -> float:
    signature = _family_signature(next_spec)
    tried = set(frontier.get("tried_family_signatures") or [])
    if signature in tried:
        return 0.0
    matrix = dict(next_spec.get("matrix") or {})
    score = 0.5
    if any(value for value in matrix.get("architecture_family", []) if value not in _lane_values(frontier, "architecture_family")):
        score += 0.2
    if any(value for value in matrix.get("feature_family", []) if value not in _lane_values(frontier, "feature_family")):
        score += 0.15
    if any(value for value in matrix.get("data_family", []) if value not in _lane_values(frontier, "data_family")):
        score += 0.15
    return min(1.0, score)


def _lane_values(frontier: dict[str, Any], lane: str) -> set[str]:
    stats = dict((frontier.get("lane_stats") or {}).get(lane) or {})
    return set(stats)


def _run_primary_score(run: dict[str, Any]) -> float | None:
    preview = dict(run.get("report_preview") or {})
    if str(run.get("model_suite") or "") == "ensemble":
        score = preview.get("ensemble_mean_rank_ic")
        if score is None:
            score = preview.get("lightgbm_mean_rank_ic")
    elif str(run.get("model_suite") or "") == "advanced":
        score = preview.get("catboost_mean_rank_ic")
        if score is None:
            score = preview.get("lightgbm_mean_rank_ic")
    else:
        score = preview.get("lightgbm_mean_rank_ic")
    if score is None:
        score = preview.get("ridge_mean_rank_ic")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _update_best_lane(
    frontier: dict[str, Any],
    *,
    lane_name: str,
    lane_value: str,
    run: dict[str, Any],
    score: float | None,
) -> None:
    if score is None:
        return
    mapping = {
        "architecture_family": "best_by_architecture_family",
        "feature_family": "best_by_feature_family",
        "data_family": "best_by_data_family",
    }
    key = mapping[lane_name]
    bucket = dict(frontier.get(key) or {})
    current = bucket.get(lane_value)
    if current is not None and float(current.get("best_primary_score", -10.0)) >= score:
        frontier[key] = bucket
        return
    bucket[lane_value] = {
        "run_id": run.get("run_id"),
        "experiment_id": run.get("experiment_id"),
        "best_candidate": run.get("model_suite"),
        "best_primary_score": score,
        "architecture_lane": (run.get("matrix_values") or {}).get("architecture_family"),
        "complexity_tier": run.get("complexity_tier"),
        "objective_verdict": run.get("objective_verdict", {}),
        "evaluation_stage": run.get("evaluation_stage"),
        "shortlisted": bool(run.get("shortlisted")),
    }
    frontier[key] = bucket


def _budget_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policy = dict(DEFAULT_BUDGET_POLICY)
    policy.update(dict(spec.get("budget_policy") or {}))
    return policy


def _exhaustion_mode(spec: dict[str, Any]) -> str:
    policy = dict(spec.get("search_policy") or {})
    return str(policy.get("exhaustion_mode") or DEFAULT_EXHAUSTION_MODE)


def _review_packet_due(*, state: dict[str, Any], spec: dict[str, Any]) -> bool:
    last = state.get("last_review_packet_at")
    cadence_hours = int((spec.get("review_policy") or {}).get("cadence_hours", DEFAULT_RESEARCH_REVIEW_HOURS))
    if not last:
        return True
    with contextlib.suppress(ValueError):
        return (datetime.now(tz=UTC) - datetime.fromisoformat(str(last))) >= timedelta(hours=cadence_hours)
    return True


def _next_phase(*, spec: dict[str, Any], phase: int) -> int | None:
    order = [int(item) for item in list(spec.get("ssot_phase_order") or spec.get("phase_order") or [])]
    if phase not in order:
        return None
    idx = order.index(phase)
    if idx + 1 >= len(order):
        return None
    return int(order[idx + 1])


def _current_data_revision(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    targets_config_path: Path,
    python_executable: str,
) -> str | None:
    try:
        target = resolve_training_target(
            target_name=str(spec.get("target") or "local"),
            targets_config_path=targets_config_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            python_executable=python_executable,
        )
        phase = _current_ssot_phase(state=state, spec=spec)
        reference_root = target.data_root / "data" / "reference"
        revision_parts = [f"report:{_resolve_default_report_date(target=target, phase=phase)}"]
        qc_path = target.data_root / "data" / "qc" / "partition_status.parquet"
        if qc_path.exists():
            revision_parts.append(f"partition_status.parquet:{int(qc_path.stat().st_mtime)}")
        curated_root = target.data_root / "data" / "curated" / "equities_ohlcv_adj"
        curated_files = [item for item in curated_root.glob("date=*/data.parquet") if item.is_file()]
        if curated_files:
            latest_curated = max(curated_files, key=lambda item: (item.parent.name, item.stat().st_mtime))
            revision_parts.append(f"equities_ohlcv_adj:{latest_curated.parent.name.partition('=')[2]}:{int(latest_curated.stat().st_mtime)}")
        registry = modeling_artifact_metadata(data_root=target.data_root)
        if registry.get("data_revision"):
            revision_parts.append(
                f"modeling:{registry.get('feature_version')}:{registry.get('label_version')}:{registry.get('data_revision')}"
            )
        for path in [
            reference_root / "security_master.parquet",
            reference_root / "fundamentals_daily.parquet",
            reference_root / "earnings_calendar_pit.parquet",
            reference_root / "sec_filing_index.parquet",
            reference_root / "macro_vintages.parquet",
            reference_root / "universe_snapshots",
        ]:
            if not path.exists():
                continue
            if path.is_dir():
                candidates = [item for item in path.rglob("*") if item.is_file()]
                if not candidates:
                    continue
                latest = max(candidates, key=lambda item: item.stat().st_mtime)
                revision_parts.append(f"{path.name}:{int(latest.stat().st_mtime)}")
            else:
                revision_parts.append(f"{path.name}:{int(path.stat().st_mtime)}")
        return "|".join(revision_parts)
    except Exception:  # noqa: BLE001
        return None


def _resume_if_data_changed(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int,
) -> dict[str, Any] | None:
    if str(state.get("status") or "").upper() == "INFRA_BLOCKED":
        if not _infra_retry_due(state=state, poll_seconds=poll_seconds):
            return None
        next_spec = deepcopy(state.get("pending_next_spec") or {})
        if not next_spec:
            return None
        state["last_infra_retry_at"] = datetime.now(tz=UTC).isoformat()
        launched = _launch_program_family(
            program_id=str(spec["program_id"]),
            program_state=state,
            next_spec=next_spec,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            targets_config_path=targets_config_path,
            python_executable=python_executable,
            poll_seconds=poll_seconds,
        )
        if launched.get("status") != "INFRA_BLOCKED":
            state["pending_next_spec"] = None
        return launched

    current_revision = _current_data_revision(
        spec=spec,
        state=state,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    previous_revision = state.get("last_seen_data_revision")
    if current_revision is None or current_revision == previous_revision:
        return None
    state["last_seen_data_revision"] = current_revision
    state["search_epoch"] = int(state.get("search_epoch", 0) or 0) + 1
    next_spec = deepcopy(state.get("pending_next_spec") or {})
    if not next_spec:
        next_spec = _build_initial_phase_experiment_spec(program_state=state, spec=spec)
    if next_spec is None:
        return None
    next_spec["search_epoch"] = int(state.get("search_epoch", 0) or 0)
    launched = _launch_program_family(
        program_id=str(spec["program_id"]),
        program_state=state,
        next_spec=next_spec,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=poll_seconds,
    )
    state["pending_next_spec"] = None
    return launched


def _infra_retry_due(*, state: dict[str, Any], poll_seconds: int) -> bool:
    """Return whether an infra-blocked pending family should retry preflight."""
    last_retry = state.get("last_infra_retry_at")
    if not last_retry:
        return True
    try:
        last_retry_at = datetime.fromisoformat(str(last_retry))
    except ValueError:
        return True
    if last_retry_at.tzinfo is None:
        last_retry_at = last_retry_at.replace(tzinfo=UTC)
    retry_seconds = max(300, int(poll_seconds) * 5)
    return datetime.now(tz=UTC) - last_retry_at >= timedelta(seconds=retry_seconds)


def _recover_stalled_active_experiment(
    *,
    state: dict[str, Any],
    experiment_supervisor: dict[str, Any] | None,
    experiment_summary: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int,
) -> dict[str, Any] | None:
    status = str((experiment_supervisor or {}).get("status") or "").upper()
    if status != "STOPPED":
        return None
    if bool((experiment_supervisor or {}).get("stop_requested")):
        return None
    if not _experiment_has_remaining_work(experiment_summary):
        return None
    spec_path_value = (
        (experiment_supervisor or {}).get("spec_path")
        or state.get("current_experiment_spec_path")
    )
    if not spec_path_value:
        return None
    return supervise_experiment(
        spec_path=Path(str(spec_path_value)),
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=poll_seconds,
        detach=True,
    )


def _sync_family_budget_with_existing_experiments(
    *,
    local_state: Path,
    program_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Keep generated family ids ahead of existing specs and summaries."""
    max_sequence = 0
    roots = [
        _program_root(local_state=local_state, program_id=program_id) / "specs",
        local_state / "experiments",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            name = path.stem if path.is_file() else path.name
            if not name.startswith(f"{program_id}-"):
                continue
            max_sequence = max(max_sequence, _family_sequence(name) or 0)
    if max_sequence <= 0:
        return state
    updated = deepcopy(state)
    budgets = dict(updated.get("budgets") or {})
    budgets["families_started"] = max(int(budgets.get("families_started", 0) or 0), max_sequence)
    updated["budgets"] = budgets
    return updated


def _experiment_has_remaining_work(summary: dict[str, Any]) -> bool:
    runs = list(summary.get("runs") or [])
    supervision = _supervision_policy(summary)
    if _count_launchable_runs(runs=[run for run in runs if isinstance(run, dict)], supervision=supervision) > 0:
        return True
    return any(str((run or {}).get("status") or "").upper() in {"RUNNING", "STARTING"} for run in runs if isinstance(run, dict))


def _launch_program_family(
    *,
    program_id: str,
    program_state: dict[str, Any],
    next_spec: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
    poll_seconds: int,
    detach: bool = True,
) -> dict[str, Any]:
    preflight = _program_family_preflight(
        next_spec=next_spec,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    if not bool(preflight.get("ok", False)):
        return {
            "status": "INFRA_BLOCKED",
            "wait_reason": f"research preflight failed: {preflight.get('reason')}",
            "last_infra_preflight": preflight,
            "pending_next_spec": deepcopy(next_spec),
        }
    spec_path = _program_root(local_state=local_state, program_id=program_id) / "specs" / f"{next_spec['experiment_id']}.yml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(next_spec, sort_keys=False), encoding="utf-8")
    payload = supervise_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        poll_seconds=poll_seconds,
        detach=detach,
    )
    return {
        "current_experiment_id": next_spec["experiment_id"],
        "current_experiment_spec_path": str(spec_path),
        "current_family_signature": _family_signature(next_spec),
        "current_generation": int(next_spec.get("generation", 0) or 0),
        "search_epoch": int(next_spec.get("search_epoch", program_state.get("search_epoch", 0)) or 0),
        "current_phase": int(next_spec.get("phase", 1) or 1),
        "current_ssot_phase": int(next_spec.get("ssot_phase", next_spec.get("phase", 1)) or 1),
        "current_track": str(next_spec.get("campaign_track") or "baseline"),
        "current_campaign_track": str(next_spec.get("campaign_track") or "baseline"),
        "architecture_lane": next_spec.get("architecture_lane"),
        "complexity_tier": next_spec.get("complexity_tier"),
        "objective_policy": next_spec.get("objective_policy", {}),
        "objective_verdict": next_spec.get("objective_verdict", {}),
        "next_lane": next_spec.get("next_lane"),
        "autonomous_progression": next_spec.get("autonomous_progression", {}),
        "active_experiment_supervisor": payload,
        "wait_reason": None,
        "waiting_since": None,
        "stop_reason": None,
        "pending_next_spec": None,
    }


def _program_family_preflight(
    *,
    next_spec: dict[str, Any],
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    target_name = str(next_spec.get("target") or "local")
    phase = int(next_spec.get("ssot_phase", next_spec.get("phase", 1)) or 1)
    config_path = repo_root / "configs" / "equities_xs.yml"
    try:
        target = resolve_training_target(
            target_name=target_name,
            targets_config_path=targets_config_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            python_executable=python_executable,
        )
        report_date = _resolve_default_report_date(target=target, phase=phase)
        payload = training_preflight(
            data_root=data_root,
            config_path=config_path,
            repo_root=repo_root,
            local_state=local_state,
            targets_config_path=targets_config_path,
            target=target_name,
            python_executable=python_executable,
            model_suite=_most_capable_model_suite(next_spec),
        )
        payload["report_date"] = report_date
        payload["feature_label"] = _feature_label_preflight_for_spec(
            next_spec=next_spec,
            target_data_root=target.data_root,
            report_date=report_date,
        )
        payload["target_paths"] = {
            "target_name": target.name,
            "target_kind": target.kind,
            "target_host": target.host,
            "target_repo_root": str(target.repo_root),
            "target_data_root": str(target.data_root),
            "controller_data_root": str(data_root),
            "controller_local_state": str(local_state),
        }
        if not bool(payload["feature_label"].get("ok", False)):
            payload["ok"] = False
            payload["reason"] = payload["feature_label"].get("reason")
        if not bool(payload.get("ok", False)):
            payload.setdefault("status", "INFRA_BLOCKED")
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "INFRA_BLOCKED",
            "reason": str(exc),
            "target_paths": {
                "target_name": target_name,
                "controller_data_root": str(data_root),
                "controller_local_state": str(local_state),
            },
        }


def _feature_label_preflight_for_spec(*, next_spec: dict[str, Any], target_data_root: Path, report_date: str | None) -> dict[str, Any]:
    matrix = dict(next_spec.get("matrix") or {})
    modeling = dict(next_spec.get("modeling") or {})
    if not bool((modeling.get("feature_store") or {}).get("enabled", False)):
        return {"ok": True, "enabled": False}
    registry = modeling_artifact_metadata(data_root=target_data_root)
    feature_version = str(
        (matrix.get("feature_version") or [None])[0]
        or modeling.get("feature_version")
        or registry.get("feature_version")
        or DEFAULT_FEATURE_VERSION
    )
    label_version = str(modeling.get("label_version") or registry.get("label_version") or DEFAULT_LABEL_VERSION)
    horizon_values = matrix.get("label_horizon") or [modeling.get("primary_label_horizon") or 5]
    checks = [
        feature_label_preflight(
            data_root=target_data_root,
            feature_version=feature_version,
            label_version=label_version,
            label_horizon=int(horizon),
            report_date=report_date,
        )
        for horizon in horizon_values
    ]
    failed = [item for item in checks if not bool(item.get("ok", False))]
    return {
        "ok": not failed,
        "enabled": True,
        "feature_version": feature_version,
        "label_version": label_version,
        "label_horizons": [int(item) for item in horizon_values],
        "checks": checks,
        "reason": "; ".join(str(item.get("reason")) for item in failed if item.get("reason")) if failed else None,
    }


def _most_capable_model_suite(spec: dict[str, Any]) -> str | None:
    """Return the most dependency-heavy model suite requested by an experiment spec."""
    suites: list[str] = []
    matrix = dict(spec.get("matrix") or {})
    for family in matrix.get("architecture_family") or []:
        preset = ARCHITECTURE_FAMILY_PRESETS.get(str(family))
        if preset is not None:
            suites.append(str(preset["model_suite"]))
    suites.extend(str(item) for item in matrix.get("model_suite") or [])
    order = {"advanced": 4, "ensemble": 3, "full": 2, "ridge_only": 1}
    return max(suites, key=lambda value: order.get(value, 0), default=None)


def _program_root(*, local_state: Path, program_id: str) -> Path:
    return local_state / "research_programs" / program_id


def _program_state_path(*, local_state: Path, program_id: str) -> Path:
    return _program_root(local_state=local_state, program_id=program_id) / "program_state.json"


def _program_log_path(*, local_state: Path, program_id: str) -> Path:
    return _program_root(local_state=local_state, program_id=program_id) / "logs" / "program.log"


def _program_lock_path(*, local_state: Path, program_id: str) -> Path:
    return _program_root(local_state=local_state, program_id=program_id) / "program.lock"


@contextlib.contextmanager
def _research_program_lock(*, local_state: Path, program_id: str) -> Iterator[bool]:
    """Hold a best-effort filesystem lock for one foreground research supervisor."""
    path = _program_lock_path(local_state=local_state, program_id=program_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            pid = _read_lock_pid(path)
            if pid is not None and not _is_local_process_running(pid):
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            acquired = False
            break
    try:
        yield acquired
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                path.unlink()


def _read_lock_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_program_state(*, local_state: Path, program_id: str, payload: dict[str, Any]) -> None:
    path = _program_state_path(local_state=local_state, program_id=program_id)
    _atomic_write_json(path, payload)


def _ensure_program_state(*, local_state: Path, program_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = read_research_program_state(local_state=local_state, program_id=program_id)
    if not existing:
        return payload
    merged = {**existing, **payload}
    payload_budgets = dict(payload.get("budgets") or {})
    existing_budgets = dict(existing.get("budgets") or {})
    merged["budgets"] = {**payload_budgets, **existing_budgets}
    for key in ("max_total_runs", "max_total_hours"):
        if key in payload_budgets:
            merged["budgets"][key] = payload_budgets[key]
    preserved_keys = (
        "frontier",
        "best_candidate_summary",
        "latest_review_packet",
        "last_review_packet_at",
        "current_phase",
        "current_ssot_phase",
        "current_track",
        "current_campaign_track",
        "current_generation",
        "search_epoch",
        "current_experiment_id",
        "current_experiment_spec_path",
        "current_family_signature",
        "active_experiment_supervisor",
        "pending_next_spec",
        "last_seen_data_revision",
        "last_transition",
        "wait_reason",
        "waiting_since",
        "architecture_lane",
        "complexity_tier",
        "objective_policy",
        "objective_verdict",
        "next_lane",
        "sentinel_delta",
        "autonomous_progression",
    )
    for key in preserved_keys:
        if key in existing and existing.get(key) not in (None, {}, []):
            merged[key] = existing[key]
    merged["steering"] = {**dict(payload.get("steering") or {}), **dict(existing.get("steering") or {})}
    merged["spec_path"] = payload.get("spec_path", merged.get("spec_path"))
    merged["poll_seconds"] = payload.get("poll_seconds", merged.get("poll_seconds"))
    return merged


def _spawn_program_supervisor_process(
    *,
    program_id: str,
    program_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    python_executable: str,
    poll_seconds: int,
) -> dict[str, Any]:
    log_path = _program_log_path(local_state=local_state, program_id=program_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    spec = _load_research_program_spec(program_path)
    payload = _initial_program_state(spec=spec, program_path=program_path, poll_seconds=poll_seconds)
    payload.update(
        {
        "pid": None,
        "status": "STARTING",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "heartbeat_at": datetime.now(tz=UTC).isoformat(),
        "paused": False,
        "stop_requested": False,
        "log_path": str(log_path),
        }
    )
    payload = _ensure_program_state(local_state=local_state, program_id=program_id, payload=payload)
    _write_program_state(local_state=local_state, program_id=program_id, payload=payload)
    command = [
        python_executable,
        "-m",
        "trademl.cli",
        "research",
        "--data-root",
        str(data_root),
        "--local-state",
        str(local_state),
        "--env-file",
        str(env_path),
        "start",
        "--program",
        str(program_path),
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
        _write_program_state(local_state=local_state, program_id=program_id, payload=payload)
        raise
    payload["pid"] = process.pid
    payload["status"] = "RUNNING"
    payload["heartbeat_at"] = datetime.now(tz=UTC).isoformat()
    _write_program_state(local_state=local_state, program_id=program_id, payload=payload)
    return payload


def _read_experiment_summary_direct(*, local_state: Path, experiment_id: str) -> dict[str, Any]:
    path = local_state / "experiments" / experiment_id / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _next_experiment_id(*, program_state: dict[str, Any], spec: dict[str, Any], phase: int) -> str:
    families_started = int((program_state.get("budgets") or {}).get("families_started", 0) or 0) + 1
    return f"{spec['program_id']}-p{phase}-f{families_started:03d}"


def _family_sequence(experiment_id: str | None) -> int | None:
    if not experiment_id:
        return None
    match = re.search(r"-f(\d+)$", str(experiment_id))
    if match is None:
        return None
    return int(match.group(1))


def _current_ssot_phase(*, state: dict[str, Any], spec: dict[str, Any]) -> int:
    order = spec.get("ssot_phase_order") or spec.get("phase_order") or [1]
    return int(
        state.get("current_ssot_phase")
        or state.get("current_phase")
        or spec.get("default_ssot_phase")
        or spec.get("default_phase")
        or order[0]
    )


def _exploration_multiplier(value: str) -> float:
    mapping = {"low": 0.75, "normal": 1.0, "high": 1.5}
    return mapping.get(str(value).lower(), 1.0)


def _program_heartbeat_stale(payload: dict[str, Any]) -> bool:
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


def _is_local_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
