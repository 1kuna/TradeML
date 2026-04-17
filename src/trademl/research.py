"""Perpetual research-program supervision on top of experiment families."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from trademl.experiments import (
    _atomic_write_json,
    _backtest_gate,
    _predictive_gate,
    _count_launchable_runs,
    _supervision_policy,
    compare_experiment,
    experiment_status,
    latest_experiment_summary,
    pause_experiment_supervisor,
    propose_next_experiment_family,
    read_experiment_supervisor_state,
    resume_experiment_supervisor,
    stop_experiment_supervisor,
    supervise_experiment,
)
from trademl.data_node.training_control import _resolve_default_report_date, resolve_training_target

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
    state["status"] = "RUNNING"
    state["stop_reason"] = None
    state["completed_at"] = None
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)

    while True:
        state = read_research_program_state(local_state=local_state, program_id=program_id)
        if not state:
            state = _initial_program_state(spec=spec, program_path=program_path, poll_seconds=resolved_poll)
        state["heartbeat_at"] = datetime.now(tz=UTC).isoformat()
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
        if state.get("status") == "WAITING_FOR_DATA":
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
    if isinstance(pid, int) and status in {"RUNNING", "PAUSED", "STARTING", "STOPPING"} and not _is_local_process_running(pid):
        payload["status"] = "STOPPED"
    return payload


def latest_research_program_summary(*, local_state: Path) -> dict[str, Any]:
    """Return the most recently updated research-program state."""
    roots = sorted((local_state / "research_programs").glob("*/program_state.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not roots:
        return {}
    return json.loads(roots[0].read_text(encoding="utf-8"))


def pause_research_program(*, local_state: Path, program_id: str) -> dict[str, Any]:
    """Pause one perpetual research program."""
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    if not state:
        raise ValueError(f"no research program state for {program_id!r}")
    state["paused"] = True
    state["status"] = "PAUSED"
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return state


def stop_research_program(*, local_state: Path, program_id: str) -> dict[str, Any]:
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
            stop_experiment_supervisor(local_state=local_state, experiment_id=str(current_experiment_id))
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
    payload = {
        "program_id": program_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": state.get("status"),
        "current_phase": state.get("current_phase"),
        "current_ssot_phase": state.get("current_ssot_phase", state.get("current_phase")),
        "current_campaign_track": state.get("current_campaign_track", state.get("current_track")),
        "current_experiment_id": current_experiment_id,
        "best_candidate_summary": state.get("best_candidate_summary", {}),
        "frontier": state.get("frontier", {}),
        "budgets": state.get("budgets", {}),
        "last_transition": state.get("last_transition", {}),
        "steering": state.get("steering", {}),
        "experiment_summary": experiment_summary,
        "comparison_best": comparison.get("best"),
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
        f"- best_candidate: {(payload['best_candidate_summary'] or {}).get('best_candidate') or '-'}",
        f"- best_primary_score: {(payload['best_candidate_summary'] or {}).get('best_primary_score')}",
        f"- last_transition: {(payload['last_transition'] or {}).get('reason') or '-'}",
        "",
        "## Top rejections",
        *[f"- {reason}: {count}" for reason, count in payload["top_rejections"]],
        "",
        "## Best comparison row",
        f"- {json.dumps(payload['comparison_best'], default=str)}",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    result = {"json_path": str(json_path), "markdown_path": str(md_path)}
    state["latest_review_packet"] = result
    state["last_review_packet_at"] = datetime.now(tz=UTC).isoformat()
    _write_program_state(local_state=local_state, program_id=program_id, payload=state)
    return result


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
        "pid": os.getpid(),
        "poll_seconds": poll_seconds,
        "paused": False,
        "stop_requested": False,
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
        "recent_rejection_reasons": [],
        "family_history": [],
    }


def _build_initial_phase_experiment_spec(*, program_state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    phase_policy = _phase_policy(spec=spec, phase=_current_ssot_phase(state=program_state, spec=spec))
    if phase_policy is None:
        return None
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
    return {
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
        "target": str(spec.get("target") or "workstation-remote"),
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
    next_spec["target"] = str(spec.get("target") or next_spec.get("target") or "workstation-remote")
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
    architecture_values = ["linear_baseline", "tree_challenger"]
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
    family_history = list(frontier.get("family_history") or [])
    family_history.append(
        {
            "experiment_id": experiment_id,
            "family_signature": signature,
            "phase": current_phase,
            "best_primary_score": experiment_summary.get("best_primary_score"),
            "shortlist_count": experiment_summary.get("shortlist_count"),
            "top_gate_failures": top_failures,
        }
    )
    frontier["family_history"] = family_history[-50:]
    budgets = state.setdefault("budgets", {})
    budgets["families_started"] = max(int(budgets.get("families_started", 0) or 0), len(frontier["family_history"]))
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
    return {
        "best_candidate": experiment_summary.get("best_candidate"),
        "best_primary_score": experiment_summary.get("best_primary_score"),
        "best_backtest_net_return": experiment_summary.get("best_backtest_net_return"),
        "best_decision": experiment_summary.get("best_decision"),
        "best_decision_reason": experiment_summary.get("best_decision_reason"),
        "experiment_id": experiment_summary.get("experiment_id"),
        "shortlist_count": experiment_summary.get("shortlist_count"),
        "best_by_phase": frontier.get("best_by_phase", {}),
        "best_by_architecture_family": frontier.get("best_by_architecture_family", {}),
        "best_by_feature_family": frontier.get("best_by_feature_family", {}),
        "best_by_data_family": frontier.get("best_by_data_family", {}),
        "best_baseline": best_by_architecture.get("linear_baseline"),
        "best_advanced": best_by_architecture.get("advanced_challenger"),
    }


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
    if str(run.get("model_suite") or "") == "advanced":
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
            target_name=str(spec.get("target") or "workstation-remote"),
            targets_config_path=targets_config_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            python_executable=python_executable,
        )
        phase = _current_ssot_phase(state=state, spec=spec)
        reference_root = target.data_root / "data" / "reference"
        revision_parts = [f"report:{_resolve_default_report_date(target=target, phase=phase)}"]
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
) -> dict[str, Any]:
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
        detach=True,
    )
    return {
        "current_experiment_id": next_spec["experiment_id"],
        "current_experiment_spec_path": str(spec_path),
        "current_family_signature": _family_signature(next_spec),
        "current_generation": int(next_spec.get("generation", 0) or 0),
        "current_phase": int(next_spec.get("phase", 1) or 1),
        "current_ssot_phase": int(next_spec.get("ssot_phase", next_spec.get("phase", 1)) or 1),
        "current_track": str(next_spec.get("campaign_track") or "baseline"),
        "current_campaign_track": str(next_spec.get("campaign_track") or "baseline"),
        "active_experiment_supervisor": payload,
    }


def _program_root(*, local_state: Path, program_id: str) -> Path:
    return local_state / "research_programs" / program_id


def _program_state_path(*, local_state: Path, program_id: str) -> Path:
    return _program_root(local_state=local_state, program_id=program_id) / "program_state.json"


def _program_log_path(*, local_state: Path, program_id: str) -> Path:
    return _program_root(local_state=local_state, program_id=program_id) / "logs" / "program.log"


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
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=repo_root, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
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


def _is_local_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
