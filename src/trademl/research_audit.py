"""Research progression audit artifacts for autonomous Mac supervision."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trademl.research import (
    read_research_progression_events,
    read_feature_family_leaderboard,
    read_research_program_state,
    research_health,
)


def run_research_progression_audit(
    *,
    program_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
    health: dict[str, Any] | None = None,
    now: datetime | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Write a compact audit of whether research is progressing intelligently."""
    current = now or datetime.now(tz=UTC)
    health_payload = health or research_health(
        program_id=program_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    state = read_research_program_state(local_state=local_state, program_id=program_id)
    leaderboard = read_feature_family_leaderboard(data_root=data_root)
    evidence = _latest_candidate_evidence(data_root=data_root, limit=20)
    recorded_events = read_research_progression_events(data_root=data_root, program_id=program_id, limit=50)
    events = _progression_events(
        health=health_payload,
        state=state,
        leaderboard=leaderboard,
        evidence=evidence,
        now=current,
    )
    issues = _research_audit_issues(
        health=health_payload,
        state=state,
        leaderboard=leaderboard,
        evidence=evidence,
        now=current,
    )
    payload = {
        "version": "research_progression_audit_v1",
        "generated_at": current.isoformat(),
        "program_id": program_id,
        "status": health_payload.get("status"),
        "current_experiment_id": health_payload.get("current_experiment_id"),
        "architecture_lane": health_payload.get("architecture_lane"),
        "next_lane": health_payload.get("next_lane"),
        "pivot_reason": health_payload.get("pivot_reason"),
        "latest_finished_at": (health_payload.get("research_throughput") or {}).get("latest_finished_at"),
        "completed_runs_24h": health_payload.get("completed_runs_24h"),
        "top_rejection_reasons": health_payload.get("top_rejection_reasons") or [],
        "frontier": health_payload.get("frontier_architecture") or {},
        "launchd": health_payload.get("launchd") or {},
        "feature_leaderboard": {
            "updated_at": leaderboard.get("updated_at"),
            "best_feature_version": leaderboard.get("best_feature_version"),
            "best_objective_score": leaderboard.get("best_objective_score"),
            "entries": list(leaderboard.get("entries") or [])[:8],
        },
        "candidate_evidence": evidence,
        "paper": {
            "smoke": health_payload.get("latest_paper_account_smoke") or {},
            "pnl": health_payload.get("latest_paper_pnl") or {},
        },
        "active_run": health_payload.get("active_run") or {},
        "latest_finished_age_seconds": _age_seconds(
            value=(health_payload.get("research_throughput") or {}).get("latest_finished_at"),
            now=current,
        ),
        "recorded_events": recorded_events,
        "events": [*recorded_events[-25:], *events],
        "issues": issues,
        "verdict": _verdict_for_issues(issues),
    }
    if write:
        path = data_root / "control" / "cluster" / "state" / "research" / "progression_audit" / "latest.json"
        _atomic_write_json(path, payload)
        payload["path"] = str(path)
    return payload


def _progression_events(
    *,
    health: dict[str, Any],
    state: dict[str, Any],
    leaderboard: dict[str, Any],
    evidence: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if health.get("current_experiment_id"):
        events.append(
            {
                "event": "supervisor_status",
                "created_at": now.isoformat(),
                "status": health.get("status"),
                "experiment_id": health.get("current_experiment_id"),
                "architecture_lane": health.get("architecture_lane"),
                "next_lane": health.get("next_lane"),
            }
        )
    transition = dict(state.get("last_transition") or {})
    if transition:
        events.append({"event": "pivot_selected", **transition})
    for item in evidence[:5]:
        autopsy = dict(item.get("candidate_autopsy") or {})
        events.append(
            {
                "event": "candidate_classified",
                "created_at": item.get("created_at") or item.get("updated_at") or now.isoformat(),
                "experiment_id": item.get("experiment_id"),
                "run_id": item.get("run_id"),
                "classification": autopsy.get("classification"),
                "root_failure_mode": autopsy.get("root_failure_mode"),
                "recommended_next_action": item.get("recommended_next_action") or autopsy.get("recommended_follow_up"),
            }
        )
    for entry in list(leaderboard.get("entries") or [])[:5]:
        if not isinstance(entry, dict):
            continue
        events.append(
            {
                "event": "feature_version_evaluated",
                "created_at": entry.get("recorded_at") or leaderboard.get("updated_at") or now.isoformat(),
                "feature_version": entry.get("feature_version"),
                "verdict": entry.get("verdict"),
                "readiness_status": entry.get("readiness_status"),
                "top_rejection_reason": entry.get("top_rejection_reason"),
            }
        )
    return events


def _research_audit_issues(
    *,
    health: dict[str, Any],
    state: dict[str, Any],
    leaderboard: dict[str, Any],
    evidence: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    status = str(health.get("status") or "").upper()
    latest_finished = (health.get("research_throughput") or {}).get("latest_finished_at")
    if status == "RUNNING" and latest_finished:
        parsed = _parse_utc_datetime(str(latest_finished))
        if parsed is not None and now - parsed >= timedelta(hours=2):
            issues.append(
                _issue(
                    "mac",
                    "warning",
                    "research_no_completed_runs",
                    f"no completed research runs for {(now - parsed).total_seconds() / 3600:.1f}h while supervisor is runnable",
                    "Inspect active run, preflight, and LaunchAgent logs.",
                )
            )
    frontier = dict(health.get("frontier_architecture") or {})
    if frontier.get("brake_active") and not health.get("next_lane"):
        issues.append(
            _issue(
                "research",
                "warning",
                "lane_exhausted_without_pivot",
                "frontier brake is active but no next lane is planned",
                "Inspect autonomous progression state and diagnostic planner.",
            )
        )
    top_reasons = list(health.get("top_rejection_reasons") or [])
    if top_reasons and int(top_reasons[0].get("count") or 0) >= 10:
        latest_classifications = {
            str((item.get("candidate_autopsy") or {}).get("classification") or "")
            for item in evidence[:10]
        }
        if not any(value.startswith("strong_") for value in latest_classifications):
            issues.append(
                _issue(
                    "research",
                    "warning",
                    "repeated_rejection_without_diagnostic",
                    f"same rejection repeated {top_reasons[0].get('count')} times: {top_reasons[0].get('reason')}",
                    "Inspect candidate evidence and whether a diagnostic family should be planned.",
                )
            )
    for entry in list(leaderboard.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("readiness_status") or "").upper() == "READY":
            for group, coverage in dict(entry.get("source_coverage") or {}).items():
                if str(coverage.get("readiness_status") or "").upper() == "BLOCKED":
                    issues.append(
                        _issue(
                            "research",
                            "warning",
                            "feature_readiness_contradiction",
                            f"{entry.get('feature_version')} is READY while group {group} is blocked",
                            "Inspect feature readiness metadata and source contract.",
                        )
                    )
                    break
    smoke = dict(health.get("latest_paper_account_smoke") or {})
    checked_at = _parse_utc_datetime(str(smoke.get("checked_at") or "")) if smoke else None
    if checked_at is not None and now - checked_at >= timedelta(hours=24):
        issues.append(
            _issue(
                "paper",
                "warning",
                "paper_smoke_stale",
                "Alpaca paper account smoke check is older than 24h",
                "Run read-only paper-smoke; do not submit orders.",
            )
        )
    paper_pnl = dict(health.get("latest_paper_pnl") or {})
    failures = list(paper_pnl.get("evidence_failures") or paper_pnl.get("failures") or [])
    if any("backtest" in str(item).lower() and "paper" in str(item).lower() for item in failures):
        issues.append(
            _issue(
                "paper",
                "warning",
                "paper_backtest_disagreement",
                "mature paper evidence disagrees with backtest evidence",
                "Inspect paper evidence before considering promotion.",
            )
        )
    if str(status) in {"INFRA_BLOCKED", "WAITING_FOR_DATA"}:
        issues.append(
            _issue(
                "mac",
                "critical",
                "research_infra_or_data_blocked",
                str(health.get("wait_reason") or "research is blocked"),
                "Inspect research health and preflight blockers.",
            )
        )
    return issues


def _latest_candidate_evidence(*, data_root: Path, limit: int) -> list[dict[str, Any]]:
    root = data_root / "control" / "cluster" / "state" / "research" / "candidate_evidence"
    if not root.exists():
        return []
    paths = sorted(root.glob("*/*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]
    payloads = []
    for path in paths:
        with contextlib.suppress(Exception):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["path"] = str(path)
            payloads.append(payload)
    return payloads


def _issue(source: str, severity: str, kind: str, message: str, suggested: str) -> dict[str, Any]:
    return {
        "source": source,
        "severity": severity,
        "kind": kind,
        "message": message,
        "suggested_codex_action": suggested,
    }


def _verdict_for_issues(issues: list[dict[str, Any]]) -> str:
    if any(str(issue.get("severity")) == "critical" for issue in issues):
        return "BLOCKED"
    if issues:
        return "DEGRADED"
    return "OK"


def _parse_utc_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _age_seconds(*, value: str | None, now: datetime) -> float | None:
    parsed = _parse_utc_datetime(str(value or ""))
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
