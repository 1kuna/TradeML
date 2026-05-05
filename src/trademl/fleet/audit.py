"""Unified fleet audit and Codex curation feed."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trademl.fleet.autopilot import collect_fleet_health, write_codex_issue_bucket
from trademl.fleet.data_quality import run_data_quality_audit
from trademl.research_audit import run_research_progression_audit


ISSUE_VALUE_RANK = {
    "blocked_collection_capacity": 100,
    "data_quality_failures": 90,
    "research_no_progress": 80,
    "paper_evidence_failures": 70,
    "cleanup_maintenance": 10,
}
SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}


def run_fleet_audit(
    *,
    local_snapshot: dict[str, Any],
    data_root: Path,
    repo_root: Path,
    local_state: Path,
    targets_config_path: Path,
    python_executable: str,
    program_id: str = "perpetual-macmini",
    pi: dict[str, str] | None = None,
    mac: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one full autopilot audit pass and persist backend artifacts."""
    current = now or datetime.now(tz=UTC)
    data_quality = run_data_quality_audit(data_root=data_root, now=current)
    research = _safe_research_audit(
        program_id=program_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
        now=current,
    )
    fleet = collect_fleet_health(
        local_snapshot=local_snapshot,
        data_root=data_root,
        pi=pi,
        mac=mac,
        heal=False,
    )
    observability = dict(fleet.get("observability") or {})
    paper = dict(observability.get("paper_pnl") or {})
    issues = _audit_issues(
        fleet=fleet,
        observability=observability,
        data_quality=data_quality,
        research=research,
        paper=paper,
        config=config or {},
    )
    bucket = write_codex_issue_bucket(data_root=data_root, issues=issues, now=current)
    open_bucket_issues = _open_issues(list(bucket.get("issues") or issues))
    codex_feed = build_codex_feed(
        issues=open_bucket_issues,
        data_quality=data_quality,
        research=research,
        observability=observability,
        now=current,
    )
    codex_feed_paths = write_codex_feed(data_root=data_root, payload=codex_feed, now=current)
    audit_latest = data_root / "control" / "cluster" / "state" / "autopilot" / "audit" / "latest.json"
    systems = dict((fleet.get("current_state") or {}).get("systems") or {})
    if mac is None:
        systems["mac"] = _local_mac_system_from_research(research)
    payload = {
        "version": "fleet_audit_v1",
        "generated_at": current.isoformat(),
        "verdict": _verdict_for_issues(open_bucket_issues),
        "systems": systems,
        "collection": {
            "saturation": observability.get("collection_saturation") or {},
            "controller": observability.get("controller") or {},
            "vendors": observability.get("vendors") or {},
        },
        "data_quality": data_quality,
        "research_intelligence": research,
        "paper": paper,
        "issues": open_bucket_issues,
        "suggested_codex_actions": codex_feed.get("ranked_actions") or [],
        "codex_feed": codex_feed,
        "artifact_path": str(audit_latest),
        "codex_feed_path": codex_feed_paths.get("latest_path"),
    }
    write_fleet_audit(data_root=data_root, payload=payload, now=current)
    return payload


def _open_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only current unresolved issues from an issue-bucket read."""
    return [issue for issue in issues if str(issue.get("status") or "open") == "open"]


def build_codex_feed(
    *,
    issues: list[dict[str, Any]],
    data_quality: dict[str, Any],
    research: dict[str, Any],
    observability: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Rank current actionable issues for Codex/operator inspection."""
    current = now or datetime.now(tz=UTC)
    open_issues = [issue for issue in issues if str(issue.get("status") or "open") == "open"]
    enriched = []
    for issue in open_issues:
        category = _issue_category(issue)
        enriched.append(
            {
                **issue,
                "category": category,
                "expected_value_rank": ISSUE_VALUE_RANK.get(category, 0),
            }
        )
    ranked = sorted(
        enriched,
        key=lambda issue: (
            int(issue.get("expected_value_rank") or 0),
            SEVERITY_RANK.get(str(issue.get("severity") or "info"), 0),
            int(issue.get("count") or 0),
            str(issue.get("last_seen") or issue.get("generated_at") or ""),
        ),
        reverse=True,
    )
    return {
        "version": "codex_feed_v1",
        "generated_at": current.isoformat(),
        "ranked_actions": [
            {
                "rank": index + 1,
                "category": issue.get("category"),
                "severity": issue.get("severity"),
                "kind": issue.get("kind"),
                "message": issue.get("message"),
                "suggested_codex_action": issue.get("suggested_codex_action"),
                "source": issue.get("source"),
            }
            for index, issue in enumerate(ranked[:25])
        ],
        "issue_count": len(open_issues),
        "data_quality_summary": data_quality.get("summary") or {},
        "research_verdict": research.get("verdict"),
        "collection_underutilized": (observability.get("vendors") or {}).get("underutilized_count"),
    }


def write_fleet_audit(*, data_root: Path, payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Persist the latest full audit artifact."""
    current = now or datetime.now(tz=UTC)
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "audit"
    latest = root / "latest.json"
    _atomic_write_json(latest, payload)
    history = root / "history" / current.date().isoformat() / f"{current.strftime('%H%M%S')}.json"
    _atomic_write_json(history, payload)
    return {"latest_path": str(latest), "history_path": str(history)}


def write_codex_feed(*, data_root: Path, payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Persist the Codex curation feed."""
    current = now or datetime.now(tz=UTC)
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "codex_feed"
    latest = root / "latest.json"
    _atomic_write_json(latest, payload)
    history = root / "history" / current.date().isoformat() / f"{current.strftime('%H%M%S')}.json"
    _atomic_write_json(history, payload)
    return {"latest_path": str(latest), "history_path": str(history)}


def _safe_research_audit(**kwargs: Any) -> dict[str, Any]:
    try:
        return run_research_progression_audit(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "version": "research_progression_audit_v1",
            "verdict": "DEGRADED",
            "issues": [
                {
                    "source": "research",
                    "severity": "warning",
                    "kind": "research_audit_failed",
                    "message": f"research audit failed: {exc}",
                    "suggested_codex_action": "Inspect research audit inputs and program state.",
                }
            ],
        }


def _local_mac_system_from_research(research: dict[str, Any]) -> dict[str, Any]:
    status = str(research.get("status") or "").upper()
    launchd = dict(research.get("launchd") or {})
    if status in {"RUNNING", "FEATURE_CANARY_RUNNING"}:
        return {
            "status": "online",
            "headline": "Research running",
            "detail": str(research.get("current_experiment_id") or "-"),
            "launchd": launchd,
        }
    if status in {"INFRA_BLOCKED", "WAITING_FOR_DATA"}:
        return {
            "status": "degraded",
            "headline": status.replace("_", " ").title(),
            "detail": str(research.get("wait_reason") or research.get("pivot_reason") or "-"),
            "launchd": launchd,
        }
    if status:
        return {
            "status": "degraded",
            "headline": status.replace("_", " ").title(),
            "detail": str(research.get("current_experiment_id") or "-"),
            "launchd": launchd,
        }
    return {"status": "unknown", "headline": "Unknown", "detail": "-", "launchd": launchd}


def _audit_issues(
    *,
    fleet: dict[str, Any],
    observability: dict[str, Any],
    data_quality: dict[str, Any],
    research: dict[str, Any],
    paper: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    _ = config
    issues: list[dict[str, Any]] = []
    issues.extend(list(fleet.get("current_issues") or []))
    issues.extend(list(observability.get("issues") or []))
    for row in list(data_quality.get("rows") or []):
        verdict = str(row.get("verdict") or "")
        if verdict in {"CRITICAL", "WARNING"}:
            issues.append(
                {
                    "source": "data",
                    "severity": "critical" if verdict == "CRITICAL" else "warning",
                    "kind": "data_quality_failure",
                    "message": f"{row.get('dataset')} quality check is {row.get('status')}: {row.get('reason') or '-'}",
                    "suggested_codex_action": "Inspect data quality latest.json and source partitions.",
                }
            )
    issues.extend(list(research.get("issues") or []))
    smoke_status = str(paper.get("paper_account_smoke_status") or "").lower()
    if smoke_status and smoke_status != "ok":
        issues.append(
            {
                "source": "paper",
                "severity": "warning",
                "kind": "paper_smoke_failed",
                "message": f"paper account smoke status is {smoke_status}",
                "suggested_codex_action": "Run read-only paper-smoke and inspect credentials; do not submit orders.",
            }
        )
    return _dedupe_issues(issues)


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        if str(issue.get("status") or "open") != "open":
            continue
        key = (str(issue.get("source") or ""), str(issue.get("kind") or ""), str(issue.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _issue_category(issue: dict[str, Any]) -> str:
    kind = str(issue.get("kind") or "")
    source = str(issue.get("source") or "")
    if kind in {"vendor_underutilized", "idle_vendor_budget", "collection_underutilized"}:
        return "blocked_collection_capacity"
    if kind.startswith("data_quality") or "schema" in kind or "source" in kind:
        return "data_quality_failures"
    if source == "research" or kind.startswith("research") or "rejection" in kind:
        return "research_no_progress"
    if source == "paper" or "paper" in kind:
        return "paper_evidence_failures"
    return "cleanup_maintenance"


def _verdict_for_issues(issues: list[dict[str, Any]]) -> str:
    if any(str(issue.get("severity")) == "critical" for issue in issues):
        return "BLOCKED"
    if issues:
        return "DEGRADED"
    return "OK"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
