"""Fleet watchdog orchestration for scheduled current-state checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trademl.fleet.autopilot import collect_fleet_health

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def run_fleet_watchdog_once(
    *,
    local_snapshot: dict[str, Any],
    data_root: Path,
    pi: dict[str, Any] | None = None,
    mac: dict[str, Any] | None = None,
    heal: bool = False,
    now: datetime | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one fleet watchdog pass and persist its current-state artifact."""
    current = now or datetime.now(tz=UTC)
    rules = _watchdog_policy(policy)
    fleet = collect_fleet_health(local_snapshot=local_snapshot, data_root=data_root, pi=pi, mac=mac, heal=heal)
    alerts = _watchdog_alerts(fleet=fleet, now=current, policy=rules)
    action = _action_for(fleet=fleet, alerts=alerts)
    payload = {
        "generated_at": current.isoformat(),
        "verdict": fleet.get("verdict"),
        "action": action,
        "heal_requested": bool(heal),
        "recovery_attempts": _recovery_attempts(fleet),
        "alerts": alerts,
        "fleet_health": fleet,
        "no_live_orders": True,
    }
    write = write_fleet_watchdog(data_root=data_root, payload=payload, now=current)
    payload["write"] = write
    return payload


def write_fleet_watchdog(
    *,
    data_root: Path,
    payload: dict[str, Any],
    now: datetime | None = None,
    write_history: bool = True,
) -> dict[str, Any]:
    """Persist the latest watchdog payload and optional history."""
    current = now or datetime.now(tz=UTC)
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "watchdog"
    latest = root / "latest.json"
    written: list[str] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        written.append(str(latest))
        if write_history:
            history = root / "history" / current.date().isoformat()
            history.mkdir(parents=True, exist_ok=True)
            history_path = history / f"{current.strftime('%H%M%S')}.json"
            history_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
            written.append(str(history_path))
    except OSError as exc:
        return {
            "watchdog_root": str(root),
            "latest_path": str(latest),
            "written": written,
            "status": "write_failed",
            "reason": str(exc),
        }
    return {"watchdog_root": str(root), "latest_path": str(latest), "written": written, "status": "written"}


def _watchdog_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(policy or {})
    return {
        "paper_smoke_stale_hours": int(raw.get("paper_smoke_stale_hours") or 24),
        "repeated_issue_count": int(raw.get("repeated_issue_count") or 3),
    }


def _watchdog_alerts(*, fleet: dict[str, Any], now: datetime, policy: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    bucket = dict(fleet.get("issue_bucket") or {})
    for issue in list(bucket.get("issues") or []):
        if issue.get("resolved_at"):
            continue
        count = int(issue.get("count") or 1)
        severity = str(issue.get("severity") or "info")
        if severity == "critical" or count >= int(policy["repeated_issue_count"]):
            alerts.append(
                _alert(
                    kind=f"repeated_{issue.get('kind')}",
                    severity="critical" if severity == "critical" else "warning",
                    message=str(issue.get("message") or issue.get("kind") or "open fleet issue"),
                    suggested_action=str(issue.get("suggested_codex_action") or "Inspect the Codex issue bucket."),
                    source=str(issue.get("source") or "fleet"),
                    count=count,
                )
            )
    paper = dict((fleet.get("observability") or {}).get("paper_pnl") or {})
    smoke_status = paper.get("paper_account_smoke_status")
    if smoke_status and smoke_status != "ok":
        alerts.append(
            _alert(
                kind="paper_smoke_failed",
                severity="warning",
                message=f"Alpaca paper account smoke status is {smoke_status}",
                suggested_action="Run research paper-smoke on the Mac and inspect persisted state.",
                source="mac",
            )
        )
    checked_at = paper.get("paper_account_smoke_checked_at")
    if checked_at:
        parsed = _parse_time(str(checked_at))
        if parsed and now - parsed > timedelta(hours=int(policy["paper_smoke_stale_hours"])):
            alerts.append(
                _alert(
                    kind="paper_smoke_stale",
                    severity="info",
                    message=f"Alpaca paper account smoke is stale: checked_at={checked_at}",
                    suggested_action="Refresh read-only paper smoke when paper validation is next reviewed.",
                    source="mac",
                )
            )
    research = dict((fleet.get("observability") or {}).get("research") or {})
    if str(research.get("status")) == "RUNNING" and not research.get("current_experiment_id"):
        alerts.append(
            _alert(
                kind="research_running_without_active_experiment",
                severity="warning",
                message="Research supervisor is running but no active experiment is visible.",
                suggested_action="Inspect research health and stale-run sweeper state.",
                source="mac",
            )
        )
    return alerts


def _alert(
    *,
    kind: str,
    severity: str,
    message: str,
    suggested_action: str,
    source: str,
    count: int = 1,
) -> dict[str, Any]:
    return {
        "source": source,
        "severity": severity,
        "kind": kind,
        "message": message,
        "count": int(count),
        "suggested_codex_action": suggested_action,
    }


def _action_for(*, fleet: dict[str, Any], alerts: list[dict[str, Any]]) -> str:
    if fleet.get("verdict") == "BLOCKED" or any(SEVERITY_RANK.get(str(a.get("severity")), 0) >= 2 for a in alerts):
        return "Codex should inspect"
    if fleet.get("verdict") == "DEGRADED" or alerts:
        return "Watch"
    return "OK"


def _recovery_attempts(fleet: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    remote = dict(fleet.get("remote") or {})
    for name in ("pi", "mac"):
        payload = dict(remote.get(name) or {})
        recovery = payload.get("recovery")
        if recovery:
            attempts.append({"target": name, "recovery": recovery})
    return attempts


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
