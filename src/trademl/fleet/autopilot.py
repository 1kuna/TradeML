"""Fleet-level autopilot health summaries and Codex issue bucket helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def build_current_state_summary(snapshot: dict[str, Any], *, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the user-facing current-state dashboard summary."""
    issues = issues or []
    pi = _pi_summary(snapshot)
    mac = _mac_summary(snapshot)
    architecture = _architecture_summary(snapshot)
    profit = _profit_summary(snapshot)
    issue_summary = summarize_issues(issues)
    verdict = _rollup_verdict([pi["status"], mac["status"]], issue_summary)
    return {
        "verdict": verdict,
        "action": _action_from_verdict(verdict, issue_summary),
        "pi": pi,
        "mac": mac,
        "architecture": architecture,
        "profit": profit,
        "codex": issue_summary,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def collect_current_state_issues(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect dashboard-visible issues for the Codex bucket."""
    issues: list[dict[str, Any]] = []
    runtime = dict(snapshot.get("runtime") or {})
    if not bool(runtime.get("running")):
        issues.append(_issue("pi", "critical", "node_offline", "Pi data node is not running", "Check Pi systemd service and node logs."))
    collection = dict(snapshot.get("collection_status") or {})
    if int(collection.get("repair_remaining_units") or 0) > 0:
        issues.append(
            _issue(
                "pi",
                "warning",
                "repair_backlog",
                f"{int(collection.get('repair_remaining_units') or 0)} canonical repair units remain",
                "Inspect repair-status and recent bad dates.",
            )
        )
    budget = dict(snapshot.get("budget_summary") or {})
    if bool(budget.get("stale")):
        issues.append(_issue("pi", "warning", "budget_snapshot_stale", "Budget snapshot is stale", "Inspect the data-node budget snapshot and node heartbeat."))
    program = dict((snapshot.get("health") or {}).get("research_program_summary") or {})
    launchd = dict(program.get("launchd") or {})
    if program and str(program.get("status") or "").upper() in {"INFRA_BLOCKED", "WAITING_FOR_DATA"}:
        issues.append(_issue("mac", "critical", "research_blocked", str(program.get("wait_reason") or "Research autopilot is blocked"), "Run research health and inspect infra blocker."))
    if launchd and not bool(launchd.get("loaded")):
        issues.append(_issue("mac", "critical", "launchd_not_loaded", "Mac research LaunchAgent is not loaded", "Kickstart or reinstall the Mac LaunchAgent."))
    for alert in list(program.get("latest_drift_alerts") or program.get("drift_alerts") or []):
        issues.append(
            _issue(
                "research",
                str(alert.get("severity") or "warning"),
                str(alert.get("kind") or "research_alert"),
                str(alert.get("message") or "Research alert"),
                "Inspect research alerts and latest review packet.",
            )
        )
    return issues


def write_codex_issue_bucket(*, data_root: Path, issues: list[dict[str, Any]], now: datetime | None = None) -> dict[str, Any]:
    """Upsert current issues into the NAS-backed Codex issue bucket."""
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "issues"
    now = now or datetime.now(tz=UTC)
    if not data_root.exists():
        unavailable = _normalize_issue(
            _issue(
                "nas",
                "critical",
                "nas_unavailable",
                f"Configured data root is unavailable: {data_root}",
                "Verify the NAS mount before trusting dashboard state.",
            ),
            now=now,
        )
        return {
            "issue_root": str(root),
            "written": [],
            "issues": [unavailable],
            "summary": summarize_issues([unavailable]),
        }
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        failure = _normalize_issue(
            _issue(
                "nas",
                "critical",
                "issue_bucket_unwritable",
                f"Codex issue bucket is not writable: {exc}",
                "Inspect NAS permissions and mount health.",
            ),
            now=now,
        )
        return {"issue_root": str(root), "written": [], "issues": [failure], "summary": summarize_issues([failure])}
    written: list[str] = []
    current_ids: set[str] = set()
    for issue in issues:
        normalized = _normalize_issue(issue, now=now)
        current_ids.add(str(normalized["id"]))
        path = root / f"{normalized['id']}.json"
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            normalized["first_seen"] = previous.get("first_seen") or normalized["first_seen"]
            normalized["count"] = int(previous.get("count") or 0) + 1
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        written.append(str(path))
    for path in root.glob("*.json"):
        if path.stem in current_ids:
            continue
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(previous.get("status") or "open") != "open":
            continue
        previous["status"] = "resolved"
        previous["resolved_at"] = now.isoformat()
        previous["last_seen"] = previous.get("last_seen") or now.isoformat()
        previous["updated_at"] = now.isoformat()
        path.write_text(json.dumps(previous, indent=2, sort_keys=True), encoding="utf-8")
    latest = read_codex_issue_bucket(data_root=data_root)
    return {"issue_root": str(root), "written": written, "issues": latest["issues"], "summary": summarize_issues(latest["issues"])}


def read_codex_issue_bucket(*, data_root: Path, limit: int = 50) -> dict[str, Any]:
    """Read recent Codex issue bucket records."""
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "issues"
    if not root.exists():
        return {"issue_root": str(root), "issues": [], "summary": summarize_issues([])}
    paths = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]
    issues = []
    for path in paths:
        try:
            issues.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return {"issue_root": str(root), "issues": issues, "summary": summarize_issues(issues)}


def summarize_issues(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize issue bucket records for the high-level dashboard."""
    unresolved = [issue for issue in issues if str(issue.get("status") or "open") == "open"]
    latest = sorted(unresolved, key=lambda issue: (SEVERITY_RANK.get(str(issue.get("severity") or "info"), 0), str(issue.get("last_seen") or "")), reverse=True)
    severity = str(latest[0].get("severity") or "info") if latest else "info"
    return {
        "status": "inspect" if latest else "ok",
        "open_count": len(unresolved),
        "critical_count": sum(1 for issue in unresolved if issue.get("severity") == "critical"),
        "warning_count": sum(1 for issue in unresolved if issue.get("severity") == "warning"),
        "latest": latest[0] if latest else {},
        "severity": severity,
    }


def collect_fleet_health(
    *,
    local_snapshot: dict[str, Any],
    data_root: Path,
    pi: dict[str, str] | None = None,
    mac: dict[str, str] | None = None,
    heal: bool = False,
) -> dict[str, Any]:
    """Collect a compact fleet health verdict with optional safe service healing."""
    remote: dict[str, Any] = {}
    if pi:
        remote["pi"] = _ssh_service_health(pi, service_command="systemctl --user", service_name="trademl-node.service", heal=heal)
    if mac:
        remote["mac"] = _ssh_mac_health(mac, heal=heal)
    issues = collect_current_state_issues(local_snapshot)
    for name, payload in remote.items():
        if payload.get("status") != "online":
            issues.append(_issue(name, "critical", f"{name}_remote_unhealthy", str(payload.get("reason") or f"{name} remote check failed"), f"Inspect {name} remote service."))
    bucket = write_codex_issue_bucket(data_root=data_root, issues=issues)
    bucket_issues = list(bucket.get("issues") or read_codex_issue_bucket(data_root=data_root)["issues"])
    summary_snapshot = {**local_snapshot, "fleet_remote": remote}
    current_state = build_current_state_summary(summary_snapshot, issues=bucket_issues)
    return {"verdict": current_state["verdict"], "current_state": current_state, "remote": remote, "issue_bucket": bucket}


def _pi_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(snapshot.get("runtime") or {})
    collection = dict(snapshot.get("collection_status") or {})
    remote = dict((snapshot.get("fleet_remote") or {}).get("pi") or {})
    running = bool(runtime.get("running"))
    if not running and remote.get("status") == "online":
        running = True
    repair = int(collection.get("repair_remaining_units") or 0)
    status = "online" if running and repair == 0 else "degraded" if running else "offline"
    return {
        "status": status,
        "headline": "Online" if running else "Offline",
        "detail": f"PID {runtime.get('pid')}" if runtime.get("pid") else "Waiting for node",
        "latest_raw_date": snapshot.get("latest_raw_date"),
        "repair_remaining_units": repair,
    }


def _mac_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    program = dict((snapshot.get("health") or {}).get("research_program_summary") or {})
    experiment = dict(snapshot.get("experiment_summary") or {})
    launchd = dict(program.get("launchd") or {})
    remote = dict((snapshot.get("fleet_remote") or {}).get("mac") or {})
    if not program and remote.get("status") == "online":
        remote_research = dict(remote.get("research") or {})
        experiment_id = str(remote_research.get("current_experiment_id") or "remote check passed")
        research_running = str(remote_research.get("status") or "").upper() == "RUNNING"
        return {
            "status": "online",
            "headline": "Research running" if research_running else "LaunchAgent online",
            "detail": experiment_id,
            "launchd": {"loaded": True, "remote": True},
        }
    running = str(program.get("status") or "").upper() == "RUNNING"
    blocked = str(program.get("status") or "").upper() in {"INFRA_BLOCKED", "WAITING_FOR_DATA"}
    launchd_ok = not launchd or bool(launchd.get("loaded"))
    status = "online" if running and launchd_ok else "blocked" if blocked else "degraded" if program else "unknown"
    incumbent = dict(program.get("incumbent") or {})
    best = dict(program.get("best_candidate_summary") or {})
    headline = "Research running" if running else str(program.get("status") or "Unknown")
    if running and not incumbent and best:
        headline = "Research running, no incumbent yet"
    detail = str(program.get("current_experiment_id") or experiment.get("experiment_id") or "-")
    if running and not incumbent and best.get("best_decision_reason"):
        detail = str(best.get("best_decision_reason"))
    return {
        "status": status,
        "headline": headline,
        "detail": detail,
        "launchd": launchd,
    }


def _architecture_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    program = dict((snapshot.get("health") or {}).get("research_program_summary") or {})
    experiment = dict(snapshot.get("experiment_summary") or {})
    incumbent = dict(program.get("incumbent") or {})
    best = dict(program.get("best_candidate_summary") or {})
    if incumbent:
        state = "Incumbent active"
        status = "online"
    elif int(experiment.get("shortlist_count") or 0) > 0:
        state = "Candidate found"
        status = "pending"
    elif str(program.get("status") or "").upper() == "RUNNING" and best:
        state = "Research running, no promotable candidate yet"
        status = "pending"
    else:
        state = "No incumbent"
        status = "pending"
    best_advanced = dict(best.get("best_advanced") or {})
    return {
        "status": status,
        "state": state,
        "best_candidate": best.get("best_candidate") or experiment.get("best_candidate"),
        "primary_score": best.get("best_primary_score") or experiment.get("best_primary_score"),
        "best_advanced_score": best_advanced.get("best_primary_score"),
        "decision": best.get("best_decision") or experiment.get("best_decision") or "No GO yet",
        "reason": best.get("best_decision_reason") or experiment.get("best_decision_reason") or "-",
    }


def _profit_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    program = dict((snapshot.get("health") or {}).get("research_program_summary") or {})
    paper = dict(program.get("latest_paper_outputs") or program.get("paper_outputs") or {})
    shadow = dict(program.get("latest_shadow_paper_outputs") or {})
    source = paper if paper.get("status") == "written" else shadow
    if not source:
        return {"status": "pending", "headline": "No validated paper PnL yet", "detail": "-"}
    return {
        "status": "online" if not source.get("non_incumbent") else "pending",
        "headline": "Shadow paper ready" if source.get("non_incumbent") else "Paper ready",
        "detail": source.get("paper_orders_path") or source.get("shadow_orders_path") or "-",
        "date": source.get("date"),
        "non_incumbent": bool(source.get("non_incumbent")),
    }


def _rollup_verdict(statuses: list[str], issue_summary: dict[str, Any]) -> str:
    if issue_summary.get("critical_count") or "blocked" in statuses or "offline" in statuses:
        return "BLOCKED"
    if issue_summary.get("warning_count") or any(status in {"degraded", "unknown"} for status in statuses):
        return "DEGRADED"
    return "OK"


def _action_from_verdict(verdict: str, issue_summary: dict[str, Any]) -> str:
    if verdict == "OK":
        return "OK"
    if issue_summary.get("critical_count"):
        return "Blocked"
    return "Codex should inspect"


def _issue(source: str, severity: str, kind: str, message: str, suggested_action: str) -> dict[str, Any]:
    return {"source": source, "severity": severity, "kind": kind, "message": message, "suggested_codex_action": suggested_action}


def _normalize_issue(issue: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    identity = {
        "source": issue.get("source"),
        "kind": issue.get("kind"),
        "message": issue.get("message"),
    }
    issue_id = hashlib.sha1(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return {
        "id": issue_id,
        "status": "open",
        "source": str(issue.get("source") or "unknown"),
        "severity": str(issue.get("severity") or "warning"),
        "kind": str(issue.get("kind") or "unknown"),
        "message": str(issue.get("message") or ""),
        "suggested_codex_action": str(issue.get("suggested_codex_action") or "Inspect this issue."),
        "first_seen": now.isoformat(),
        "last_seen": now.isoformat(),
        "count": 1,
    }


def _ssh_service_health(target: dict[str, str], *, service_command: str, service_name: str, heal: bool) -> dict[str, Any]:
    command = f"{service_command} is-active {service_name}"
    result = _run_password_ssh(target, command)
    active = result.get("returncode") == 0 and str(result.get("stdout") or "").strip() == "active"
    healed = False
    if heal and not active:
        restart = _run_password_ssh(target, f"{service_command} restart {service_name}")
        healed = restart.get("returncode") == 0
        result = _run_password_ssh(target, command)
        active = result.get("returncode") == 0 and str(result.get("stdout") or "").strip() == "active"
    return {"status": "online" if active else "offline", "healed": healed, "raw": result, "reason": None if active else result.get("stderr") or result.get("stdout")}


def _ssh_mac_health(target: dict[str, str], *, heal: bool) -> dict[str, Any]:
    label = target.get("label") or "com.trademl.research.perpetual-macmini"
    service = f"gui/$(id -u)/{label}"
    command = f"launchctl print {service} >/dev/null && echo active"
    result = _run_password_ssh(target, command)
    active = result.get("returncode") == 0 and "active" in str(result.get("stdout") or "")
    healed = False
    if heal and not active:
        restart = _run_password_ssh(target, f"launchctl kickstart -k {service}")
        healed = restart.get("returncode") == 0
        result = _run_password_ssh(target, command)
        active = result.get("returncode") == 0 and "active" in str(result.get("stdout") or "")
    research = _ssh_mac_research_status(target) if active else {}
    return {
        "status": "online" if active else "offline",
        "healed": healed,
        "raw": result,
        "research": research,
        "reason": None if active else result.get("stderr") or result.get("stdout"),
    }


def _ssh_mac_research_status(target: dict[str, str]) -> dict[str, Any]:
    repo = shlex.quote(target.get("repo_path") or "/Users/openclaw/TradeML")
    data_root = shlex.quote(target.get("data_root") or "/Users/openclaw/atlas_mounts/nas")
    local_state = shlex.quote(target.get("local_state") or "/Users/openclaw/TradeML/control")
    env_file = shlex.quote(target.get("env_file") or ".env")
    program_id = shlex.quote(target.get("program_id") or "perpetual-macmini")
    command = (
        f"cd {repo} && .venv/bin/python -m trademl.cli research "
        f"--data-root {data_root} --local-state {local_state} --env-file {env_file} "
        f"status --program-id {program_id}"
    )
    result = _run_password_ssh(target, command)
    if result.get("returncode") != 0:
        return {"status": "unknown", "error": result.get("stderr") or result.get("stdout")}
    try:
        payload = json.loads(str(result.get("stdout") or "{}"))
    except json.JSONDecodeError as exc:
        return {"status": "unknown", "error": str(exc)}
    return {
        "status": payload.get("status"),
        "program_id": payload.get("program_id"),
        "current_experiment_id": payload.get("current_experiment_id"),
        "wait_reason": payload.get("wait_reason"),
    }


def _run_password_ssh(target: dict[str, str], command: str) -> dict[str, Any]:
    host = target.get("host")
    user = target.get("user")
    password_env = target.get("password_env")
    if not host or not user:
        return {"returncode": 2, "stdout": "", "stderr": "missing host/user"}
    env = dict(os.environ)
    prefix = ["ssh"]
    if password_env and env.get(password_env):
        env["SSHPASS"] = env[password_env]
        prefix = ["sshpass", "-e", "ssh"]
    ssh = [
        *prefix,
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=8",
        f"{user}@{host}",
        command,
    ]
    result = subprocess.run(ssh, capture_output=True, text=True, check=False, timeout=30, env=env)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
