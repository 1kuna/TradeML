"""Research alert persistence and SMTP notification helpers."""

from __future__ import annotations

import os
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable

from trademl.experiments import _atomic_write_json

DEFAULT_ALERT_POLICY = {
    "email_enabled": False,
    "write_files": True,
    "cadence_hours": 168,
}


def write_research_alerts(
    *,
    program_id: str,
    alerts: list[dict[str, Any]],
    local_state: Path,
    data_root: Path,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist research alerts and optionally send an SMTP notification."""
    merged_policy = {**DEFAULT_ALERT_POLICY, **dict(policy or {})}
    if not alerts:
        return {"status": "skipped", "reason": "no alerts"}
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    local_root = _local_research_root(local_state=local_state) / "alerts"
    local_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "program_id": program_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "alerts": alerts,
    }
    json_path = local_root / f"{stamp}.json"
    md_path = local_root / f"{stamp}.md"
    if bool(merged_policy.get("write_files", True)):
        _atomic_write_json(json_path, payload)
        md_path.write_text(_render_alert_markdown(payload), encoding="utf-8")
        shared_root = _shared_research_root(data_root=data_root) / "alerts"
        shared_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(shared_root / f"{stamp}.json", payload)
        (shared_root / f"{stamp}.md").write_text(
            _render_alert_markdown(payload), encoding="utf-8"
        )
    email = {"status": "skipped", "reason": "email disabled"}
    if bool(merged_policy.get("email_enabled", False)):
        email = send_research_alert_email(program_id=program_id, alerts=alerts)
    return {
        "status": "written",
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "email": email,
    }


def send_research_alert_email(
    *,
    program_id: str,
    alerts: list[dict[str, Any]],
    smtp_factory: Callable[[str, int], Any] | None = None,
) -> dict[str, Any]:
    """Send research alerts through SMTP when all required env vars are configured."""
    host = os.getenv("TRADEML_SMTP_HOST")
    to_addr = os.getenv("TRADEML_ALERT_EMAIL_TO")
    from_addr = os.getenv("TRADEML_ALERT_EMAIL_FROM")
    if not host or not to_addr or not from_addr:
        return {"status": "skipped", "reason": "smtp env not configured"}
    port = int(os.getenv("TRADEML_SMTP_PORT") or "25")
    message = EmailMessage()
    message["Subject"] = f"TradeML research alert: {program_id}"
    message["To"] = to_addr
    message["From"] = from_addr
    message.set_content(
        _render_alert_markdown({"program_id": program_id, "alerts": alerts})
    )
    factory = smtp_factory or smtplib.SMTP
    with factory(host, port) as smtp:
        if str(os.getenv("TRADEML_SMTP_STARTTLS", "")).lower() in {"1", "true", "yes"}:
            smtp.starttls()
        username = os.getenv("TRADEML_SMTP_USERNAME")
        password = os.getenv("TRADEML_SMTP_PASSWORD")
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)
    return {"status": "sent", "host": host, "port": port, "to": to_addr}


def _render_alert_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# Research alerts: {payload.get('program_id')}", ""]
    for alert in payload.get("alerts", []) or []:
        lines.append(
            f"- [{alert.get('severity', 'info')}] {alert.get('kind')}: {alert.get('message')}"
        )
    return "\n".join(lines) + "\n"


def _local_research_root(*, local_state: Path) -> Path:
    return local_state / "research"


def _shared_research_root(*, data_root: Path) -> Path:
    return data_root / "control" / "cluster" / "state" / "research"
