from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trademl.fleet.autopilot import (
    build_current_state_summary,
    collect_fleet_health,
    read_codex_issue_bucket,
    write_codex_issue_bucket,
)


def test_current_state_summary_rolls_up_system_architecture_profit_and_codex() -> None:
    snapshot = {
        "runtime": {"running": True, "pid": 42},
        "latest_raw_date": "2026-04-27",
        "collection_status": {"repair_remaining_units": 0},
        "health": {
            "research_program_summary": {
                "status": "RUNNING",
                "current_experiment_id": "exp-a",
                "launchd": {"loaded": True, "state": "running"},
                "best_candidate_summary": {"best_candidate": "advanced", "best_primary_score": 0.02, "best_decision": "NO_GO"},
                "latest_shadow_paper_outputs": {"status": "written", "date": "2026-04-24", "non_incumbent": True},
            }
        },
        "experiment_summary": {"shortlist_count": 0},
    }

    payload = build_current_state_summary(snapshot, issues=[])

    assert payload["verdict"] == "DEGRADED"
    assert payload["pi"]["status"] == "online"
    assert payload["mac"]["status"] == "online"
    assert payload["architecture"]["state"] == "No incumbent"
    assert payload["profit"]["headline"] == "Shadow paper ready"


def test_codex_issue_bucket_deduplicates_and_counts(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    issue = {"source": "pi", "severity": "warning", "kind": "repair_backlog", "message": "repair remains", "suggested_codex_action": "inspect"}

    first = write_codex_issue_bucket(data_root=data_root, issues=[issue], now=datetime(2026, 4, 28, tzinfo=UTC))
    second = write_codex_issue_bucket(data_root=data_root, issues=[issue], now=datetime(2026, 4, 29, tzinfo=UTC))
    bucket = read_codex_issue_bucket(data_root=data_root)

    assert len(first["written"]) == 1
    assert len(second["written"]) == 1
    assert len(bucket["issues"]) == 1
    assert bucket["issues"][0]["count"] == 2
    assert bucket["summary"]["open_count"] == 1


def test_codex_issue_bucket_reports_missing_data_root_without_creating_it(tmp_path: Path) -> None:
    data_root = tmp_path / "missing-nas"

    payload = write_codex_issue_bucket(data_root=data_root, issues=[])

    assert not data_root.exists()
    assert payload["summary"]["critical_count"] == 1
    assert payload["issues"][0]["kind"] == "nas_unavailable"


def test_fleet_health_records_remote_failures_without_healing_active_services(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    snapshot = {
        "runtime": {"running": True, "pid": 42},
        "collection_status": {"repair_remaining_units": 0},
        "health": {"research_program_summary": {"status": "RUNNING", "launchd": {"loaded": True}}},
        "experiment_summary": {},
    }
    calls: list[str] = []

    def fake_ssh(target, command):  # noqa: ANN001
        calls.append(command)
        return {"returncode": 1, "stdout": "", "stderr": "offline"}

    monkeypatch.setattr("trademl.fleet.autopilot._run_password_ssh", fake_ssh)

    payload = collect_fleet_health(
        local_snapshot=snapshot,
        data_root=data_root,
        pi={"host": "pi", "user": "zach", "password_env": "PI_PASS"},
        mac={"host": "mac", "user": "openclaw", "password_env": "MAC_PASS"},
        heal=False,
    )

    assert payload["verdict"] == "BLOCKED"
    assert any("is-active" in call for call in calls)
    assert not any("restart" in call for call in calls)
    assert (data_root / "control" / "cluster" / "state" / "autopilot" / "issues").exists()


def test_fleet_health_uses_remote_mac_status_when_local_program_state_is_absent(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()

    def fake_ssh(target, command):  # noqa: ANN001
        if "research --data-root" in command:
            return {
                "returncode": 0,
                "stdout": '{"status": "RUNNING", "program_id": "perpetual-macmini", "current_experiment_id": "exp-remote"}',
                "stderr": "",
            }
        return {"returncode": 0, "stdout": "active\n", "stderr": ""}

    monkeypatch.setattr("trademl.fleet.autopilot._run_password_ssh", fake_ssh)

    payload = collect_fleet_health(
        local_snapshot={"runtime": {"running": True}, "collection_status": {"repair_remaining_units": 0}, "health": {}},
        data_root=data_root,
        mac={"host": "mac", "user": "openclaw", "password_env": "MAC_PASS"},
    )

    assert payload["current_state"]["mac"]["status"] == "online"
    assert payload["current_state"]["mac"]["headline"] == "Research running"
    assert payload["current_state"]["mac"]["detail"] == "exp-remote"
