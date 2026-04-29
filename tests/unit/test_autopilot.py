from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trademl.fleet.autopilot import (
    build_current_state_summary,
    collect_fleet_health,
    read_codex_issue_bucket,
    write_codex_issue_bucket,
)
from trademl.fleet.observability import build_fleet_observability


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

    assert payload["verdict"] == "OK"
    assert payload["pi"]["status"] == "online"
    assert payload["mac"]["status"] == "online"
    assert payload["architecture"]["state"] == "Research running, no promotable candidate yet"
    assert payload["architecture"]["status"] == "pending"
    assert payload["profit"]["headline"] == "Shadow paper ready"
    assert payload["profit"]["status"] == "pending"


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


def test_codex_issue_bucket_resolves_absent_current_issues(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    issue = {
        "source": "pi",
        "severity": "critical",
        "kind": "pi_remote_unhealthy",
        "message": "permission denied",
        "suggested_codex_action": "inspect",
    }

    write_codex_issue_bucket(
        data_root=data_root,
        issues=[issue],
        now=datetime(2026, 4, 28, 10, tzinfo=UTC),
    )
    payload = write_codex_issue_bucket(
        data_root=data_root,
        issues=[],
        now=datetime(2026, 4, 28, 11, tzinfo=UTC),
    )
    bucket = read_codex_issue_bucket(data_root=data_root)

    assert payload["summary"]["open_count"] == 0
    assert payload["summary"]["critical_count"] == 0
    assert bucket["issues"][0]["status"] == "resolved"
    assert bucket["issues"][0]["resolved_at"] == "2026-04-28T11:00:00+00:00"


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


def test_current_state_distinguishes_running_research_without_promotable_candidate() -> None:
    snapshot = {
        "runtime": {"running": True},
        "collection_status": {"repair_remaining_units": 0},
        "health": {
            "research_program_summary": {
                "status": "RUNNING",
                "current_experiment_id": "exp-a",
                "launchd": {"loaded": True},
                "best_candidate_summary": {
                    "best_candidate": "advanced",
                    "best_primary_score": 0.024,
                    "best_decision": "NO_GO",
                    "best_decision_reason": "ic_ok=False; years_positive=False",
                    "best_advanced": {"best_primary_score": 0.024},
                },
                "frontier_architecture": {"enabled": True, "active": True},
            }
        },
        "experiment_summary": {"shortlist_count": 0},
    }

    payload = build_current_state_summary(snapshot, issues=[])

    assert payload["mac"]["headline"] == "Research running, no incumbent yet"
    assert payload["verdict"] == "OK"
    assert payload["architecture"]["state"] == "Research running, no promotable candidate yet"
    assert payload["architecture"]["status"] == "pending"
    assert payload["architecture"]["best_advanced_score"] == 0.024
    assert payload["architecture"]["reason"] == "ic_ok=False; years_positive=False"


def test_current_state_treats_no_incumbent_and_no_pnl_as_pending_not_degraded() -> None:
    snapshot = {
        "runtime": {"running": True, "pid": 42},
        "collection_status": {"repair_remaining_units": 0},
        "health": {
            "research_program_summary": {
                "status": "RUNNING",
                "current_experiment_id": "exp-a",
                "launchd": {"loaded": True},
            }
        },
        "experiment_summary": {"shortlist_count": 0},
    }

    payload = build_current_state_summary(snapshot, issues=[])

    assert payload["verdict"] == "OK"
    assert payload["action"] == "OK"
    assert payload["architecture"]["state"] == "No incumbent"
    assert payload["architecture"]["status"] == "pending"
    assert payload["profit"]["headline"] == "No validated paper PnL yet"
    assert payload["profit"]["status"] == "pending"


def test_observability_snapshot_reports_vendor_underutilization_and_pending_pnl(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    snapshot = {
        "runtime": {"running": True, "pid": 42, "heartbeat_at": "2026-04-29T12:00:00+00:00"},
        "latest_raw_date": "2026-04-29",
        "latest_curated_date": "2026-04-29",
        "planner_summary": {"progress": {"supplemental_research": {"remaining_units": 10}}},
        "budget_summary": {
            "rows": [
                {
                    "vendor": "alpaca",
                    "state": "available",
                    "rpm_limit": 200,
                    "rpm_used": 0,
                    "day_remaining": 1000,
                    "minute_available": True,
                    "day_available": True,
                    "outbound_requests_60s": 0,
                }
            ]
        },
        "vendor_throughput": {"rows": [{"vendor": "alpaca", "state": "available", "rows_per_min": 0.0}]},
        "scheduler_decisions": {
            "window_minutes": 15,
            "rows": [{"vendor": "alpaca", "dataset": "equities_minute", "decision": "no_task", "count": 2}],
        },
        "archive_write_telemetry": {
            "rows": [{"output_name": "ticker_news", "failures": 0, "schema_mismatches": 0}]
        },
        "health": {
            "research_program_summary": {
                "status": "RUNNING",
                "current_experiment_id": "exp-a",
                "latest_shadow_paper_outputs": {
                    "status": "written",
                    "date": "2026-04-29",
                    "shadow_orders_path": "/tmp/shadow_orders.parquet",
                    "non_incumbent": True,
                },
            }
        },
    }

    payload = build_fleet_observability(
        snapshot=snapshot,
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    assert payload["vendors"]["underutilized_count"] == 1
    assert payload["scheduler"]["totals"]["no_task"] == 2
    assert payload["research"]["status"] == "RUNNING"
    assert payload["paper_pnl"]["status"] == "pending_labels"
    assert payload["verdict"] == "DEGRADED"
    assert payload["issues"][0]["kind"] == "vendor_underutilized"


def test_current_state_uses_observability_top_cards_without_degrading_for_pending_pnl(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    observability = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
            "latest_raw_date": "2026-04-29",
            "latest_curated_date": "2026-04-29",
            "health": {
                "research_program_summary": {
                    "status": "RUNNING",
                    "latest_shadow_paper_outputs": {"status": "written", "non_incumbent": True},
                }
            },
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, tzinfo=UTC),
    )

    payload = build_current_state_summary(
        {
            "runtime": {"running": True},
            "collection_status": {"repair_remaining_units": 0},
            "health": {"research_program_summary": {"status": "RUNNING"}},
            "observability": observability,
        },
        issues=[],
    )

    assert payload["systems"]["pi"]["status"] == "online"
    assert payload["data"]["headline"] == "Collecting"
    assert payload["research"]["headline"] == "Research running"
    assert payload["paper"]["headline"] == "Paper labels pending"
    assert payload["verdict"] == "OK"
