from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trademl.fleet.autopilot import (
    build_current_state_summary,
    collect_current_state_issues,
    collect_fleet_health,
    read_codex_issue_bucket,
    write_codex_issue_bucket,
)
from trademl.fleet.audit import run_fleet_audit
from trademl.fleet.observability import build_fleet_observability
from trademl.fleet.watchdog import run_fleet_watchdog_once


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


def test_fleet_audit_resolves_stale_data_quality_issues_after_current_ok(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    stale_issue = {
        "source": "data",
        "severity": "warning",
        "kind": "data_quality_failure",
        "message": "stock_trades quality check is source_unavailable: no readable parquet files found at expected source paths",
        "suggested_codex_action": "inspect",
    }
    write_codex_issue_bucket(
        data_root=data_root,
        issues=[stale_issue],
        now=datetime(2026, 4, 28, 10, tzinfo=UTC),
    )

    monkeypatch.setattr(
        "trademl.fleet.audit.run_data_quality_audit",
        lambda **kwargs: {  # noqa: ARG005
            "rows": [{"dataset": "stock_trades", "verdict": "OK", "status": "ok"}],
            "summary": {"ok": 1, "warning": 0, "critical": 0, "info": 0},
        },
    )
    monkeypatch.setattr(
        "trademl.fleet.audit.collect_fleet_health",
        lambda **kwargs: {  # noqa: ARG005
            "current_state": {"systems": {}},
            "current_issues": [],
            "issue_bucket": {"issues": [stale_issue]},
            "observability": {"issues": [], "paper_pnl": {}},
        },
    )
    monkeypatch.setattr(
        "trademl.fleet.audit._safe_research_audit",
        lambda **kwargs: {"verdict": "OK", "status": "RUNNING", "issues": []},  # noqa: ARG005
    )

    payload = run_fleet_audit(
        local_snapshot={},
        data_root=data_root,
        repo_root=tmp_path,
        local_state=tmp_path / "control",
        targets_config_path=tmp_path / "targets.yml",
        python_executable="python",
        now=datetime(2026, 4, 28, 11, tzinfo=UTC),
    )
    bucket = read_codex_issue_bucket(data_root=data_root)

    assert bucket["summary"]["open_count"] == 0
    assert bucket["issues"][0]["status"] == "resolved"
    assert payload["suggested_codex_actions"] == []
    assert Path(payload["artifact_path"]).exists()
    assert Path(payload["codex_feed_path"]).exists()


def test_codex_issue_bucket_reports_missing_data_root_without_creating_it(tmp_path: Path) -> None:
    data_root = tmp_path / "missing-nas"

    payload = write_codex_issue_bucket(data_root=data_root, issues=[])

    assert not data_root.exists()
    assert payload["summary"]["critical_count"] == 1
    assert payload["issues"][0]["kind"] == "nas_unavailable"


def test_current_state_issues_accept_remote_pi_online_when_local_pid_missing() -> None:
    issues = collect_current_state_issues(
        {
            "runtime": {"running": False},
            "collection_status": {"repair_remaining_units": 0},
            "fleet_remote": {"pi": {"status": "online"}},
        }
    )

    assert [issue["kind"] for issue in issues] == []


def test_current_state_issues_flag_pi_schema_drift() -> None:
    issues = collect_current_state_issues(
        {
            "runtime": {"running": True},
            "collection_status": {"repair_remaining_units": 0},
            "deployed_version": {"commit": "abc123"},
            "db_schema": {
                "status": "missing_tables",
                "missing_tables": ["scheduler_decisions"],
            },
        }
    )

    assert [issue["kind"] for issue in issues] == ["pi_sqlite_schema_drift"]


def test_observability_uses_remote_mac_research_when_local_summary_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()

    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
            "health": {
                "research_program_summary": {
                    "status": "UNKNOWN",
                    "current_experiment_id": "stale-local",
                    "completed_runs_24h": 0,
                    "completed_runs_7d": 0,
                }
            },
            "fleet_remote": {
                "mac": {
                    "status": "online",
                    "research": {
                        "status": "RUNNING",
                        "current_experiment_id": "perpetual-macmini-p1-f188",
                        "completed_runs_24h": 12,
                        "completed_runs_7d": 42,
                        "wait_reason": None,
                        "latest_paper_account_smoke": {
                            "status": "ok",
                            "read_only": True,
                            "base_url": "https://paper-api.alpaca.markets/v2",
                        },
                        "latest_paper_pnl": {
                            "status": "available",
                            "net_return": 0.01,
                            "no_live_orders": True,
                        },
                        "feature_family_leaderboard": {
                            "entries": [
                                {
                                    "feature_version": "news_event_aggregates_v1",
                                    "readiness_status": "BLOCKED",
                                }
                            ]
                        },
                    },
                }
            },
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, tzinfo=UTC),
    )

    assert payload["research"]["status"] == "RUNNING"
    assert payload["research"]["current_experiment_id"] == "perpetual-macmini-p1-f188"
    assert payload["research"]["supervisor_running"] is True
    assert payload["research"]["completed_runs_24h"] == 12
    assert payload["research"]["completed_runs_7d"] == 42
    assert payload["research"]["paper_account_smoke"]["status"] == "ok"
    assert payload["research"]["feature_family_leaderboard"]["entries"][0]["feature_version"] == "news_event_aggregates_v1"
    assert payload["paper_pnl"]["status"] == "available"
    assert payload["paper_pnl"]["paper_account_smoke_status"] == "ok"


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
    calls: list[str] = []

    def fake_ssh(target, command):  # noqa: ANN001
        calls.append(command)
        if "research --data-root" in command:
            return {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "status": "RUNNING",
                        "program_id": "perpetual-macmini",
                        "current_experiment_id": "exp-remote",
                        "completed_runs_24h": 7,
                        "completed_runs_7d": 21,
                        "failed_runs_24h": 1,
                        "infra_blocked_runs_24h": 1,
                        "top_rejection_reasons": [{"reason": "not all yearly IC values are positive", "count": 3}],
                        "latest_paper_account_smoke": {"status": "ok", "read_only": True},
                    }
                ),
                "stderr": "",
            }
        return {"returncode": 0, "stdout": "active\n", "stderr": ""}

    monkeypatch.setattr("trademl.fleet.autopilot._run_password_ssh", fake_ssh)

    payload = collect_fleet_health(
        local_snapshot={"runtime": {"running": True}, "collection_status": {"repair_remaining_units": 0}, "health": {}},
        data_root=data_root,
        mac={"host": "mac", "user": "openclaw", "password_env": "MAC_PASS"},
    )

    assert any("health --program-id" in call for call in calls)
    assert payload["current_state"]["mac"]["status"] == "online"
    assert payload["current_state"]["mac"]["headline"] == "Research running"
    assert payload["current_state"]["mac"]["detail"] == "exp-remote"
    assert payload["observability"]["research"]["paper_account_smoke"]["status"] == "ok"
    assert payload["observability"]["research"]["completed_runs_24h"] == 7
    assert payload["observability"]["research"]["completed_runs_7d"] == 21
    assert payload["observability"]["research"]["failed_runs_24h"] == 1
    assert payload["observability"]["research"]["infra_blocked_runs_24h"] == 1
    assert payload["observability"]["research"]["top_rejection_reasons"][0]["count"] == 3


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


def test_observability_uses_remote_pi_dataset_level_activity(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": False},
            "fleet_remote": {
                "pi": {
                    "status": "online",
                    "node_state": {
                        "node_observability": {
                            "budget_summary": {
                                "rows": [
                                    {
                                        "vendor": "alpaca",
                                        "state": "available",
                                        "rpm_limit": 200,
                                        "rpm_used": 10,
                                        "day_remaining": 1000,
                                        "minute_available": True,
                                        "day_available": True,
                                    }
                                ]
                            },
                            "scheduler_decisions": {
                                "window_minutes": 15,
                                "rows": [
                                    {"vendor": "alpaca", "dataset": "stock_trades", "decision": "claimed", "count": 30},
                                    {"vendor": "finnhub", "dataset": "company_news", "decision": "claimed", "count": 15},
                                ],
                            },
                            "archive_write_telemetry": {
                                "window_minutes": 60,
                                "rows": [
                                    {
                                        "output_name": "ticker_news",
                                        "status": "success",
                                        "writes": 8,
                                        "rows_in": 1200,
                                        "rows_written": 1200,
                                        "latest_at": "2026-04-29T12:00:00+00:00",
                                    }
                                ],
                            },
                            "lane_health": {
                                "rows": [
                                    {"vendor": "alpaca", "dataset": "stock_trades", "state": "HEALTHY"},
                                    {"vendor": "finnhub", "dataset": "company_news", "state": "HEALTHY"},
                                ]
                            },
                        }
                    },
                }
            },
            "budget_summary": {"rows": []},
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    rows = {(row["vendor"], row["dataset"]): row for row in payload["vendors"]["rows"]}
    assert rows[("alpaca", "stock_trades")]["utilization_verdict"] == "active"
    assert rows[("alpaca", "stock_trades")]["actual_requests_per_minute"] == 2.0
    assert rows[("finnhub", "company_news")]["rows_written_current_window"] == 1200
    assert payload["collection_saturation"]["summary"]["active"] == 2
    assert payload["vendors"]["underutilized_count"] == 0


def test_observability_treats_disabled_lane_as_blocked_not_underutilized(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
            "planner_summary": {"progress": {"supplemental_research": {"remaining_units": 10}}},
            "budget_summary": {
                "rows": [
                    {
                        "vendor": "alpha_vantage",
                        "state": "available",
                        "rpm_limit": 5,
                        "rpm_used": 0,
                        "day_remaining": 25,
                        "minute_available": True,
                        "day_available": True,
                        "outbound_requests_60s": 0,
                    }
                ]
            },
            "vendor_throughput": {"rows": [{"vendor": "alpha_vantage", "state": "available"}]},
            "lane_health": {"rows": [{"vendor": "alpha_vantage", "dataset": "news_sentiment", "state": "DISABLED"}]},
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    row = payload["vendors"]["rows"][0]
    assert row["current_blocker"] == "disabled"
    assert row["utilization_verdict"] == "blocked"
    assert payload["issues"] == []


def test_observability_writes_collection_saturation_audit(tmp_path: Path) -> None:
    from trademl.fleet.observability import write_fleet_observability

    data_root = tmp_path / "nas"
    data_root.mkdir()
    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
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
            "vendor_throughput": {"rows": [{"vendor": "alpaca", "state": "available"}]},
            "lane_health": {"rows": [{"vendor": "alpaca", "dataset": "equities_minute", "state": "HEALTHY"}]},
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )
    result = write_fleet_observability(data_root=data_root, payload=payload, now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC))

    audit_path = data_root / "control" / "cluster" / "state" / "autopilot" / "collection_saturation" / "latest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert str(audit_path) in result["written"]
    assert audit["summary"]["underutilized"] == 1
    assert audit["rows"][0]["vendor"] == "alpaca"
    assert audit["rows"][0]["intentionally_idle"] is False


def test_observability_reports_blocked_feature_source_readiness(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
            "health": {
                "research_program_summary": {
                    "status": "RUNNING",
                    "current_experiment_id": "exp-a",
                    "feature_family_leaderboard": {
                        "entries": [
                            {
                                "feature_version": "news_event_aggregates_v1",
                                "readiness_status": "BLOCKED",
                                "top_rejection_reason": "news_events has no usable source-backed feature coverage",
                                "source_coverage": {
                                    "news_events": {
                                        "sources": {
                                            "ticker_news": {"status": "missing"},
                                        }
                                    }
                                },
                            }
                        ]
                    },
                }
            },
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    kinds = {issue["kind"] for issue in payload["issues"]}
    assert {"feature_source_blocked", "feature_source_missing", "feature_source_path_missing"}.issubset(kinds)


def test_observability_does_not_degrade_for_known_unavailable_free_plan_source(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    source_root = data_root / "control" / "cluster" / "state" / "data" / "source_availability"
    source_root.mkdir(parents=True)
    (source_root / "latest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "fundamentals_tiingo": {
                        "state": "ENTITLEMENT_UNAVAILABLE",
                        "status": "empty",
                        "known_unavailable": True,
                        "actionable": False,
                        "rows": 0,
                        "reason": "free-plan source unavailable",
                    }
                },
                "state_counts": {"ENTITLEMENT_UNAVAILABLE": 1},
            }
        ),
        encoding="utf-8",
    )

    payload = build_fleet_observability(
        snapshot={
            "runtime": {"running": True},
            "health": {
                "research_program_summary": {
                    "status": "RUNNING",
                    "feature_family_leaderboard": {
                        "entries": [
                            {
                                "feature_version": "sec_filing_events_v1",
                                "readiness_status": "BLOCKED",
                                "source_coverage": {
                                    "fundamentals_sec": {
                                        "sources": {
                                            "fundamentals_tiingo": {
                                                "status": "empty",
                                            }
                                        }
                                    }
                                },
                            }
                        ]
                    },
                }
            },
        },
        data_root=data_root,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    kinds = {issue["kind"] for issue in payload["issues"]}
    assert "feature_source_zero_coverage" not in kinds
    assert "feature_source_blocked" not in kinds
    assert "source_availability_actionable" not in kinds


def test_fleet_health_merges_remote_pi_observability_before_building_snapshot(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    data_root.mkdir()
    observed = {}

    def fake_service_health(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "status": "online",
            "node_state": {
                "db_schema": {"status": "ok", "missing_tables": []},
                "deployed_version": {"commit": "abc123"},
                "node_observability": {
                    "scheduler_decisions": {
                        "window_minutes": 15,
                        "rows": [{"vendor": "alpaca", "dataset": "stock_trades", "decision": "claimed", "count": 3}],
                    },
                    "controller_decisions": {
                        "window_minutes": 15,
                        "rows": [
                            {
                                "vendor": "alpaca",
                                "dataset": "stock_trades",
                                "action": "scale_up",
                                "count": 1,
                                "target_width": 8,
                            }
                        ],
                    },
                    "lane_health": {"rows": [{"vendor": "alpaca", "dataset": "stock_trades", "state": "HEALTHY"}]},
                    "ingestion_ledger": {
                        "window_minutes": 60,
                        "rows": [{"vendor": "alpaca", "dataset": "stock_trades", "output_name": "alpaca_market_events", "status": "success", "events": 1}],
                    },
                    "data_quality_checks": {
                        "window_hours": 24,
                        "rows": [{"dataset": "stock_trades", "verdict": "OK", "status": "ok", "checks": 1}],
                    },
                },
            },
        }

    def fake_build(*, snapshot, **kwargs):  # noqa: ANN001, ANN003
        observed.update(snapshot)
        return {"verdict": "OK", "vendors": {"rows": [], "underutilized_count": 0}, "issues": []}

    monkeypatch.setattr("trademl.fleet.autopilot._ssh_service_health", fake_service_health)
    monkeypatch.setattr("trademl.fleet.autopilot.build_fleet_observability", fake_build)

    payload = collect_fleet_health(
        local_snapshot={
            "health": {"research_program_summary": {"status": "RUNNING", "current_experiment_id": "exp-a"}},
        },
        data_root=data_root,
        pi={"host": "pi", "user": "zach"},
        mac=None,
    )

    assert "verdict" in payload
    assert observed["scheduler_decisions"]["rows"][0]["dataset"] == "stock_trades"
    assert observed["controller_decisions"]["rows"][0]["target_width"] == 8
    assert observed["lane_health"]["rows"][0]["vendor"] == "alpaca"
    assert observed["ingestion_ledger"]["rows"][0]["dataset"] == "stock_trades"
    assert observed["data_quality_checks"]["rows"][0]["verdict"] == "OK"


def test_fleet_watchdog_writes_current_state_and_repeated_issue_alert(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    issue = {
        "source": "pi",
        "severity": "warning",
        "kind": "vendor_underutilized",
        "message": "alpaca idle",
        "count": 4,
        "suggested_codex_action": "inspect scheduler",
    }

    def fake_collect(**kwargs):  # noqa: ANN003
        assert kwargs["heal"] is True
        return {
            "verdict": "DEGRADED",
            "issue_bucket": {"issues": [issue]},
            "remote": {"pi": {"recovery": {"status": "not_needed"}}},
            "observability": {"paper_pnl": {"paper_account_smoke_status": "ok"}},
        }

    monkeypatch.setattr("trademl.fleet.watchdog.collect_fleet_health", fake_collect)

    payload = run_fleet_watchdog_once(
        local_snapshot={},
        data_root=data_root,
        heal=True,
        now=datetime(2026, 4, 29, 12, 1, tzinfo=UTC),
    )

    latest = data_root / "control" / "cluster" / "state" / "autopilot" / "watchdog" / "latest.json"
    assert latest.exists()
    assert payload["action"] == "Watch"
    assert payload["alerts"][0]["kind"] == "repeated_vendor_underutilized"
    assert payload["recovery_attempts"][0]["target"] == "pi"
    assert "password" not in json.dumps(payload).lower()


def test_fleet_watchdog_reports_write_failure_without_crashing(tmp_path: Path, monkeypatch) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("file where directory should be", encoding="utf-8")
    monkeypatch.setattr(
        "trademl.fleet.watchdog.collect_fleet_health",
        lambda **kwargs: {"verdict": "OK", "issue_bucket": {"issues": []}, "observability": {}},  # noqa: ARG005
    )

    payload = run_fleet_watchdog_once(local_snapshot={}, data_root=blocker)

    assert payload["verdict"] == "OK"
    assert payload["write"]["status"] == "write_failed"


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
