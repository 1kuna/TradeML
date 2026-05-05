from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trademl.fleet.audit import build_codex_feed
from trademl.research import record_research_progression_event
from trademl.research_audit import run_research_progression_audit


def test_research_audit_flags_no_progress_without_degrading_missing_incumbent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "nas"
    local_state = tmp_path / "state"
    data_root.mkdir()
    local_state.mkdir()
    (local_state / "research" / "perpetual-macmini.json").parent.mkdir(parents=True)
    (local_state / "research" / "perpetual-macmini.json").write_text(
        json.dumps({"status": "RUNNING", "current_experiment_id": "exp-a"}),
        encoding="utf-8",
    )
    latest_finished = datetime(2026, 5, 5, 8, tzinfo=UTC).isoformat()

    monkeypatch.setattr(
        "trademl.research_audit.research_health",
        lambda **kwargs: {
            "program_id": "perpetual-macmini",
            "status": "RUNNING",
            "current_experiment_id": "exp-a",
            "completed_runs_24h": 0,
            "research_throughput": {"latest_finished_at": latest_finished},
            "top_rejection_reasons": [],
            "latest_paper_account_smoke": {},
        },
    )
    monkeypatch.setattr(
        "trademl.research_audit.read_research_program_state",
        lambda **kwargs: {"status": "RUNNING", "current_experiment_id": "exp-a"},
    )
    monkeypatch.setattr(
        "trademl.research_audit.read_feature_family_leaderboard",
        lambda **kwargs: {"entries": []},
    )

    payload = run_research_progression_audit(
        program_id="perpetual-macmini",
        local_state=local_state,
        repo_root=tmp_path,
        data_root=data_root,
        targets_config_path=tmp_path / "node.yml",
        python_executable="python",
        now=datetime(2026, 5, 5, 11, tzinfo=UTC),
    )

    assert payload["verdict"] == "DEGRADED"
    assert payload["issues"][0]["kind"] == "research_no_completed_runs"
    assert payload["latest_finished_age_seconds"] == 10800.0
    assert "incumbent" not in json.dumps(payload["issues"]).lower()
    assert (data_root / "control" / "cluster" / "state" / "research" / "progression_audit" / "latest.json").exists()


def test_research_audit_flags_stale_paper_smoke(tmp_path: Path, monkeypatch) -> None:
    old_smoke = (datetime(2026, 5, 4, 10, tzinfo=UTC) - timedelta(hours=1)).isoformat()
    monkeypatch.setattr(
        "trademl.research_audit.research_health",
        lambda **kwargs: {
            "status": "RUNNING",
            "current_experiment_id": "exp-a",
            "research_throughput": {"latest_finished_at": datetime(2026, 5, 5, 10, tzinfo=UTC).isoformat()},
            "top_rejection_reasons": [],
            "latest_paper_account_smoke": {"status": "ok", "checked_at": old_smoke},
        },
    )
    monkeypatch.setattr("trademl.research_audit.read_research_program_state", lambda **kwargs: {})
    monkeypatch.setattr("trademl.research_audit.read_feature_family_leaderboard", lambda **kwargs: {"entries": []})

    payload = run_research_progression_audit(
        program_id="perpetual-macmini",
        local_state=tmp_path / "state",
        repo_root=tmp_path,
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "node.yml",
        python_executable="python",
        now=datetime(2026, 5, 5, 12, tzinfo=UTC),
    )

    assert any(issue["kind"] == "paper_smoke_stale" for issue in payload["issues"])


def test_research_audit_includes_durable_progression_events(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    record_research_progression_event(
        data_root=data_root,
        program_id="perpetual-macmini",
        event="decision_selected",
        payload={"action": "launch_family", "reason": "advanced-first"},
        now=datetime(2026, 5, 5, 10, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "trademl.research_audit.research_health",
        lambda **kwargs: {  # noqa: ARG005
            "status": "RUNNING",
            "current_experiment_id": "exp-a",
            "research_throughput": {"latest_finished_at": datetime(2026, 5, 5, 11, 59, tzinfo=UTC).isoformat()},
            "top_rejection_reasons": [],
        },
    )
    monkeypatch.setattr("trademl.research_audit.read_research_program_state", lambda **kwargs: {})
    monkeypatch.setattr("trademl.research_audit.read_feature_family_leaderboard", lambda **kwargs: {"entries": []})

    payload = run_research_progression_audit(
        program_id="perpetual-macmini",
        local_state=tmp_path / "state",
        repo_root=tmp_path,
        data_root=data_root,
        targets_config_path=tmp_path / "node.yml",
        python_executable="python",
        now=datetime(2026, 5, 5, 12, tzinfo=UTC),
    )

    assert payload["recorded_events"][0]["event"] == "decision_selected"
    assert payload["events"][0]["reason"] == "advanced-first"


def test_codex_feed_prioritizes_collection_before_cleanup() -> None:
    payload = build_codex_feed(
        issues=[
            {
                "source": "system",
                "severity": "warning",
                "kind": "cleanup_needed",
                "message": "old files",
                "suggested_codex_action": "clean",
            },
            {
                "source": "pi",
                "severity": "warning",
                "kind": "vendor_underutilized",
                "message": "alpaca idle",
                "suggested_codex_action": "inspect collection",
            },
        ],
        data_quality={"summary": {}},
        research={"verdict": "OK"},
        observability={"vendors": {"underutilized_count": 1}},
        now=datetime(2026, 5, 5, tzinfo=UTC),
    )

    assert payload["ranked_actions"][0]["kind"] == "vendor_underutilized"
    assert payload["ranked_actions"][0]["category"] == "blocked_collection_capacity"
