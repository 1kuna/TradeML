from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import trademl.research as research


def _program_spec(tmp_path: Path) -> Path:
    path = tmp_path / "perpetual.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "program_id": "perpetual-macmini",
                "target": "workstation-remote",
                "phase_order": [1, 2],
                "default_phase": 1,
                "budget_policy": {
                    "max_total_runs": 50,
                    "max_total_hours": 24,
                    "max_consecutive_non_improving_families": 3,
                    "max_repeat_rejection_reason": 3,
                    "max_low_novelty_families": 2,
                    "min_novelty_score": 0.25,
                    "max_infra_failures": 2,
                },
                "review_policy": {"cadence_hours": 12},
                "search_policy": {"exhaustion_mode": "wait_for_new_data"},
                "phase_policies": {
                    "1": {
                        "phase": 1,
                        "auto_unlock": True,
                        "allowed_dimensions": ["architecture_family", "feature_family", "data_family"],
                        "allowed_data_families": ["price_only", "price_plus_liquidity"],
                        "family_size_cap": 4,
                        "max_generations": 6,
                        "max_phase_runs": 40,
                        "unlock_gate": {"require_shortlist": True, "require_cost_positive": True},
                        "initial_matrix": {
                            "architecture_family": ["linear_baseline", "tree_challenger"],
                            "feature_family": ["price_core"],
                            "data_family": ["price_only", "price_plus_liquidity"],
                        },
                        "predictive_gate": {"min_rank_ic": 0.01, "require_go_decision": True},
                        "backtest_gate": {"min_net_return": 0.0, "require_cost_positive": True},
                    },
                    "2": {
                        "phase": 2,
                        "auto_unlock": False,
                        "allowed_dimensions": ["architecture_family", "feature_family", "data_family"],
                        "allowed_data_families": ["price_only"],
                        "family_size_cap": 3,
                        "max_generations": 4,
                        "max_phase_runs": 30,
                        "initial_matrix": {
                            "architecture_family": ["tree_challenger"],
                            "feature_family": ["price_short_horizon"],
                            "data_family": ["price_only"],
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_research_program_spec_normalizes_ssot_phase_aliases(tmp_path: Path) -> None:
    path = tmp_path / "perpetual_ssot.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "program_id": "perpetual-macmini",
                "ssot_phase_order": [1, 2],
                "default_ssot_phase": 1,
                "ssot_phase_policies": {
                    "1": {"ssot_phase": 1, "campaign_track": "baseline"},
                    "2": {"ssot_phase": 2, "campaign_track": "phase2_free_data"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    spec = research._load_research_program_spec(path)  # noqa: SLF001

    assert spec["phase_order"] == [1, 2]
    assert spec["ssot_phase_order"] == [1, 2]
    assert spec["default_phase"] == 1
    assert spec["default_ssot_phase"] == 1
    assert spec["phase_policies"]["2"]["campaign_track"] == "phase2_free_data"


def test_determine_program_transition_advances_phase_when_unlock_gate_met(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f001"
    state["best_candidate_summary"] = {"best_primary_score": 0.01}
    frontier = research._empty_frontier()  # noqa: SLF001
    summary = {
        "experiment_id": "perpetual-macmini-p1-f001",
        "shortlist_count": 1,
        "best_backtest_net_return": 0.12,
        "best_primary_score": 0.021,
        "best_candidate": "full",
        "best_decision": "GO",
        "top_gate_failures": [],
    }

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=frontier,
        experiment_summary=summary,
        proposal={},
    )

    assert decision["action"] == "advance_phase"
    assert decision["next_phase"] == 2
    assert decision["next_spec"]["phase"] == 2


def test_determine_program_transition_prefers_phase_unlock_over_phase_budget_stop(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f006"
    state["budgets"]["phase_family_counts"] = {"1": 6}
    state["best_candidate_summary"] = {"best_primary_score": 0.01}
    summary = {
        "experiment_id": "perpetual-macmini-p1-f006",
        "shortlist_count": 1,
        "best_backtest_net_return": 0.12,
        "best_primary_score": 0.03,
        "best_candidate": "full",
        "best_decision": "GO",
        "top_gate_failures": [],
    }

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=research._empty_frontier(),  # noqa: SLF001
        experiment_summary=summary,
        proposal={},
    )

    assert decision["action"] == "advance_phase"
    assert decision["next_phase"] == 2


def test_program_next_spec_respects_steering_preferences(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["steering"] = {
        **state["steering"],
        "prefer_architecture_families": ["tree_challenger"],
        "avoid_data_families": ["price_plus_liquidity"],
        "force_pivot": True,
    }
    phase_policy = research._phase_policy(spec=spec, phase=1)  # noqa: SLF001
    proposal = {
        "next_spec": {
            "experiment_id": "phase1-g1",
            "phase": 1,
            "matrix": {
                "architecture_family": ["linear_baseline", "tree_challenger"],
                "feature_family": ["price_core"],
                "data_family": ["price_only", "price_plus_liquidity"],
            },
            "proposal_policy": {"auto_launch_next_family": True},
            "supervision": {"auto_propose_next_family": True},
        },
        "rationale": ["pivot"],
    }

    next_spec = research._program_next_spec_from_proposal(  # noqa: SLF001
        spec=spec,
        state=state,
        phase_policy=phase_policy,
        proposal=proposal,
        frontier=research._empty_frontier(),  # noqa: SLF001
    )

    assert next_spec is not None
    assert next_spec["matrix"]["architecture_family"][0] == "linear_baseline"
    assert next_spec["matrix"]["data_family"] == ["price_only"]
    assert next_spec["supervision"]["auto_propose_next_family"] is False
    assert next_spec["proposal_policy"]["auto_launch_next_family"] is False


def test_determine_program_transition_waits_for_new_data_on_duplicate_low_novelty(tmp_path: Path, monkeypatch) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    phase_policy = research._phase_policy(spec=spec, phase=1)  # noqa: SLF001
    duplicate_spec = research._make_experiment_spec_from_phase_policy(  # noqa: SLF001
        spec=spec,
        program_state=state,
        phase_policy=phase_policy,
        experiment_id="perpetual-macmini-p1-f002",
        generation=1,
    )
    signature = research._family_signature(duplicate_spec)  # noqa: SLF001
    frontier = research._empty_frontier()  # noqa: SLF001
    frontier["tried_family_signatures"] = [signature]
    monkeypatch.setattr(research, "_expanded_search_spec", lambda **kwargs: None)

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=frontier,
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f001", "shortlist_count": 0, "top_gate_failures": []},
        proposal={"next_spec": duplicate_spec, "rationale": ["duplicate"]},
    )

    assert decision["action"] == "wait_for_data"
    assert "waiting for new data revision" in decision["reason"]


def test_determine_program_transition_waits_when_phase_budget_is_exhausted_in_perpetual_mode(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["budgets"]["phase_run_counts"] = {"1": 40}

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=research._empty_frontier(),  # noqa: SLF001
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f001", "shortlist_count": 0, "top_gate_failures": []},
        proposal={},
    )

    assert decision["action"] == "wait_for_data"
    assert "phase 1 run budget exhausted" in decision["reason"]


def test_determine_program_transition_broadens_search_before_waiting(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=research._empty_frontier(),  # noqa: SLF001
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f001", "shortlist_count": 0, "top_gate_failures": []},
        proposal={},
    )

    assert decision["action"] == "launch_family"
    assert "broadening the bounded search frontier" in decision["reason"]
    assert decision["next_spec"]["matrix"]["data_profile"] == ["phase1_default", "phase1_short_window", "phase1_long_window"]


def test_frontier_architecture_policy_unlocks_phase1_advanced_first(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    spec["frontier_architecture_policy"] = {
        "enabled": True,
        "allow_phase1_advanced": True,
        "trigger_min_completed_runs": 100,
        "advanced_first": True,
        "family_size_cap": 6,
        "sentinel_baseline_runs": 2,
        "max_advanced_failures_per_epoch": 12,
    }
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["budgets"]["runs_completed"] = 1186
    state["budgets"]["max_total_runs"] = 5000
    state["budgets"]["non_improving_families"] = 10
    frontier = research._empty_frontier()  # noqa: SLF001
    frontier["best_by_feature_family"] = {
        "price_liquidity": {"best_primary_score": 0.007},
        "price_core": {"best_primary_score": 0.004},
    }
    frontier["best_by_data_family"] = {
        "price_plus_liquidity": {"best_primary_score": 0.007},
        "price_only": {"best_primary_score": 0.005},
    }

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=frontier,
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f052", "shortlist_count": 0, "top_gate_failures": []},
        proposal={},
    )

    assert decision["action"] == "launch_family"
    assert "frontier architecture" in decision["reason"]
    matrix = decision["next_spec"]["matrix"]
    assert matrix["architecture_family"] == ["advanced_challenger", "tree_challenger", "linear_baseline"]
    assert matrix["feature_family"] == ["price_liquidity"]
    assert matrix["data_family"] == ["price_plus_liquidity"]
    assert decision["next_spec"]["proposal_policy"]["family_size_cap"] == 6


def test_frontier_architecture_policy_disabled_preserves_phase1_search(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    spec["frontier_architecture_policy"] = {"enabled": False, "allow_phase1_advanced": True}
    spec["budget_policy"]["max_total_runs"] = 5000
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["budgets"]["runs_completed"] = 100
    state["budgets"]["max_total_runs"] = 5000

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=research._empty_frontier(),  # noqa: SLF001
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f052", "shortlist_count": 0, "top_gate_failures": []},
        proposal={},
    )

    assert decision["action"] == "launch_family"
    assert "advanced_challenger" not in decision["next_spec"]["matrix"]["architecture_family"]


def test_frontier_architecture_policy_respects_advanced_failure_brake(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    spec["frontier_architecture_policy"] = {
        "enabled": True,
        "allow_phase1_advanced": True,
        "trigger_min_completed_runs": 100,
        "max_advanced_failures_per_epoch": 12,
    }
    spec["budget_policy"]["max_total_runs"] = 5000
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["budgets"]["runs_completed"] = 1186
    state["budgets"]["max_total_runs"] = 5000
    frontier = research._empty_frontier()  # noqa: SLF001
    frontier["lane_stats"]["architecture_family"]["advanced_challenger"] = 12

    decision = research._determine_program_transition(  # noqa: SLF001
        spec=spec,
        state=state,
        frontier=frontier,
        experiment_summary={"experiment_id": "perpetual-macmini-p1-f052", "shortlist_count": 0, "top_gate_failures": []},
        proposal={},
    )

    assert decision["action"] == "launch_family"
    assert "advanced_challenger" not in decision["next_spec"]["matrix"]["architecture_family"]


def test_program_family_preflight_blocks_missing_advanced_dependencies(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    target = SimpleNamespace(
        name="workstation-remote",
        kind="ssh",
        host="192.168.68.70",
        repo_root=tmp_path / "remote" / "TradeML",
        data_root=data_root,
    )
    monkeypatch.setattr(research, "resolve_training_target", lambda **kwargs: target)
    monkeypatch.setattr(research, "_resolve_default_report_date", lambda **kwargs: "2026-03-09")
    captured: dict[str, object] = {}

    def fake_training_preflight(**kwargs):  # noqa: ANN001
        captured["model_suite"] = kwargs.get("model_suite")
        return {
            "ok": False,
            "reason": "missing python packages for advanced: catboost, lightgbm, optuna",
            "dependencies": {"ok": False, "missing": ["catboost", "lightgbm", "optuna"]},
        }

    monkeypatch.setattr(research, "training_preflight", fake_training_preflight)

    payload = research._program_family_preflight(  # noqa: SLF001
        next_spec={
            "target": "workstation-remote",
            "phase": 1,
            "matrix": {"architecture_family": ["advanced_challenger", "tree_challenger", "linear_baseline"]},
        },
        repo_root=tmp_path / "repo",
        data_root=data_root,
        local_state=tmp_path / "local",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert captured["model_suite"] == "advanced"
    assert payload["ok"] is False
    assert payload["status"] == "INFRA_BLOCKED"
    assert payload["dependencies"]["missing"] == ["catboost", "lightgbm", "optuna"]


def test_resume_if_data_changed_launches_pending_spec(tmp_path: Path, monkeypatch) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["status"] = "WAITING_FOR_DATA"
    state["last_seen_data_revision"] = "2026-03-09"
    state["pending_next_spec"] = {
        "experiment_id": "perpetual-macmini-p1-f002",
        "phase": 1,
        "matrix": {"architecture_family": ["linear_baseline"]},
        "proposal_policy": {"auto_launch_next_family": False},
        "supervision": {"auto_propose_next_family": False},
    }
    monkeypatch.setattr(research, "_current_data_revision", lambda **kwargs: "2026-03-10")

    launched_payload = {"current_experiment_id": "perpetual-macmini-p1-f002", "active_experiment_supervisor": {"pid": 999}}
    captured: dict[str, object] = {}

    def fake_launch_program_family(**kwargs):  # noqa: ANN001
        captured["next_spec"] = kwargs["next_spec"]
        return launched_payload

    monkeypatch.setattr(research, "_launch_program_family", fake_launch_program_family)

    resumed = research._resume_if_data_changed(  # noqa: SLF001
        spec=spec,
        state=state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        local_state=tmp_path / "local",
        env_path=tmp_path / ".env",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    assert resumed == launched_payload
    assert state["search_epoch"] == 1
    assert state["last_seen_data_revision"] == "2026-03-10"
    assert state["pending_next_spec"] is None
    assert captured["next_spec"]["search_epoch"] == 1


def test_current_data_revision_tracks_qc_and_latest_curated_partition(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "nas"
    qc_path = data_root / "data" / "qc" / "partition_status.parquet"
    curated_path = data_root / "data" / "curated" / "equities_ohlcv_adj" / "date=2026-04-24" / "data.parquet"
    qc_path.parent.mkdir(parents=True)
    curated_path.parent.mkdir(parents=True)
    qc_path.write_text("qc", encoding="utf-8")
    curated_path.write_text("curated", encoding="utf-8")
    monkeypatch.setattr(
        research,
        "resolve_training_target",
        lambda **kwargs: SimpleNamespace(data_root=data_root),
    )
    monkeypatch.setattr(research, "_resolve_default_report_date", lambda **kwargs: "2026-04-22")

    revision = research._current_data_revision(  # noqa: SLF001
        spec={"target": "workstation-remote", "phase_order": [1]},
        state={"current_phase": 1},
        repo_root=tmp_path / "repo",
        data_root=data_root,
        local_state=tmp_path / "local",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert revision is not None
    assert "report:2026-04-22" in revision
    assert "partition_status.parquet" in revision
    assert "equities_ohlcv_adj:2026-04-24" in revision


def test_recover_stalled_active_experiment_relaunches_same_family_when_work_remains(tmp_path: Path, monkeypatch) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f032"
    state["current_experiment_spec_path"] = str(tmp_path / "specs" / "perpetual-macmini-p1-f032.yml")
    experiment_supervisor = {
        "status": "STOPPED",
        "experiment_id": "perpetual-macmini-p1-f032",
        "spec_path": str(tmp_path / "specs" / "perpetual-macmini-p1-f032.yml"),
        "last_error": "phase 1 training preflight failed: ssh command timed out for target=workstation-remote",
        "last_error_kind": "infra",
    }
    experiment_summary = {
        "experiment_id": "perpetual-macmini-p1-f032",
        "counts": {"COMPLETED": 68, "FAILED": 1},
        "runs": [
            {"run_id": "retry-me", "status": "FAILED", "failure_kind": "infra", "retry_count": 1},
        ],
    }
    observed: dict[str, object] = {}

    def fake_supervise_experiment(**kwargs):  # noqa: ANN001
        observed["spec_path"] = kwargs["spec_path"]
        observed["detach"] = kwargs["detach"]
        return {"experiment_id": "perpetual-macmini-p1-f032", "pid": 99991, "status": "RUNNING"}

    monkeypatch.setattr(research, "supervise_experiment", fake_supervise_experiment)

    relaunched = research._recover_stalled_active_experiment(  # noqa: SLF001
        state=state,
        experiment_supervisor=experiment_supervisor,
        experiment_summary=experiment_summary,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        local_state=tmp_path / "local",
        env_path=tmp_path / ".env",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    assert relaunched == {"experiment_id": "perpetual-macmini-p1-f032", "pid": 99991, "status": "RUNNING"}
    assert observed["detach"] is True
    assert observed["spec_path"] == Path(state["current_experiment_spec_path"])


def test_recover_stalled_active_experiment_skips_relaunch_when_family_is_idle(tmp_path: Path, monkeypatch) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f032"
    state["current_experiment_spec_path"] = str(tmp_path / "specs" / "perpetual-macmini-p1-f032.yml")
    experiment_supervisor = {
        "status": "STOPPED",
        "experiment_id": "perpetual-macmini-p1-f032",
        "spec_path": str(tmp_path / "specs" / "perpetual-macmini-p1-f032.yml"),
        "last_error_kind": "infra",
    }
    experiment_summary = {
        "experiment_id": "perpetual-macmini-p1-f032",
        "counts": {"COMPLETED": 68, "FAILED": 4},
        "runs": [
            {"run_id": "a", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
            {"run_id": "b", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
            {"run_id": "c", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
            {"run_id": "d", "status": "FAILED", "failure_kind": "model", "retry_count": 0},
        ],
    }

    def unexpected_supervise(**kwargs):  # noqa: ANN001
        raise AssertionError("stalled family recovery should not relaunch an idle family")

    monkeypatch.setattr(research, "supervise_experiment", unexpected_supervise)

    relaunched = research._recover_stalled_active_experiment(  # noqa: SLF001
        state=state,
        experiment_supervisor=experiment_supervisor,
        experiment_summary=experiment_summary,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        local_state=tmp_path / "local",
        env_path=tmp_path / ".env",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    assert relaunched is None


def test_ensure_program_state_refreshes_budget_caps_from_latest_spec(tmp_path: Path) -> None:
    program_path = _program_spec(tmp_path)
    spec = research._load_research_program_spec(program_path)  # noqa: SLF001
    initial = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=30)  # noqa: SLF001
    initial["budgets"]["runs_completed"] = 138
    initial["budgets"]["max_total_runs"] = 250
    initial["budgets"]["max_total_hours"] = 168
    research._write_program_state(local_state=tmp_path, program_id="perpetual-macmini", payload=initial)  # noqa: SLF001

    spec["budget_policy"]["max_total_runs"] = 5000
    spec["budget_policy"]["max_total_hours"] = 720
    refreshed = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=45)  # noqa: SLF001

    merged = research._ensure_program_state(  # noqa: SLF001
        local_state=tmp_path,
        program_id="perpetual-macmini",
        payload=refreshed,
    )

    assert merged["budgets"]["runs_completed"] == 138
    assert merged["budgets"]["max_total_runs"] == 5000
    assert merged["budgets"]["max_total_hours"] == 720
    assert merged["poll_seconds"] == 45



def test_write_review_packet_writes_json_and_markdown(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    program_path = _program_spec(tmp_path)
    spec = research._load_research_program_spec(program_path)  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f001"
    state["best_candidate_summary"] = {"best_candidate": "full", "best_primary_score": 0.02}
    state["last_transition"] = {"reason": "pivot to tree challenger"}
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=state)  # noqa: SLF001

    experiment_root = local_state / "experiments" / "perpetual-macmini-p1-f001"
    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "perpetual-macmini-p1-f001",
                "best_candidate": "full",
                "best_primary_score": 0.02,
                "best_backtest_net_return": 0.11,
                "best_decision": "GO",
                "top_gate_failures": [["assessment.decision != GO", 2]],
            }
        ),
        encoding="utf-8",
    )

    packet = research.write_research_review_packet(
        program_id="perpetual-macmini",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert Path(packet["json_path"]).exists()
    assert Path(packet["markdown_path"]).exists()
    payload = json.loads(Path(packet["json_path"]).read_text(encoding="utf-8"))
    assert payload["program_id"] == "perpetual-macmini"
    assert payload["best_candidate_summary"]["best_candidate"] == "full"


def test_write_review_packet_skips_compare_without_completed_runs(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    program_path = _program_spec(tmp_path)
    spec = research._load_research_program_spec(program_path)  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=30)  # noqa: SLF001
    state["current_experiment_id"] = "perpetual-macmini-p1-f001"
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=state)  # noqa: SLF001

    experiment_root = local_state / "experiments" / "perpetual-macmini-p1-f001"
    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "perpetual-macmini-p1-f001",
                "counts": {"PLANNED": 24},
                "top_gate_failures": [],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_compare(**kwargs):  # noqa: ANN001
        raise AssertionError("compare_experiment should not run without completed runs")

    monkeypatch.setattr(research, "compare_experiment", unexpected_compare)

    packet = research.write_research_review_packet(
        program_id="perpetual-macmini",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert Path(packet["json_path"]).exists()
    payload = json.loads(Path(packet["json_path"]).read_text(encoding="utf-8"))
    assert payload["comparison_best"] is None


def test_update_frontier_memory_records_lane_coverage_and_best_scores(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_phase"] = 1
    state["current_experiment_id"] = "perpetual-macmini-p1-f001"
    state["current_family_signature"] = "family-sig-001"
    summary = {
        "experiment_id": "perpetual-macmini-p1-f001",
        "best_candidate": "full",
        "best_primary_score": 0.031,
        "best_decision": "GO",
        "shortlist_count": 1,
        "run_count": 2,
        "top_gate_failures": [["rank_ic<0.01", 1]],
        "runs": [
            {
                "experiment_id": "perpetual-macmini-p1-f001",
                "run_id": "run-a",
                "model_suite": "full",
                "matrix_values": {
                    "architecture_family": "tree_challenger",
                    "feature_family": "price_core",
                    "data_family": "price_only",
                },
                "report_preview": {"lightgbm_mean_rank_ic": 0.031},
                "evaluation_stage": "SHORTLISTED",
                "shortlisted": True,
            },
            {
                "experiment_id": "perpetual-macmini-p1-f001",
                "run_id": "run-b",
                "model_suite": "ridge_only",
                "matrix_values": {
                    "architecture_family": "linear_baseline",
                    "feature_family": "price_core",
                    "data_family": "price_plus_liquidity",
                },
                "report_preview": {"ridge_mean_rank_ic": 0.014},
                "evaluation_stage": "REJECTED_BACKTEST",
                "shortlisted": False,
            },
        ],
    }

    frontier = research._update_frontier_memory(state=state, experiment_summary=summary)  # noqa: SLF001

    assert frontier["tried_family_signatures"] == ["family-sig-001"]
    assert frontier["lane_stats"]["architecture_family"]["tree_challenger"] == 1
    assert frontier["lane_stats"]["architecture_family"]["linear_baseline"] == 1
    assert frontier["best_by_architecture_family"]["tree_challenger"]["best_primary_score"] == 0.031
    assert frontier["best_by_data_family"]["price_plus_liquidity"]["best_primary_score"] == 0.014
    assert frontier["best_by_phase"]["1"]["best_primary_score"] == 0.031


def test_update_frontier_memory_preserves_family_counter_beyond_history_cap(tmp_path: Path) -> None:
    spec = research._load_research_program_spec(_program_spec(tmp_path))  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=_program_spec(tmp_path), poll_seconds=30)  # noqa: SLF001
    state["current_phase"] = 1
    state["budgets"]["families_started"] = 50
    frontier = {
        **research._empty_frontier(),  # noqa: SLF001
        "family_history": [{"experiment_id": f"old-p1-f{i:03d}"} for i in range(1, 51)],
    }
    state["frontier"] = frontier
    state["current_experiment_id"] = "perpetual-macmini-p1-f051"
    state["current_family_signature"] = "family-sig-051"
    summary = {
        "experiment_id": "perpetual-macmini-p1-f051",
        "best_candidate": "full",
        "best_primary_score": 0.02,
        "best_decision": "NO_GO",
        "shortlist_count": 0,
        "run_count": 1,
        "top_gate_failures": [],
        "runs": [],
    }

    updated = research._update_frontier_memory(state=state, experiment_summary=summary)  # noqa: SLF001

    assert updated["family_history"][-1]["experiment_id"] == "perpetual-macmini-p1-f051"
    assert state["budgets"]["families_started"] == 51
    assert research._next_experiment_id(program_state=state, spec=spec, phase=1) == "perpetual-macmini-p1-f052"  # noqa: SLF001


def test_steer_research_program_persists_manual_overrides(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    program_path = _program_spec(tmp_path)
    spec = research._load_research_program_spec(program_path)  # noqa: SLF001
    state = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=30)  # noqa: SLF001
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=state)  # noqa: SLF001

    updated = research.steer_research_program(
        local_state=local_state,
        program_id="perpetual-macmini",
        prefer_architecture_families=["tree_challenger"],
        avoid_data_families=["price_plus_liquidity"],
        freeze_phase=1,
        force_pivot=True,
        exploration_breadth="high",
    )

    assert updated["steering"]["prefer_architecture_families"] == ["tree_challenger"]
    assert updated["steering"]["avoid_data_families"] == ["price_plus_liquidity"]
    assert updated["steering"]["freeze_phase"] == 1
    assert updated["steering"]["force_pivot"] is True
    assert updated["steering"]["exploration_breadth"] == "high"


def test_write_program_state_is_atomic(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    program_id = "perpetual-macmini"
    state_path = local_state / "research_programs" / program_id / "program_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")

    original_write_text = Path.write_text

    def flaky_write_text(self: Path, data: str, *args, **kwargs):  # noqa: ANN001
        if self.name.startswith(f"{state_path.name}.tmp-"):
            original_write_text(self, "{", *args, **kwargs)
            raise RuntimeError("simulated mid-write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    with pytest.raises(RuntimeError, match="mid-write failure"):
        research._write_program_state(  # noqa: SLF001
            local_state=local_state,
            program_id=program_id,
            payload={"status": "FAILED"},
        )

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"status": "RUNNING"}
    assert list(state_path.parent.glob(f"{state_path.name}.tmp-*")) == []


def test_spawn_program_supervisor_process_writes_bootstrap_state_before_launch(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    program_path = _program_spec(tmp_path)

    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 8765

    def fake_popen(command, cwd, stdout, stderr, start_new_session):  # noqa: ANN001
        state = research.read_research_program_state(local_state=local_state, program_id="perpetual-macmini")
        observed["state_before_spawn"] = state
        return FakeProcess()

    monkeypatch.setattr(research.subprocess, "Popen", fake_popen)

    payload = research._spawn_program_supervisor_process(  # noqa: SLF001
        program_id="perpetual-macmini",
        program_path=program_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    state_before_spawn = observed["state_before_spawn"]
    assert isinstance(state_before_spawn, dict)
    assert state_before_spawn["status"] == "STARTING"
    assert state_before_spawn["pid"] is None
    assert payload["pid"] == 8765
    assert payload["status"] == "RUNNING"


def test_spawn_program_supervisor_process_preserves_existing_frontier(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    program_path = _program_spec(tmp_path)
    existing = {
        "program_id": "perpetual-macmini",
        "frontier": {"tried_family_signatures": ["abc"]},
        "current_phase": 1,
        "status": "STOPPED",
    }
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=existing)  # noqa: SLF001

    class FakeProcess:
        pid = 2468

    monkeypatch.setattr(research.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    payload = research._spawn_program_supervisor_process(  # noqa: SLF001
        program_id="perpetual-macmini",
        program_path=program_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    stored = json.loads((local_state / "research_programs" / "perpetual-macmini" / "program_state.json").read_text(encoding="utf-8"))
    assert payload["pid"] == 2468
    assert stored["frontier"]["tried_family_signatures"] == ["abc"]
    assert stored["status"] == "RUNNING"
    assert stored["stop_requested"] is False


def test_spawn_program_supervisor_process_refreshes_budget_caps_from_spec(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    program_path = _program_spec(tmp_path)
    existing = {
        "program_id": "perpetual-macmini",
        "budgets": {"runs_completed": 138, "max_total_runs": 250, "max_total_hours": 168},
        "status": "STOPPED",
        "stop_requested": True,
    }
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=existing)  # noqa: SLF001

    class FakeProcess:
        pid = 1357

    monkeypatch.setattr(research.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    payload = research._spawn_program_supervisor_process(  # noqa: SLF001
        program_id="perpetual-macmini",
        program_path=program_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        python_executable="/usr/bin/python3",
        poll_seconds=45,
    )

    assert payload["budgets"]["runs_completed"] == 138
    assert payload["budgets"]["max_total_runs"] == 50
    assert payload["budgets"]["max_total_hours"] == 24
    assert payload["poll_seconds"] == 45
    assert payload["stop_requested"] is False


def test_spawn_program_supervisor_process_marks_state_stopped_when_popen_raises(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    program_path = _program_spec(tmp_path)

    monkeypatch.setattr(research.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))

    with pytest.raises(OSError, match="spawn failed"):
        research._spawn_program_supervisor_process(  # noqa: SLF001
            program_id="perpetual-macmini",
            program_path=program_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            python_executable="/usr/bin/python3",
            poll_seconds=30,
        )

    stored = research.read_research_program_state(local_state=local_state, program_id="perpetual-macmini")
    assert stored["status"] == "STOPPED"
    assert stored["last_error"] == "spawn failed"


def test_read_research_program_state_marks_dead_pid_stopped(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    state_path = local_state / "research_programs" / "perpetual-macmini" / "program_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"pid": 8765, "status": "STOPPING"}), encoding="utf-8")
    monkeypatch.setattr(research, "_is_local_process_running", lambda pid: False)

    payload = research.read_research_program_state(local_state=local_state, program_id="perpetual-macmini")

    assert payload["status"] == "STOPPED"


def test_supervise_research_program_clears_prior_stop_markers(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    program_path = _program_spec(tmp_path)
    spec = research._load_research_program_spec(program_path)  # noqa: SLF001
    stopped_state = research._initial_program_state(spec=spec, program_path=program_path, poll_seconds=30)  # noqa: SLF001
    stopped_state["status"] = "STOPPED"
    stopped_state["stop_reason"] = "old reason"
    stopped_state["completed_at"] = "2026-04-13T00:00:00+00:00"
    research._write_program_state(local_state=local_state, program_id="perpetual-macmini", payload=stopped_state)  # noqa: SLF001
    monkeypatch.setattr(research, "_build_initial_phase_experiment_spec", lambda **kwargs: None)

    result = research.supervise_research_program(
        program_path=program_path,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        local_state=local_state,
        env_path=tmp_path / ".env",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=1,
    )

    assert result["status"] == "STOPPED"
    assert result["stop_reason"] == "no_initial_phase_spec"


def test_launch_program_family_blocks_when_research_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    next_spec = {
        "experiment_id": "perpetual-macmini-p1-f001",
        "phase": 1,
        "ssot_phase": 1,
        "generation": 0,
        "campaign_track": "baseline",
        "matrix": {"architecture_family": ["linear_baseline"]},
    }

    monkeypatch.setattr(
        research,
        "_program_family_preflight",
        lambda **kwargs: {
            "ok": False,
            "status": "INFRA_BLOCKED",
            "reason": "missing qc parquet: /remote/data/qc/partition_status.parquet",
            "target_paths": {"data_root": "/remote"},
        },
    )
    monkeypatch.setattr(
        research,
        "supervise_experiment",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked preflight must not launch experiments")),
    )

    payload = research._launch_program_family(  # noqa: SLF001
        program_id="perpetual-macmini",
        program_state={},
        next_spec=next_spec,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        local_state=local_state,
        env_path=tmp_path / ".env",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    assert payload["status"] == "INFRA_BLOCKED"
    assert payload["wait_reason"].startswith("research preflight failed")
    assert payload["pending_next_spec"] == next_spec
    assert not (local_state / "research_programs" / "perpetual-macmini" / "specs").exists()


def test_sweep_stale_experiment_runs_reconciles_dead_remote_runtime(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    run_root = local_state / "experiments" / "exp-a" / "runs"
    run_root.mkdir(parents=True)
    manifest = {
        "experiment_id": "exp-a",
        "run_id": "run-a",
        "phase": 1,
        "target": "workstation-remote",
        "status": "RUNNING",
        "model_suite": "ridge_only",
        "matrix_values": {},
        "runtime_name": "exp-a-run-a",
        "report_date": "2026-04-22",
        "report_path": str(tmp_path / "missing-report.json"),
        "output_root": str(tmp_path / "output"),
        "supervisor_history": [],
    }
    (run_root / "run-a.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        research,
        "training_status_snapshot",
        lambda **kwargs: {
            "runtime": {
                "status": "failed",
                "running": False,
                "error": "remote process is no longer running",
            },
            "log_tail": ["dead"],
        },
    )

    result = research.sweep_stale_experiment_runs(
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    stored = json.loads((run_root / "run-a.json").read_text(encoding="utf-8"))
    assert result["reconciled"] == 1
    assert stored["status"] == "FAILED"
    assert stored["failure_kind"] == "infra"
    assert "no longer running" in stored["last_error"]
    summary = json.loads((local_state / "experiments" / "exp-a" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["FAILED"] == 1


def test_update_research_incumbent_promotes_only_fully_gated_candidate(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    data_root = tmp_path / "nas"
    policy = {
        "max_pbo": 0.5,
        "min_net_return": 0.0,
        "min_rank_ic_improvement": 0.002,
        "min_net_return_improvement": 0.01,
    }
    strong = {
        "experiment_id": "exp-a",
        "run_id": "strong",
        "shortlisted": True,
        "survived_predictive": True,
        "decision": "GO",
        "years_positive": True,
        "primary_score": 0.031,
        "backtest_net_return": 0.12,
        "pbo": 0.2,
        "assessment": {"decision": "GO"},
        "backtest_status": "COMPLETED",
        "artifacts": {"primary_predictions_path": str(tmp_path / "predictions.parquet")},
    }
    weak = {**strong, "run_id": "weak", "primary_score": 0.032, "backtest_net_return": 0.121}

    promoted = research.update_research_incumbent(
        program_id="perpetual-macmini",
        local_state=local_state,
        data_root=data_root,
        candidates=[strong],
        policy=policy,
    )
    rejected = research.update_research_incumbent(
        program_id="perpetual-macmini",
        local_state=local_state,
        data_root=data_root,
        candidates=[weak],
        policy=policy,
    )

    assert promoted["promoted"] is True
    assert rejected["promoted"] is False
    incumbent = research.read_research_incumbent(local_state=local_state, program_id="perpetual-macmini")
    assert incumbent["run_id"] == "strong"
    assert "does not improve incumbent" in rejected["rejections"][0]["reasons"]
    assert (data_root / "control" / "cluster" / "state" / "research" / "incumbents" / "perpetual-macmini.json").exists()


def test_write_paper_outputs_for_incumbent_is_deterministic(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "date": ["2026-04-24", "2026-04-24", "2026-04-24", "2026-04-24", "2026-04-24"],
            "symbol": ["A", "B", "C", "D", "E"],
            "prediction": [0.4, 0.1, 0.3, 0.2, 0.0],
        }
    ).to_parquet(predictions_path, index=False)
    incumbent = {
        "program_id": "perpetual-macmini",
        "run_id": "strong",
        "artifacts": {"primary_predictions_path": str(predictions_path)},
    }

    first = research.write_paper_outputs_for_incumbent(
        incumbent=incumbent,
        data_root=tmp_path / "nas",
        local_state=tmp_path / "local",
        policy={"enabled": True, "rebalance_day": "FRI", "no_live_orders": True},
    )
    second = research.write_paper_outputs_for_incumbent(
        incumbent=incumbent,
        data_root=tmp_path / "nas",
        local_state=tmp_path / "local",
        policy={"enabled": True, "rebalance_day": "FRI", "no_live_orders": True},
    )

    assert first["date"] == "2026-04-24"
    assert first["paper_orders_path"] == second["paper_orders_path"]
    signals = pd.read_parquet(first["signals_path"])
    orders = pd.read_parquet(first["paper_orders_path"])
    assert signals["symbol"].tolist()[0] == "A"
    assert orders.loc[orders["symbol"] == "A", "order_delta"].iloc[0] == pytest.approx(1.0)


def test_research_alerts_write_files_and_skip_email_without_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRADEML_SMTP_HOST", raising=False)
    monkeypatch.delenv("TRADEML_ALERT_EMAIL_TO", raising=False)

    alerts = research.evaluate_research_drift(
        program_id="perpetual-macmini",
        state={"status": "RUNNING", "last_completed_run_at": "2026-04-24T00:00:00+00:00"},
        incumbent={"primary_score": 0.03},
        latest_summary={
            "counts": {"FAILED": 4, "COMPLETED": 0},
            "runs": [{"report_preview": {"coverage": 0.95}}],
        },
        policy={"min_coverage": 0.98, "max_infra_failures": 3, "max_hours_without_completed_run": 24},
        now=datetime.fromisoformat("2026-04-26T12:00:00+00:00"),
    )
    result = research.write_research_alerts(
        program_id="perpetual-macmini",
        alerts=alerts,
        local_state=tmp_path / "local",
        data_root=tmp_path / "nas",
        policy={"email_enabled": True, "write_files": True},
    )

    assert {alert["kind"] for alert in alerts} >= {"coverage", "infra_failures", "stalled_research"}
    assert Path(result["json_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    assert result["email"]["status"] == "skipped"


def test_send_research_alert_email_uses_configured_smtp(monkeypatch) -> None:
    sent: dict[str, object] = {}

    class FakeSMTP:
        def __init__(self, host, port):  # noqa: ANN001
            sent["connect"] = (host, port)

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, *args):  # noqa: ANN001
            return None

        def starttls(self):  # noqa: ANN001
            sent["starttls"] = True

        def login(self, username, password):  # noqa: ANN001
            sent["login"] = (username, password)

        def send_message(self, message):  # noqa: ANN001
            sent["message"] = message

    monkeypatch.setenv("TRADEML_SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("TRADEML_SMTP_PORT", "2525")
    monkeypatch.setenv("TRADEML_ALERT_EMAIL_TO", "ops@example.test")
    monkeypatch.setenv("TRADEML_ALERT_EMAIL_FROM", "trademl@example.test")
    monkeypatch.setenv("TRADEML_SMTP_STARTTLS", "1")
    monkeypatch.setenv("TRADEML_SMTP_USERNAME", "user")
    monkeypatch.setenv("TRADEML_SMTP_PASSWORD", "pass")

    result = research.send_research_alert_email(
        program_id="perpetual-macmini",
        alerts=[{"kind": "coverage", "severity": "warning", "message": "low coverage"}],
        smtp_factory=FakeSMTP,
    )

    assert result["status"] == "sent"
    assert sent["connect"] == ("smtp.example.test", 2525)
    assert sent["login"] == ("user", "pass")
    assert "coverage" in sent["message"].get_content()
