from __future__ import annotations

import json
from pathlib import Path

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
