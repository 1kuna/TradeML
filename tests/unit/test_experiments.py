from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

import trademl.experiments as experiments
from trademl.research_architecture import (
    architecture_registry_payload,
    objective_registry_payload,
    resolve_architecture_entry,
)


def test_plan_experiment_materializes_deterministic_runs(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text(
        yaml.safe_dump({"model": {"learning_rate": 0.05}}, sort_keys=False),
        encoding="utf-8",
    )
    (repo_root / "configs" / "node.yml").write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
                "matrix": {
                    "model_suite": ["ridge_only", "full"],
                    "model.learning_rate": [0.05],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-04-02")

    summary = experiments.plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert summary["experiment_id"] == "phase1-baselines"
    assert summary["run_count"] == 2
    manifest_paths = sorted((local_state / "experiments" / "phase1-baselines" / "runs").glob("*.json"))
    assert len(manifest_paths) == 2
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["report_date"] == "2026-04-02"
    assert manifest["status"] == "PLANNED"
    assert manifest["output_root"].endswith(f"/experiments/phase1-baselines/runs/{manifest['run_id']}")
    assert manifest["report_path"].endswith(f"/experiments/phase1-baselines/runs/{manifest['run_id']}/reports/daily/2026-04-02.json")


def test_plan_experiment_merges_program_modeling_config_into_run_config(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text(
        yaml.safe_dump(
            {
                "modeling": {
                    "feature_store": {"enabled": True},
                    "feature_version": "price_liquidity_v1",
                    "label_version": "universe_relative_forward_return_v1",
                    "primary_label_horizon": 5,
                },
                "validation": {
                    "primary": "expanding_walk_forward",
                    "negative_controls": {"max_abs_negative_control_ic": 0.10},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (repo_root / "configs" / "node.yml").write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-news",
                "phase": 1,
                "modeling": {
                    "feature_store": {"enabled": True},
                    "feature_version": "news_event_aggregates_v1",
                    "label_version": "universe_relative_forward_return_v1",
                    "primary_label_horizon": 5,
                },
                "validation": {"negative_controls": {"max_abs_future_news_leak_ic": 0.05}},
                "matrix": {
                    "architecture_family": ["advanced_challenger"],
                    "label_horizon": [5],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-04-02")

    summary = experiments.plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    manifest = json.loads(
        next((local_state / "experiments" / "phase1-news" / "runs").glob("*.json")).read_text(encoding="utf-8")
    )
    run_config = yaml.safe_load(Path(str(manifest["config_path"])).read_text(encoding="utf-8"))
    assert summary["runs"][0]["feature_version"] == "news_event_aggregates_v1"
    assert manifest["feature_version"] == "news_event_aggregates_v1"
    assert run_config["modeling"]["feature_version"] == "news_event_aggregates_v1"
    assert run_config["modeling"]["label_horizon"] == 5
    assert run_config["validation"]["primary"] == "expanding_walk_forward"
    assert run_config["validation"]["negative_controls"]["max_abs_negative_control_ic"] == 0.10
    assert run_config["validation"]["negative_controls"]["max_abs_future_news_leak_ic"] == 0.05


def test_plan_experiment_resolves_remote_report_date_from_target(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text(
        yaml.safe_dump({"model": {"learning_rate": 0.05}}, sort_keys=False),
        encoding="utf-8",
    )
    (repo_root / "configs" / "node.yml").write_text(
        yaml.safe_dump(
            {
                "training": {"default_target": "workstation-remote"},
                "training_targets": {
                    "workstation-remote": {
                        "kind": "ssh",
                        "host": "remote-box",
                        "user": "zach",
                        "repo_root": "/srv/trademl",
                        "data_root": "/srv/nas",
                        "python_executable": "/usr/bin/python3",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
                "target": "workstation-remote",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-03-09")

    summary = experiments.plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert summary["report_date"] == "2026-03-09"


def test_compare_and_render_experiment_reports_pick_best_run(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    runs_root = experiment_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    report_root = tmp_path / "nas" / "reports" / "daily"
    report_root.mkdir(parents=True, exist_ok=True)

    run_a = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "target": "local",
        "model_suite": "ridge_only",
        "matrix_values": {"model_suite": "ridge_only"},
        "report_path": str(report_root / "2026-04-02-a.json"),
    }
    run_b = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-b",
        "target": "workstation-remote",
        "model_suite": "full",
        "matrix_values": {"model_suite": "full"},
        "report_path": str(report_root / "2026-04-02-b.json"),
    }
    (runs_root / "run-a.json").write_text(json.dumps(run_a, indent=2), encoding="utf-8")
    (runs_root / "run-b.json").write_text(json.dumps(run_b, indent=2), encoding="utf-8")
    (report_root / "2026-04-02-a.json").write_text(
        json.dumps({"coverage": 0.91, "ridge": {"mean_rank_ic": 0.03}, "assessment": {"decision": "GO"}}),
        encoding="utf-8",
    )
    (report_root / "2026-04-02-b.json").write_text(
        json.dumps(
            {
                "coverage": 0.93,
                "ridge": {"mean_rank_ic": 0.02},
                "lightgbm": {"mean_rank_ic": 0.05},
                "diagnostics": {"pbo": 0.12, "dsr": 0.9},
                "assessment": {"decision": "GO"},
            }
        ),
        encoding="utf-8",
    )

    comparison = experiments.compare_experiment(experiment_id="phase1-baselines", local_state=local_state)
    rendered = experiments.render_experiment_report(experiment_id="phase1-baselines", local_state=local_state)

    assert comparison["best"]["run_id"] == "run-b"
    assert Path(rendered["json_path"]).exists()
    assert Path(rendered["markdown_path"]).exists()


def test_write_run_manifest_serializes_runtime_payload_with_paths(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    manifest = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "runtime": {
            "config_path": Path("/tmp/generated.yml"),
            "output_root": Path("/tmp/output"),
        },
    }

    experiments._write_run_manifest(  # noqa: SLF001
        local_state=local_state,
        experiment_id="phase1-baselines",
        manifest=manifest,
    )

    written = json.loads(
        (local_state / "experiments" / "phase1-baselines" / "runs" / "run-a.json").read_text(encoding="utf-8")
    )
    assert written["runtime"]["config_path"] == "/tmp/generated.yml"
    assert written["runtime"]["output_root"] == "/tmp/output"


def test_summary_run_row_preserves_launch_critical_fields() -> None:
    row = experiments._summary_run_row(  # noqa: SLF001
        {
            "experiment_id": "phase1-baselines",
            "run_id": "run-a",
            "phase": 1,
            "target": "local",
            "target_kind": "local",
            "report_date": "2026-04-02",
            "status": "PLANNED",
            "model_suite": "ridge_only",
            "matrix_values": {"model_suite": "ridge_only"},
            "config_overrides": {"validation.initial_train_years": 2},
            "config_path": "/tmp/run-a.yml",
            "runtime_name": "experiment_phase1-baselines__run-a",
            "local_runtime_path": "/tmp/local-runtime.json",
            "shared_runtime_path": "/tmp/shared-runtime.json",
            "output_root": "/tmp/output",
            "report_path": "/tmp/report.json",
            "retry_count": 0,
            "evaluation_stage": "PLANNED",
            "shortlisted": False,
        }
    )

    assert row["phase"] == 1
    assert row["target"] == "local"
    assert row["runtime_name"] == "experiment_phase1-baselines__run-a"
    assert row["config_path"] == "/tmp/run-a.yml"
    assert row["output_root"] == "/tmp/output"


def test_materialize_matrix_row_resolves_architecture_feature_and_data_profiles() -> None:
    row = experiments._materialize_matrix_row(  # noqa: SLF001
        {
            "architecture_family": "linear_baseline",
            "feature_family": "price_core",
            "data_profile": "phase1_short_window",
            "data_family": "price_only",
            "validation.initial_train_years": 3,
        }
    )

    assert row["model_suite"] == "ridge_only"
    assert row["config_overrides"]["features.liquidity.adv_dollar"] == []
    assert row["config_overrides"]["features.liquidity.amihud"] == []
    assert row["config_overrides"]["data.window_years"] == 3
    assert row["config_overrides"]["validation.initial_train_years"] == 3


def test_frontier_architecture_manifest_order_prefers_advanced(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data: {}\n", encoding="utf-8")
    (repo_root / "configs" / "node.yml").write_text("", encoding="utf-8")
    spec_path = tmp_path / "frontier.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "frontier-order",
                "phase": 1,
                "frontier_architecture": True,
                "matrix": {
                    "architecture_family": ["advanced_challenger", "tree_challenger", "linear_baseline"],
                    "feature_family": ["price_liquidity"],
                    "data_family": ["price_plus_liquidity"],
                    "data_profile": ["phase1_default"],
                    "validation.initial_train_years": [2],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-03-09")

    experiments.plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    manifests = experiments._load_run_manifests(local_state=local_state, experiment_id="frontier-order")  # noqa: SLF001
    assert [manifest["model_suite"] for manifest in manifests] == ["advanced", "full", "ridge_only"]
    assert [manifest["run_priority"] for manifest in manifests] == [0, 1, 2]
    assert manifests[0]["architecture_registry_entry"]["family"] == "advanced_challenger"
    assert manifests[0]["objective_policy"]["primary"] == "research_profitability_v1"
    assert manifests[0]["complexity_tier"] == 2


def test_architecture_registry_resolves_current_and_blocks_deferred_lanes() -> None:
    registry = architecture_registry_payload()
    objective = objective_registry_payload()

    assert registry["linear_baseline"]["model_suite"] == "ridge_only"
    assert registry["tree_challenger"]["required_packages"] == ["lightgbm", "optuna"]
    assert registry["advanced_challenger"]["primary_metrics"][0] == "catboost_mean_rank_ic"
    assert registry["ensemble_meta"]["implemented"] is True
    assert registry["ensemble_meta"]["model_suite"] == "ensemble"
    assert registry["ensemble_meta"]["canary_eligible"] is True
    assert registry["ensemble_meta"]["pivot_role"] == "advanced_failure_pivot"
    assert registry["tabular_deep_challenger"]["canary_eligible"] is False
    assert objective["research_profitability_v1"]["primary_metric"] == "rank_ic"

    with pytest.raises(ValueError, match="deferred"):
        resolve_architecture_entry("tabular_deep_challenger")


def test_plan_experiment_rejects_disabled_future_architecture_lane(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text("data: {}\n", encoding="utf-8")
    (repo_root / "configs" / "node.yml").write_text("", encoding="utf-8")
    spec_path = tmp_path / "future.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "future-lane",
                "phase": 1,
                "matrix": {"architecture_family": ["rl_policy"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-03-09")

    with pytest.raises(ValueError, match="unsupported architecture_family"):
        experiments.plan_experiment(
            spec_path=spec_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            targets_config_path=repo_root / "configs" / "node.yml",
            python_executable="/usr/bin/python3",
        )


def test_primary_rank_ic_prefers_catboost_for_advanced_suite() -> None:
    score = experiments._primary_rank_ic(  # noqa: SLF001
        manifest={"model_suite": "advanced"},
        report={
            "ridge": {"mean_rank_ic": 0.01},
            "lightgbm": {"mean_rank_ic": 0.02},
            "catboost": {"mean_rank_ic": 0.05},
        },
    )

    assert score == 0.05


def test_objective_evaluation_groups_gate_failures_and_adjusts_for_complexity() -> None:
    evaluation = experiments._evaluate_report(  # noqa: SLF001
        manifest={
            "run_id": "advanced-a",
            "model_suite": "advanced",
            "matrix_values": {"architecture_family": "advanced_challenger"},
        },
        report={
            "coverage": 0.99,
            "ridge": {"mean_rank_ic": 0.01},
            "lightgbm": {"mean_rank_ic": 0.02},
            "catboost": {"mean_rank_ic": 0.05},
            "assessment": {"decision": "GO"},
            "diagnostics": {
                "ic_by_year": {"2024": 0.01, "2025": 0.02},
                "placebo": [0.01],
                "cost_stress": {"net_return": 0.02},
                "pbo": 0.2,
                "negative_controls": {
                    "shuffled_label_max_abs_ic": 0.01,
                    "date_shifted_label_max_abs_ic": 0.01,
                    "random_feature_max_abs_ic": 0.01,
                    "ticker_news_permutation_max_abs_ic": 0.0,
                    "future_news_leak_sentinel_ic": 0.0,
                    "max_single_feature_score_drop": 0.1,
                    "min_feature_ablation_score_ratio": 0.9,
                },
            },
        },
        gate={
            "require_go_decision": True,
            "min_rank_ic": 0.01,
            "require_all_years_positive": True,
            "max_abs_placebo_ic": 0.10,
            "min_cost_stress_net_return": 0.0,
            "max_pbo": 0.5,
            "min_dsr": None,
            "min_coverage": 0.0,
        },
    )

    assert evaluation["primary_rank_ic"] == 0.05
    assert evaluation["objective_verdict"]["primary_score"] == 0.05
    assert evaluation["objective_verdict"]["complexity_tier"] == 2
    assert evaluation["objective_verdict"]["complexity_adjusted_score"] < 0.05
    assert evaluation["gate_failures_by_objective"] == {}


def test_strong_unstable_candidate_gets_autopsy_classification() -> None:
    evaluation = experiments._evaluate_report(  # noqa: SLF001
        manifest={
            "run_id": "advanced-unstable",
            "model_suite": "advanced",
            "matrix_values": {"architecture_family": "advanced_challenger"},
        },
        report={
            "coverage": 1.0,
            "ridge": {"mean_rank_ic": -0.08},
            "lightgbm": {"mean_rank_ic": 0.05},
            "catboost": {
                "mean_rank_ic": 0.077,
                "folds": [
                    {"rank_ic": -0.29},
                    {"rank_ic": 0.13},
                    {"rank_ic": 0.19},
                ],
            },
            "assessment": {"decision": "NO_GO"},
            "diagnostics": {
                "ic_by_year": {"2025": -0.01, "2026": 0.28},
                "ic_by_quarter": {"2025Q2": -0.29, "2025Q3": 0.12},
                "placebo": [0.01],
                "cost_stress": {"net_return": 1.5},
                "pbo": 0.57,
                "cpcv": {"mean_oos_score": -0.03},
                "negative_controls": {
                    "shuffled_label_max_abs_ic": 0.01,
                    "date_shifted_label_max_abs_ic": 0.01,
                    "random_feature_max_abs_ic": 0.01,
                    "ticker_news_permutation_max_abs_ic": 0.0,
                    "future_news_leak_sentinel_ic": 0.0,
                    "max_single_feature_score_drop": 0.1,
                    "min_feature_ablation_score_ratio": 0.9,
                },
            },
        },
        gate={
            "require_go_decision": True,
            "min_rank_ic": 0.01,
            "require_all_years_positive": True,
            "max_abs_placebo_ic": 0.10,
            "min_cost_stress_net_return": 0.0,
            "max_pbo": None,
            "min_dsr": None,
            "min_coverage": 0.0,
        },
    )

    autopsy = evaluation["candidate_autopsy"]
    assert autopsy["classification"] == "strong_unstable"
    assert autopsy["recommended_follow_up"]["diagnostic_mode"] == "strong_unstable"
    assert autopsy["evidence"]["worst_quarter"] == {"period": "2025Q2", "value": -0.29}
    assert "cpcv_mean_oos_score<0" in autopsy["evidence"]["overfit_evidence"]


def test_objective_evaluation_rejects_negative_control_failures() -> None:
    evaluation = experiments._evaluate_report(  # noqa: SLF001
        manifest={"run_id": "advanced-leaky", "model_suite": "advanced"},
        report={
            "coverage": 0.99,
            "catboost": {"mean_rank_ic": 0.05},
            "assessment": {"decision": "GO"},
            "diagnostics": {
                "ic_by_year": {"2024": 0.01, "2025": 0.02},
                "placebo": [0.01],
                "cost_stress": {"net_return": 0.02},
                "pbo": 0.2,
                "negative_controls": {
                    "shuffled_label_max_abs_ic": 0.22,
                    "future_news_leak_sentinel_ic": 0.18,
                    "max_single_feature_score_drop": 0.9,
                },
            },
        },
        gate={
            "require_go_decision": True,
            "min_rank_ic": 0.01,
            "require_all_years_positive": True,
            "max_abs_placebo_ic": 0.10,
            "max_abs_negative_control_ic": 0.10,
            "max_abs_future_news_leak_ic": 0.10,
            "max_single_feature_score_drop": 0.75,
            "min_feature_ablation_score_ratio": 0.25,
            "min_cost_stress_net_return": 0.0,
            "max_pbo": 0.5,
            "min_dsr": None,
            "min_coverage": 0.0,
        },
    )

    assert evaluation["survived_predictive"] is False
    assert "negative_control.shuffled_label_max_abs_ic>0.1" in evaluation["gate_failures"]
    assert "future_news_leak_sentinel_ic>0.1" in evaluation["gate_failures"]
    assert "single_feature_dependence>0.75" in evaluation["gate_failures"]
    assert set(evaluation["gate_failures_by_objective"]) == {"false_discovery", "leakage", "fragility"}


def test_report_preview_preserves_catboost_primary_score() -> None:
    preview = experiments._report_preview_from_report(  # noqa: SLF001
        {
            "coverage": 0.99,
            "ridge": {"mean_rank_ic": 0.01},
            "lightgbm": {"mean_rank_ic": 0.02},
            "catboost": {"mean_rank_ic": 0.05},
        }
    )

    assert experiments._preview_primary_score(model_suite="advanced", preview=preview) == 0.05  # noqa: SLF001


def test_report_preview_preserves_ensemble_primary_score() -> None:
    preview = experiments._report_preview_from_report(  # noqa: SLF001
        {
            "coverage": 0.99,
            "ridge": {"mean_rank_ic": 0.01},
            "lightgbm": {"mean_rank_ic": 0.02},
            "ensemble": {"mean_rank_ic": 0.04},
        }
    )

    assert experiments._preview_primary_score(model_suite="ensemble", preview=preview) == 0.04  # noqa: SLF001


def test_completed_advanced_manifest_refreshes_missing_catboost_preview() -> None:
    assert experiments._manifest_needs_report_refresh(  # noqa: SLF001
        {
            "status": "COMPLETED",
            "model_suite": "advanced",
            "assessment": {"decision": "NO_GO"},
            "report_preview": {"ridge_mean_rank_ic": 0.01, "lightgbm_mean_rank_ic": 0.02},
        }
    )


def test_experiment_summary_ignores_unscored_planned_runs_for_best(tmp_path: Path) -> None:
    def run_row(run_id: str, model_suite: str, status: str, preview: dict[str, float]) -> dict[str, object]:
        return {
            "experiment_id": "frontier",
            "run_id": run_id,
            "phase": 1,
            "target": "workstation-remote",
            "target_kind": "ssh",
            "report_date": "2026-03-09",
            "status": status,
            "model_suite": model_suite,
            "matrix_values": {"model_suite": model_suite},
            "evaluation_stage": "PLANNED",
            "assessment": {},
            "report_preview": preview,
        }

    summary = experiments._refresh_experiment_summary(  # noqa: SLF001
        local_state=tmp_path,
        experiment_id="frontier",
        summary={
            "experiment_id": "frontier",
            "counts": {"PLANNED": 1, "COMPLETED": 1},
            "runs": [
                run_row("planned-ridge", "ridge_only", "PLANNED", {}),
                run_row("advanced", "advanced", "COMPLETED", {"catboost_mean_rank_ic": -0.006}),
            ],
        },
    )

    assert summary["best_run_id"] == "advanced"
    assert summary["best_primary_score"] == -0.006


def test_classify_failure_treats_remote_import_bootstrap_as_infra() -> None:
    kind = experiments._classify_failure(  # noqa: SLF001
        "remote process is not running\n"
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'trademl.models.catboost'\n"
    )

    assert kind == "infra"


def test_manifest_requires_runtime_refresh_for_generic_remote_exit_wrapper() -> None:
    assert experiments._manifest_requires_runtime_refresh(  # noqa: SLF001
        {
            "status": "FAILED",
            "failure_kind": "model",
            "last_error": "remote process is not running",
        }
    )


def test_run_experiment_until_idle_drains_planned_runs_without_sleeping(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    sleep_calls: list[int] = []
    monkeypatch.setattr(
        experiments,
        "supervise_experiment",
        lambda **kwargs: {"experiment_id": "phase1-baselines", "counts": {"COMPLETED": 2}, "launch_history": [{"run_id": "run-a"}, {"run_id": "run-b"}]},
    )
    monkeypatch.setattr(experiments.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    payload = experiments.run_experiment_until_idle(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=7,
    )

    assert payload["counts"] == {"COMPLETED": 2}
    assert len(payload["launch_history"]) == 2


def test_launch_experiment_uses_full_manifests_not_summary_rows(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "equities_xs.yml").write_text(
        yaml.safe_dump({"model": {"learning_rate": 0.05}}, sort_keys=False),
        encoding="utf-8",
    )
    (repo_root / "configs" / "node.yml").write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
                "matrix": {"model_suite": ["ridge_only"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(experiments, "_resolve_default_report_date", lambda **kwargs: "2026-04-02")
    monkeypatch.setattr(
        experiments,
        "training_status_snapshot",
        lambda **kwargs: {"runtime": {"status": "PLANNED"}, "log_tail": ""},
    )

    seen: dict[str, object] = {}

    def fake_launch_training_process(**kwargs):
        seen["phase"] = kwargs["phase"]
        seen["runtime_name"] = kwargs["runtime_name"]
        return {"status": "starting", "pid": 1234}

    monkeypatch.setattr(experiments, "launch_training_process", fake_launch_training_process)

    payload = experiments.launch_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert payload["launched"][0]["run_id"]
    assert seen["phase"] == 1
    assert str(seen["runtime_name"]).startswith("experiment_phase1-baselines__")


def test_evaluate_experiment_marks_predictive_rejections(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    runs_root = experiment_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": "phase1-baselines",
        "spec_path": str(tmp_path / "phase1.yml"),
    }
    (experiment_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "phase1.yml").write_text(
        yaml.safe_dump({"experiment_id": "phase1-baselines", "predictive_gate": {"require_go_decision": True, "min_rank_ic": 0.02}}, sort_keys=False),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "coverage": 1.0,
                "ridge": {"mean_rank_ic": 0.01},
                "diagnostics": {"ic_by_year": {"2025": -0.01}, "placebo": [0.0], "cost_stress": {"net_return": 0.1}, "pbo": 0.1, "dsr": 0.5},
                "assessment": {"decision": "NO_GO", "reason": "bad ic"},
            }
        ),
        encoding="utf-8",
    )
    (runs_root / "run-a.json").write_text(
        json.dumps(
            {
                "experiment_id": "phase1-baselines",
                "run_id": "run-a",
                "status": "COMPLETED",
                "target": "local",
                "target_kind": "local",
                "phase": 1,
                "report_path": str(report_path),
                "model_suite": "ridge_only",
                "matrix_values": {},
                "runtime_name": "run-a",
            }
        ),
        encoding="utf-8",
    )

    payload = experiments.evaluate_experiment(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path,
        data_root=tmp_path,
        targets_config_path=tmp_path / "node.yml",
        python_executable="/usr/bin/python3",
    )

    manifest = json.loads((runs_root / "run-a.json").read_text(encoding="utf-8"))
    assert payload["evaluated"][0]["evaluation_stage"] == "REJECTED_PREDICTIVE"
    assert manifest["evaluation_stage"] == "REJECTED_PREDICTIVE"
    assert manifest["gate_failures"]


def test_propose_next_experiment_family_writes_bounded_spec(tmp_path: Path) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    experiment_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "phase1.yml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
                "target": "workstation-remote",
                "proposal_policy": {
                    "allowed_dimensions": ["architecture_family", "feature_family", "validation.initial_train_years", "data_profile"],
                    "max_generations": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (experiment_root / "summary.json").write_text(json.dumps({"experiment_id": "phase1-baselines", "spec_path": str(tmp_path / "phase1.yml")}), encoding="utf-8")
    runs_root = experiment_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "coverage": 1.0,
                "ridge": {"mean_rank_ic": 0.03},
                "diagnostics": {"pbo": 0.1, "dsr": 0.5},
                "assessment": {"decision": "NO_GO", "reason": "ic bad"},
            }
        ),
        encoding="utf-8",
    )
    (runs_root / "run-a.json").write_text(
        json.dumps(
            {
                "experiment_id": "phase1-baselines",
                "run_id": "run-a",
                "target": "local",
                "status": "COMPLETED",
                "phase": 1,
                "model_suite": "ridge_only",
                "matrix_values": {"validation.initial_train_years": 2},
                "evaluation_stage": "REJECTED_PREDICTIVE",
                "report_path": str(report_path),
                "report_preview": {"ridge_mean_rank_ic": 0.03},
            }
        ),
        encoding="utf-8",
    )

    payload = experiments.propose_next_experiment_family(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path,
        data_root=tmp_path,
        targets_config_path=tmp_path / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert payload["proposal"]["recommended_experiment_id"] == "phase1-baselines-g1"
    assert Path(payload["proposal"]["spec_path"]).exists()
    assert payload["proposal"]["chain_allowed"] is True
    assert payload["proposal"]["next_spec"]["matrix"]["architecture_family"] == ["linear_baseline"]
    assert "feature_family" in payload["proposal"]["next_spec"]["matrix"]


def test_build_next_family_proposal_pivots_axes_after_feature_sweep(tmp_path: Path) -> None:
    comparison = {
        "rows": [
            {
                "model_suite": "ridge_only",
                "primary_score": 0.004,
                "evaluation_stage": "REJECTED_PREDICTIVE",
                "matrix_values": {
                    "architecture_family": "linear_baseline",
                    "feature_family": "price_liquidity",
                    "validation.initial_train_years": 3,
                },
            },
            {
                "model_suite": "full",
                "primary_score": -0.002,
                "evaluation_stage": "REJECTED_PREDICTIVE",
                "matrix_values": {
                    "architecture_family": "tree_challenger",
                    "feature_family": "price_core",
                    "validation.initial_train_years": 2,
                },
            },
        ]
    }
    base_spec = {
        "experiment_id": "phase1-macmini-autoloop-g2",
        "phase": 1,
        "target": "local",
        "generation": 2,
        "family_root": "phase1-macmini-autoloop",
        "proposal_policy": {
            "allowed_dimensions": [
                "architecture_family",
                "feature_family",
                "validation.initial_train_years",
                "data_profile",
            ],
            "family_size_cap": 6,
            "max_generations": 4,
            "auto_launch_next_family": True,
        },
    }

    proposal = experiments._build_next_family_proposal(  # noqa: SLF001
        experiment_id="phase1-macmini-autoloop-g2",
        base_spec=base_spec,
        comparison=comparison,
    )

    matrix = proposal["next_spec"]["matrix"]
    assert matrix["feature_family"] == ["price_liquidity"]
    assert sorted(matrix["architecture_family"]) == ["linear_baseline", "tree_challenger"]
    assert matrix["data_profile"] == ["phase1_default", "phase1_short_window", "phase1_long_window"]
    assert "validation.initial_train_years" not in matrix


def test_build_next_family_proposal_prioritizes_strong_unstable_follow_up() -> None:
    proposal = experiments._build_next_family_proposal(  # noqa: SLF001
        experiment_id="phase1-advanced",
        base_spec={
            "experiment_id": "phase1-advanced",
            "family_root": "perpetual-macmini",
            "generation": 10,
            "phase": 1,
            "target": "local",
            "objective_policy": {"primary": "research_profitability_v1"},
            "matrix": {
                "architecture_family": ["advanced_challenger", "tree_challenger"],
                "feature_family": ["price_short_horizon"],
                "data_family": ["price_plus_liquidity"],
                "data_profile": ["phase1_long_window"],
                "label_horizon": [20],
                "validation.initial_train_years": [4],
            },
            "proposal_policy": {
                "family_size_cap": 12,
                "allowed_dimensions": [
                    "architecture_family",
                    "feature_family",
                    "data_family",
                    "label_horizon",
                    "validation.initial_train_years",
                    "data_profile",
                ],
                "max_generations": 100,
            },
        },
        comparison={
            "rows": [
                {
                    "run_id": "ae6b12e3df",
                    "primary_score": 0.077,
                    "model_suite": "advanced",
                    "feature_version": "price_liquidity_v1",
                    "label_version": "universe_relative_forward_return_v1",
                    "data_revision": "rev1",
                    "portfolio_profile": "cost_aware_long_only_v1",
                    "matrix_values": {
                        "architecture_family": "advanced_challenger",
                        "feature_family": "price_short_horizon",
                        "data_family": "price_plus_liquidity",
                        "data_profile": "phase1_long_window",
                        "label_horizon": 20,
                        "validation.initial_train_years": 4,
                    },
                    "candidate_autopsy": {"classification": "strong_unstable"},
                }
            ]
        },
    )

    assert proposal["diagnostic_mode"] == "strong_unstable"
    assert proposal["next_spec"]["follow_up_of_run_id"] == "ae6b12e3df"
    assert proposal["next_spec"]["matrix"]["architecture_family"] == [
        "advanced_challenger",
        "ensemble_meta",
        "tree_challenger",
        "linear_baseline",
    ]
    matrix = proposal["next_spec"]["matrix"]
    assert matrix["label_horizon"] == [20, 1, 5]
    assert _matrix_size(matrix) <= 12
    assert proposal["next_spec"]["diagnostic_family_signature"]


def test_build_next_family_proposal_pivots_feature_family_after_architecture_window_sweep() -> None:
    comparison = {
        "rows": [
            {
                "model_suite": "ridge_only",
                "primary_score": -0.001,
                "evaluation_stage": "REJECTED_PREDICTIVE",
                "matrix_values": {
                    "architecture_family": "linear_baseline",
                    "feature_family": "price_liquidity",
                    "data_profile": "phase1_default",
                },
            },
            {
                "model_suite": "full",
                "primary_score": -0.003,
                "evaluation_stage": "REJECTED_PREDICTIVE",
                "matrix_values": {
                    "architecture_family": "tree_challenger",
                    "feature_family": "price_liquidity",
                    "data_profile": "phase1_short_window",
                },
            },
        ]
    }
    base_spec = {
        "experiment_id": "phase1-macmini-autoloop-g3",
        "phase": 1,
        "target": "local",
        "generation": 3,
        "family_root": "phase1-macmini-autoloop",
        "matrix": {
            "architecture_family": ["linear_baseline", "tree_challenger"],
            "feature_family": ["price_liquidity"],
            "data_profile": ["phase1_default", "phase1_short_window", "phase1_long_window"],
        },
        "proposal_policy": {
            "allowed_dimensions": [
                "architecture_family",
                "feature_family",
                "validation.initial_train_years",
                "data_profile",
            ],
            "family_size_cap": 6,
            "max_generations": 6,
            "auto_launch_next_family": True,
        },
    }

    proposal = experiments._build_next_family_proposal(  # noqa: SLF001
        experiment_id="phase1-macmini-autoloop-g3",
        base_spec=base_spec,
        comparison=comparison,
    )

    matrix = proposal["next_spec"]["matrix"]
    assert sorted(matrix["architecture_family"]) == ["linear_baseline", "tree_challenger"]
    assert sorted(matrix["feature_family"]) == ["price_core", "price_short_horizon"]
    assert matrix["data_profile"] == ["phase1_default"]
    assert "validation.initial_train_years" not in matrix


def test_supervise_experiment_autochains_next_family_when_policy_allows(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
                "proposal_policy": {"auto_launch_next_family": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiments,
        "plan_experiment",
        lambda **kwargs: {"experiment_id": "phase1-baselines", "max_concurrent": 1},
    )
    monkeypatch.setattr(experiments, "_supervision_policy", lambda spec: {"poll_seconds": 1, "auto_backtest_survivors": False, "auto_propose_next_family": True, "max_retry_count": 1})
    monkeypatch.setattr(experiments, "_proposal_policy", lambda spec: {"family_size_cap": 6, "allowed_dimensions": [], "max_generations": 1, "auto_launch_next_family": True})
    monkeypatch.setattr(experiments, "read_experiment_supervisor_state", lambda **kwargs: {})
    monkeypatch.setattr(experiments, "_write_supervisor_state", lambda **kwargs: None)
    monkeypatch.setattr(
        experiments,
        "experiment_status",
        lambda **kwargs: {"counts": {"COMPLETED": 4}, "runs": []},
    )
    monkeypatch.setattr(experiments, "evaluate_experiment", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(experiments, "propose_next_experiment_family", lambda **kwargs: {"proposal": {"chain_allowed": True, "recommended_experiment_id": "phase1-baselines-g1", "spec_path": str(tmp_path / "next.yml")}})

    spawned: list[dict[str, object]] = []

    def fake_spawn_supervisor_process(**kwargs):
        spawned.append(kwargs)
        return {"pid": 4321, "experiment_id": kwargs["experiment_id"]}

    monkeypatch.setattr(experiments, "_spawn_supervisor_process", fake_spawn_supervisor_process)

    state = experiments.supervise_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=1,
        detach=False,
    )

    assert state["status"] == "COMPLETED"
    assert state["next_experiment_id"] == "phase1-baselines-g1"
    assert spawned[0]["experiment_id"] == "phase1-baselines-g1"


def test_supervise_experiment_completes_when_only_exhausted_failed_runs_remain(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    local_state = tmp_path / "local"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    spec_path = tmp_path / "phase1.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "phase1-baselines",
                "phase": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiments,
        "plan_experiment",
        lambda **kwargs: {"experiment_id": "phase1-baselines", "max_concurrent": 1},
    )
    monkeypatch.setattr(
        experiments,
        "_supervision_policy",
        lambda spec: {"poll_seconds": 1, "auto_backtest_survivors": False, "auto_propose_next_family": False, "max_retry_count": 2},
    )
    monkeypatch.setattr(experiments, "_proposal_policy", lambda spec: {"family_size_cap": 6, "allowed_dimensions": [], "max_generations": 1, "auto_launch_next_family": False})
    monkeypatch.setattr(experiments, "read_experiment_supervisor_state", lambda **kwargs: {})
    monkeypatch.setattr(experiments, "_write_supervisor_state", lambda **kwargs: None)
    monkeypatch.setattr(
        experiments,
        "experiment_status",
        lambda **kwargs: {
            "counts": {"COMPLETED": 68, "FAILED": 4},
            "runs": [
                {"run_id": "run-a", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
                {"run_id": "run-b", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
                {"run_id": "run-c", "status": "FAILED", "failure_kind": "infra", "retry_count": 2},
                {"run_id": "run-d", "status": "FAILED", "failure_kind": "model", "retry_count": 0},
            ],
        },
    )
    monkeypatch.setattr(experiments, "evaluate_experiment", lambda **kwargs: {"ok": True})

    def unexpected_launch(**kwargs):  # noqa: ANN001
        raise AssertionError("terminal failed runs should not be relaunched")

    monkeypatch.setattr(experiments, "launch_experiment", unexpected_launch)

    state = experiments.supervise_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=repo_root / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
        poll_seconds=1,
        detach=False,
    )

    assert state["status"] == "COMPLETED"


def test_experiment_status_skips_remote_polling_for_planned_runs(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    (experiment_root / "runs").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "phase": 1,
        "target": "workstation-remote",
        "runtime_name": "experiment_phase1-baselines__run-a",
        "status": "PLANNED",
        "model_suite": "ridge_only",
        "matrix_values": {"architecture_family": "linear_baseline"},
        "config_path": str(tmp_path / "run-a.yml"),
        "output_root": "/remote/output",
        "report_path": "/remote/report.json",
    }
    (experiment_root / "runs" / "run-a.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment_root / "summary.json").write_text(json.dumps({"experiment_id": "phase1-baselines"}), encoding="utf-8")

    def unexpected_status_snapshot(**kwargs):  # noqa: ANN001
        raise AssertionError("planned runs should not invoke training_status_snapshot")

    def unexpected_report_load(**kwargs):  # noqa: ANN001
        raise AssertionError("planned runs should not invoke report loading")

    monkeypatch.setattr(experiments, "training_status_snapshot", unexpected_status_snapshot)
    monkeypatch.setattr(experiments, "_load_report_payload", unexpected_report_load)

    status = experiments.experiment_status(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert status["counts"] == {"PLANNED": 1}
    assert status["runs"][0]["run_id"] == "run-a"


def test_experiment_status_skips_remote_polling_for_completed_runs(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    (experiment_root / "runs").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "phase": 1,
        "target": "workstation-remote",
        "runtime_name": "experiment_phase1-baselines__run-a",
        "status": "COMPLETED",
        "evaluation_stage": "REJECTED_PREDICTIVE",
        "assessment": {"decision": "NO_GO"},
        "model_suite": "ridge_only",
        "matrix_values": {"architecture_family": "linear_baseline"},
        "config_path": str(tmp_path / "run-a.yml"),
        "output_root": "/remote/output",
        "report_path": "/remote/report.json",
    }
    (experiment_root / "runs" / "run-a.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment_root / "summary.json").write_text(json.dumps({"experiment_id": "phase1-baselines"}), encoding="utf-8")

    def unexpected_status_snapshot(**kwargs):  # noqa: ANN001
        raise AssertionError("completed runs should not invoke training_status_snapshot")

    monkeypatch.setattr(experiments, "training_status_snapshot", unexpected_status_snapshot)

    status = experiments.experiment_status(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert status["counts"] == {"COMPLETED": 1}
    assert status["runs"][0]["run_id"] == "run-a"


def test_experiment_status_writes_shared_summary_to_data_root(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    data_root = tmp_path / "nas"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    (experiment_root / "runs").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "phase": 1,
        "target": "workstation-remote",
        "runtime_name": "experiment_phase1-baselines__run-a",
        "status": "COMPLETED",
        "evaluation_stage": "REJECTED_PREDICTIVE",
        "assessment": {"decision": "NO_GO", "reason": "bad ic"},
        "report_preview": {"coverage": 0.98, "ridge_mean_rank_ic": 0.031},
        "model_suite": "ridge_only",
        "matrix_values": {"architecture_family": "linear_baseline"},
        "config_path": str(tmp_path / "run-a.yml"),
        "output_root": "/remote/output",
        "report_path": "/remote/report.json",
    }
    (experiment_root / "runs" / "run-a.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment_root / "summary.json").write_text(json.dumps({"experiment_id": "phase1-baselines"}), encoding="utf-8")

    def unexpected_status_snapshot(**kwargs):  # noqa: ANN001
        raise AssertionError("completed runs should not invoke training_status_snapshot")

    monkeypatch.setattr(experiments, "training_status_snapshot", unexpected_status_snapshot)

    experiments.experiment_status(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=data_root,
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    shared_summary = json.loads((data_root / "experiments" / "phase1-baselines" / "summary.json").read_text(encoding="utf-8"))
    dashboard_summary = json.loads(
        (data_root / "experiments" / "phase1-baselines" / "dashboard_summary.json").read_text(encoding="utf-8")
    )

    assert shared_summary["experiment_id"] == "phase1-baselines"
    assert shared_summary["runs"][0]["run_id"] == "run-a"
    assert shared_summary["runs"][0]["assessment"]["decision"] == "NO_GO"
    assert shared_summary["runs"][0]["report_preview"]["ridge_mean_rank_ic"] == pytest.approx(0.031)
    assert dashboard_summary["experiment_id"] == "phase1-baselines"
    assert dashboard_summary["recent_runs"][0]["run_id"] == "run-a"
    assert dashboard_summary["recent_runs"][0]["assessment"]["decision"] == "NO_GO"


def test_experiment_status_reclassifies_unknown_runtime_as_failed_infra(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    experiment_root = local_state / "experiments" / "phase1-baselines"
    (experiment_root / "runs").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": "phase1-baselines",
        "run_id": "run-a",
        "phase": 1,
        "target": "local",
        "runtime_name": "experiment_phase1-baselines__run-a",
        "status": "RUNNING",
        "model_suite": "ridge_only",
        "matrix_values": {"architecture_family": "linear_baseline"},
        "config_path": str(tmp_path / "run-a.yml"),
        "output_root": str(tmp_path / "output"),
        "report_path": str(tmp_path / "report.json"),
    }
    (experiment_root / "runs" / "run-a.json").write_text(json.dumps(manifest), encoding="utf-8")
    (experiment_root / "summary.json").write_text(json.dumps({"experiment_id": "phase1-baselines"}), encoding="utf-8")

    monkeypatch.setattr(
        experiments,
        "training_status_snapshot",
        lambda **kwargs: {
            "runtime": {"status": "unknown", "running": False, "error": "local process is not running"},
            "log_tail": "Traceback\\nlocal process is not running",
        },
    )
    monkeypatch.setattr(experiments, "_load_report_payload", lambda **kwargs: None)

    status = experiments.experiment_status(
        experiment_id="phase1-baselines",
        local_state=local_state,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    run = status["runs"][0]
    assert run["status"] == "FAILED"
    assert run["failure_kind"] == "infra"
    assert run["last_error"] == "local process is not running"


def test_write_supervisor_state_is_atomic(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    experiment_id = "phase1-baselines"
    state_path = local_state / "experiment_supervisors" / f"{experiment_id}.json"
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
        experiments._write_supervisor_state(  # noqa: SLF001
            local_state=local_state,
            experiment_id=experiment_id,
            payload={"status": "FAILED"},
        )

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"status": "RUNNING"}
    assert list(state_path.parent.glob(f"{state_path.name}.tmp-*")) == []


def test_spawn_supervisor_process_writes_bootstrap_state_before_launch(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    spec_path = tmp_path / "phase1.yml"
    env_path.write_text("", encoding="utf-8")
    spec_path.write_text("experiment_id: phase1-baselines\n", encoding="utf-8")

    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, cwd, stdout, stderr, start_new_session):  # noqa: ANN001
        state = experiments.read_experiment_supervisor_state(local_state=local_state, experiment_id="phase1-baselines")
        observed["state_before_spawn"] = state
        return FakeProcess()

    monkeypatch.setattr(experiments.subprocess, "Popen", fake_popen)

    payload = experiments._spawn_supervisor_process(  # noqa: SLF001
        experiment_id="phase1-baselines",
        spec_path=spec_path,
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
    assert payload["pid"] == 4321
    assert payload["status"] == "RUNNING"


def test_spawn_supervisor_process_preserves_existing_state(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    spec_path = tmp_path / "phase1.yml"
    env_path.write_text("", encoding="utf-8")
    spec_path.write_text("experiment_id: phase1-baselines\n", encoding="utf-8")
    experiments._write_supervisor_state(  # noqa: SLF001
        local_state=local_state,
        experiment_id="phase1-baselines",
        payload={"experiment_id": "phase1-baselines", "queue_counts": {"COMPLETED": 12}, "status": "STOPPED"},
    )

    class FakeProcess:
        pid = 7654

    monkeypatch.setattr(experiments.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    payload = experiments._spawn_supervisor_process(  # noqa: SLF001
        experiment_id="phase1-baselines",
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        python_executable="/usr/bin/python3",
        poll_seconds=30,
    )

    stored = experiments.read_experiment_supervisor_state(local_state=local_state, experiment_id="phase1-baselines")
    assert payload["pid"] == 7654
    assert stored["queue_counts"] == {"COMPLETED": 12}


def test_spawn_supervisor_process_marks_state_stopped_when_popen_raises(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "nas"
    env_path = tmp_path / ".env"
    spec_path = tmp_path / "phase1.yml"
    env_path.write_text("", encoding="utf-8")
    spec_path.write_text("experiment_id: phase1-baselines\n", encoding="utf-8")

    monkeypatch.setattr(experiments.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))

    with pytest.raises(OSError, match="spawn failed"):
        experiments._spawn_supervisor_process(  # noqa: SLF001
            experiment_id="phase1-baselines",
            spec_path=spec_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            python_executable="/usr/bin/python3",
            poll_seconds=30,
        )

    stored = experiments.read_experiment_supervisor_state(local_state=local_state, experiment_id="phase1-baselines")
    assert stored["status"] == "STOPPED"
    assert stored["last_error"] == "spawn failed"


def test_read_experiment_supervisor_state_marks_dead_pid_stopped(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    path = local_state / "experiment_supervisors" / "phase1-baselines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": 4321, "status": "STOPPING", "active_run_ids": ["run-a"]}), encoding="utf-8")
    monkeypatch.setattr(experiments, "_is_local_process_running", lambda pid: False)

    payload = experiments.read_experiment_supervisor_state(local_state=local_state, experiment_id="phase1-baselines")

    assert payload["status"] == "STOPPED"
    assert payload["active_run_ids"] == []
    assert payload["stop_reason"] == "local experiment supervisor pid 4321 is not running"


def test_read_experiment_supervisor_state_marks_stale_heartbeat_stopped(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    path = local_state / "experiment_supervisors" / "phase1-baselines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_heartbeat = (datetime.now(tz=experiments.UTC) - experiments.timedelta(minutes=20)).isoformat()
    path.write_text(
        json.dumps(
            {
                "pid": 4321,
                "status": "RUNNING",
                "heartbeat_at": stale_heartbeat,
                "poll_seconds": 30,
                "active_run_ids": ["run-a"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiments, "_is_local_process_running", lambda pid: True)

    payload = experiments.read_experiment_supervisor_state(local_state=local_state, experiment_id="phase1-baselines")

    assert payload["status"] == "STOPPED"
    assert payload["active_run_ids"] == []
    assert payload["stop_reason"] == "experiment supervisor heartbeat is stale"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "STOPPED"


def test_ensure_supervisor_state_clears_stale_stop_markers_when_restarting(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    path = local_state / "experiment_supervisors" / "phase1-baselines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "STOPPED", "completed_at": "2026-04-01T00:00:00+00:00", "stop_reason": "old"}),
        encoding="utf-8",
    )

    state = experiments._ensure_supervisor_state(  # noqa: SLF001
        local_state=local_state,
        experiment_id="phase1-baselines",
        payload={
            "experiment_id": "phase1-baselines",
            "status": "RUNNING",
            "completed_at": None,
            "stop_reason": None,
            "stop_requested": False,
        },
    )

    assert state["completed_at"] is None
    assert state["stop_reason"] is None
    assert state["stop_requested"] is False


def test_stop_experiment_supervisor_stops_active_training_runs(tmp_path: Path, monkeypatch) -> None:
    local_state = tmp_path / "local"
    experiment_id = "phase1-baselines"
    supervisor_path = local_state / "experiment_supervisors" / f"{experiment_id}.json"
    run_root = local_state / "experiments" / experiment_id / "runs"
    supervisor_path.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True)
    supervisor_path.write_text(
        json.dumps({"pid": 4321, "status": "RUNNING", "active_run_ids": ["run-a"]}),
        encoding="utf-8",
    )
    (run_root / "run-a.json").write_text(
        json.dumps({"run_id": "run-a", "phase": 1, "target": "local", "runtime_name": "runtime-a"}),
        encoding="utf-8",
    )
    killed: list[int] = []
    stopped: list[dict[str, object]] = []
    monkeypatch.setattr(experiments, "_is_local_process_running", lambda pid: True)
    monkeypatch.setattr(experiments.os, "kill", lambda pid, signal: killed.append(pid))

    def fake_stop_training_process(**kwargs):  # noqa: ANN001
        stopped.append(kwargs)
        return {"stopped": True}

    monkeypatch.setattr(experiments, "stop_training_process", fake_stop_training_process)

    payload = experiments.stop_experiment_supervisor(
        local_state=local_state,
        experiment_id=experiment_id,
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "nas",
        targets_config_path=tmp_path / "repo" / "configs" / "node.yml",
        python_executable="/usr/bin/python3",
    )

    assert killed == [4321]
    assert payload["stopped_training_runs"] == [{"run_id": "run-a", "stopped": True}]
    assert stopped[0]["runtime_name"] == "runtime-a"


def _matrix_size(matrix: dict[str, list[object]]) -> int:
    size = 1
    for values in matrix.values():
        size *= max(1, len(values))
    return size
