from __future__ import annotations

import json
from pathlib import Path

import yaml

import trademl.experiments as experiments


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
    (tmp_path / "phase1.yml").write_text(yaml.safe_dump({"experiment_id": "phase1-baselines", "phase": 1, "target": "workstation-remote"}, sort_keys=False), encoding="utf-8")
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

    assert payload["proposal"]["recommended_experiment_id"] == "phase1-baselines-next"
    assert Path(payload["proposal"]["spec_path"]).exists()
