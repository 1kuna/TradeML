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
    monkeypatch.setattr(experiments, "_stage_symbol_count", lambda root: 500)
    monkeypatch.setattr(experiments, "recommended_training_cutoff", lambda **kwargs: {"date": "2026-04-02"})

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
