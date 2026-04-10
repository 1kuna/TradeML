from __future__ import annotations

import json
from pathlib import Path

import yaml

from trademl import cli


def test_dashboard_cli_builds_http_server_command(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    class _Result:
        returncode = 0

    def fake_run(command, check):  # noqa: ANN001
        seen["command"] = command
        seen["check"] = check
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["dashboard", "--workspace-root", str(workspace), "--config", str(config_path), "--no-browser"])

    assert rc == 0
    assert seen["check"] is False
    assert seen["command"][:3] == [cli.sys.executable, "-m", "trademl.dashboard.server"]
    assert "--workspace-root" in seen["command"]


def test_node_status_cli_prints_snapshot_json(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rc = cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "status"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["workspace_root"] == str(workspace)


def test_join_cluster_cli_bootstraps_manifest(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    env_path = workspace / ".env"
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL", "MSFT"], "years": 1}, sort_keys=False),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                },
                "stage": {},
                "vendors": {"alpaca": {"rpm": 150, "daily_cap": 10000}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        f"NAS_MOUNT={tmp_path / 'nas'}\nNAS_SHARE=//nas/trademl\nLOCAL_STATE={workspace / 'control'}\nALPACA_API_KEY=test\n",
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "node",
            "--workspace-root",
            str(workspace),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "join-cluster",
            "--passphrase",
            "pass123",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["worker_id"]
    assert (tmp_path / "nas" / "control" / "cluster" / "manifest.yml").exists()


def test_reset_update_and_uninstall_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "reset_worker", lambda settings, passphrase=None: {"action": "reset", "passphrase": passphrase})
    monkeypatch.setattr(cli, "update_worker", lambda settings: {"action": "update"})
    monkeypatch.setattr(cli, "uninstall_worker", lambda settings: {"action": "uninstall"})

    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "reset", "--passphrase", "pw"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "reset"
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "update"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "update"
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "uninstall"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "uninstall"


def test_audit_and_replan_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "run_vendor_audit", lambda settings: {"action": "audit"})
    monkeypatch.setattr(cli, "replan_coverage", lambda settings: {"action": "replan"})

    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "run-audit"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "audit"
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "replan-coverage"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "replan"


def test_repair_canonical_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "repair_canonical_backlog", lambda settings, trading_date=None: {"action": "repair", "date": trading_date})

    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "repair-canonical", "--date", "2026-04-10"]) == 0
    assert json.loads(capsys.readouterr().out) == {"action": "repair", "date": "2026-04-10"}


def test_node_health_cli_dispatches(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "verify_recent_canonical_dates", lambda settings, **kwargs: {"action": "verify", **kwargs})
    monkeypatch.setattr(cli, "repair_status", lambda settings: {"action": "repair-status"})
    monkeypatch.setattr(cli, "lane_health", lambda settings, dataset="equities_eod": {"action": "lane-health", "dataset": dataset})
    monkeypatch.setattr(cli, "show_leases", lambda settings, family=None: {"action": "show-leases", "family": family})

    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "verify-recent", "--days", "5"]) == 0
    assert json.loads(capsys.readouterr().out)["days"] == 5
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "repair-status"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "repair-status"
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "lane-health", "--dataset", "equities_eod"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "lane-health"
    assert cli.main(["node", "--workspace-root", str(workspace), "--config", str(config_path), "show-leases", "--family", "canonical_repair"]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == "canonical_repair"


def test_train_start_and_status_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "nas"
    local_state = tmp_path / "train-state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "start_training_run",
        lambda settings, *, phase, report_date=None, target=None: {"action": "start", "phase": phase, "target": target},
    )
    monkeypatch.setattr(
        cli,
        "training_runtime_status",
        lambda settings, *, phase, target=None: {"runtime": {"status": "running", "phase": phase}, "target": {"name": target or "local"}},
    )

    assert (
        cli.main(
            [
                "train",
                "--data-root",
                str(data_root),
                "--local-state",
                str(local_state),
                "--env-file",
                str(env_path),
                "start",
                "--target",
                "workstation-remote",
                "--phase",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["action"] == "start"

    assert (
        cli.main(
            [
                "train",
                "--data-root",
                str(data_root),
                "--local-state",
                str(local_state),
                "status",
                "--target",
                "workstation-remote",
                "--phase",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["status"] == "running"


def test_experiments_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text("experiment_id: phase1\n", encoding="utf-8")
    monkeypatch.setattr(cli, "plan_experiment", lambda spec_path, **kwargs: {"action": "plan", "spec": str(spec_path)})
    monkeypatch.setattr(cli, "launch_experiment", lambda spec_path, **kwargs: {"action": "launch", "spec": str(spec_path)})
    monkeypatch.setattr(cli, "experiment_status", lambda experiment_id, **kwargs: {"action": "status", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "compare_experiment", lambda experiment_id, local_state: {"action": "compare", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "render_experiment_report", lambda experiment_id, local_state: {"action": "report", "experiment_id": experiment_id})

    assert cli.main(["experiments", "plan", "--spec", str(spec_path)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "plan"
    assert cli.main(["experiments", "launch", "--spec", str(spec_path)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "launch"
    assert cli.main(["experiments", "status", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "status"
    assert cli.main(["experiments", "compare", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "compare"
    assert cli.main(["experiments", "report", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "report"
