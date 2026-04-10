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


def test_train_start_and_status_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "nas"
    local_state = tmp_path / "train-state"
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "launch_training_process",
        lambda **kwargs: {"action": "start", "phase": kwargs["phase"], "data_root": str(kwargs["data_root"])},
    )
    monkeypatch.setattr(
        cli,
        "read_training_runtime",
        lambda *, path=None, local_state=None, phase=None: {"status": "running", "phase": phase or 1, "path": str(path) if path else str(local_state)},
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
                "--phase",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["shared"]["status"] == "running"
