from __future__ import annotations

import json
from pathlib import Path

import yaml

from trademl import cli


def test_dashboard_cli_builds_streamlit_command(tmp_path: Path, monkeypatch) -> None:
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
    assert "streamlit" in seen["command"]
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
