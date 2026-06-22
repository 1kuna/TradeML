from __future__ import annotations

import json
import os
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


def test_fleet_health_cli_prints_current_state(tmp_path: Path, monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(cli, "collect_dashboard_live_snapshot", lambda settings: {"runtime": {"running": True}, "collection_status": {}, "health": {}})
    data_root = tmp_path / "health-root"

    def fake_collect_fleet_health(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["data_root"] == data_root
        return {"verdict": "OK", "current_state": {"action": "OK"}}

    monkeypatch.setattr(cli, "collect_fleet_health", fake_collect_fleet_health)

    rc = cli.main(["fleet", "--workspace-root", str(workspace), "--config", str(config_path), "health", "--data-root", str(data_root)])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "OK"


def test_fleet_cli_loads_env_file_for_password_env(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    env_path = workspace / ".env"
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
    env_path.write_text(
        "TRADEML_PI_PASSWORD=secret\n"
        "TRADEML_PI_TAILSCALE_HOST=100.76.4.69\n"
        "TRADEML_PI_USER=zach\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TRADEML_PI_PASSWORD", raising=False)
    monkeypatch.setattr(cli, "collect_dashboard_live_snapshot", lambda settings: {"runtime": {"running": True}, "collection_status": {}, "health": {}})

    def fake_collect_fleet_health(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["pi"]["host"] == "100.76.4.69"
        assert kwargs["pi"]["user"] == "zach"
        assert kwargs["pi"]["password_env"] == "TRADEML_PI_PASSWORD"
        assert os.environ["TRADEML_PI_PASSWORD"] == "secret"
        return {"verdict": "OK", "current_state": {"action": "OK"}}

    monkeypatch.setattr(cli, "collect_fleet_health", fake_collect_fleet_health)

    rc = cli.main(
        [
            "fleet",
            "--workspace-root",
            str(workspace),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "health",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "OK"


def test_fleet_observability_cli_writes_snapshot(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    data_root = tmp_path / "nas"
    workspace.mkdir(parents=True, exist_ok=True)
    data_root.mkdir()
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(data_root),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                },
                "collection": {"saturation": {"target_utilization": 0.98}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "collect_dashboard_live_snapshot",
        lambda settings: {
            "runtime": {"running": True},
            "collection_status": {},
            "latest_raw_date": "2026-04-29",
            "latest_curated_date": "2026-04-29",
            "health": {"research_program_summary": {"status": "RUNNING"}},
        },
    )

    rc = cli.main(
        [
            "fleet",
            "--workspace-root",
            str(workspace),
            "--config",
            str(config_path),
            "observability",
            "--data-root",
            str(data_root),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["research"]["status"] == "RUNNING"
    assert (data_root / "control" / "cluster" / "state" / "autopilot" / "observability" / "latest.json").exists()


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


def test_fleet_watchdog_cli_dispatches_without_leaking_password(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    env_path = workspace / ".env"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {"nas_mount": str(tmp_path / "nas"), "nas_share": "//nas/trademl", "local_state": str(workspace / "control")},
                "fleet": {"watchdog": {"repeated_issue_count": 2}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env_path.write_text("TRADEML_PI_PASSWORD=super-secret\n", encoding="utf-8")
    monkeypatch.setattr(cli, "collect_dashboard_live_snapshot", lambda settings: {"runtime": {"running": True}})
    monkeypatch.setattr(
        cli,
        "run_fleet_watchdog_once",
        lambda **kwargs: {"verdict": "OK", "heal_requested": kwargs["heal"], "pi": kwargs["pi"]},
    )

    rc = cli.main(
        [
            "fleet",
            "--workspace-root",
            str(workspace),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "watchdog",
            "--data-root",
            str(tmp_path / "nas"),
            "--pi-host",
            "100.76.4.69",
            "--heal",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["heal_requested"] is True
    assert payload["pi"]["host"] == "100.76.4.69"
    assert "super-secret" not in json.dumps(payload)


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


def test_compact_archives_cli_uses_node_nas_mount(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    nas_root = tmp_path / "nas"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas_root),
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

    def fake_compact_archive_partitions(**kwargs):  # noqa: ANN003, ANN202
        seen.update(kwargs)
        return {"action": "compact", "data_root": str(kwargs["data_root"])}

    monkeypatch.setattr(cli, "compact_archive_partitions", fake_compact_archive_partitions)

    assert (
        cli.main(
            [
                "node",
                "--workspace-root",
                str(workspace),
                "--config",
                str(config_path),
                "compact-archives",
                "--dataset",
                "ticker_news",
                "--max-partitions",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"action": "compact", "data_root": str(nas_root)}
    assert seen["data_root"] == nas_root
    assert seen["datasets"] == ["ticker_news"]
    assert seen["max_partitions"] == 2
    assert seen["dry_run"] is True


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
    monkeypatch.setattr(cli, "supervise_experiment", lambda spec_path, **kwargs: {"action": "supervise", "spec": str(spec_path), "detach": kwargs.get("detach", False)})
    monkeypatch.setattr(cli, "run_experiment_until_idle", lambda spec_path, **kwargs: {"action": "run-until-idle", "spec": str(spec_path)})
    monkeypatch.setattr(cli, "experiment_status", lambda experiment_id, **kwargs: {"action": "status", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "pause_experiment_supervisor", lambda local_state, experiment_id: {"action": "pause", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "resume_experiment_supervisor", lambda experiment_id, **kwargs: {"action": "resume", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "stop_experiment_supervisor", lambda local_state, experiment_id, **kwargs: {"action": "stop", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "evaluate_experiment", lambda experiment_id, **kwargs: {"action": "evaluate", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "backtest_experiment_survivors", lambda experiment_id, **kwargs: {"action": "backtest", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "propose_next_experiment_family", lambda experiment_id, **kwargs: {"action": "propose-next", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "compare_experiment", lambda experiment_id, **kwargs: {"action": "compare", "experiment_id": experiment_id})
    monkeypatch.setattr(cli, "render_experiment_report", lambda experiment_id, **kwargs: {"action": "report", "experiment_id": experiment_id})

    assert cli.main(["experiments", "plan", "--spec", str(spec_path)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "plan"
    assert cli.main(["experiments", "launch", "--spec", str(spec_path)]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "launch"
    assert cli.main(["experiments", "supervise", "--spec", str(spec_path), "--detach"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "supervise"
    assert cli.main(["experiments", "run-until-idle", "--spec", str(spec_path), "--poll-seconds", "11"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "run-until-idle"
    assert cli.main(["experiments", "status", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "status"
    assert cli.main(["experiments", "pause", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "pause"
    assert cli.main(["experiments", "resume", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "resume"
    assert cli.main(["experiments", "stop", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "stop"
    assert cli.main(["experiments", "evaluate", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "evaluate"
    assert cli.main(["experiments", "backtest-survivors", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "backtest"
    assert cli.main(["experiments", "propose-next", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "propose-next"
    assert cli.main(["experiments", "compare", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "compare"
    assert cli.main(["experiments", "report", "--experiment", "phase1"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "report"


def test_research_cli_dispatch(tmp_path: Path, monkeypatch, capsys) -> None:
    program_path = tmp_path / "program.yml"
    program_path.write_text("program_id: perpetual\nphase_order: [1]\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("ALPACA_API_KEY=paper-key\nALPACA_API_SECRET=paper-secret\n", encoding="utf-8")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.setattr(cli, "start_research_program", lambda program_path, **kwargs: {"action": "start", "program": str(program_path), "detach": kwargs.get("detach", False)})
    monkeypatch.setattr(cli, "run_research_canary", lambda program_path, **kwargs: {"action": "canary", "program": str(program_path), "detach": kwargs.get("detach", False)})
    monkeypatch.setattr(
        cli,
        "run_feature_version_canary_batch",
        lambda program_path, **kwargs: {
            "action": "feature-canary",
            "program": str(program_path),
            "feature_versions": kwargs.get("feature_versions"),
            "label_horizon": kwargs.get("label_horizon"),
        },
    )
    monkeypatch.setattr(cli, "run_and_persist_paper_account_smoke", lambda program_path, local_state: {"action": "paper-smoke", "api_key": os.getenv("ALPACA_API_KEY"), "program": str(program_path), "local_state": str(local_state)})
    monkeypatch.setattr(cli, "submit_paper_orders", lambda payloads_path, policy: {"action": "paper-submit", "payloads": str(payloads_path), "enabled": policy.get("enabled")})
    monkeypatch.setattr(
        cli,
        "run_form4_fixture_gate_from_env",
        lambda **kwargs: {
            "action": "form4-fixture-gate",
            "data_root": str(kwargs["data_root"]),
            "limit": kwargs["limit"],
            "user_agent": kwargs["user_agent"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_candidate_curation",
        lambda data_root: {
            "action": "form4-candidates",
            "data_root": str(data_root),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_ingest_from_env",
        lambda **kwargs: {
            "action": "form4-ingest",
            "data_root": str(kwargs["data_root"]),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "limit": kwargs["limit"],
            "max_retrieval_attempts": kwargs["max_retrieval_attempts"],
            "rate_limit_pause_seconds": kwargs["rate_limit_pause_seconds"],
            "use_cache": kwargs["use_cache"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_market_backfill_from_env",
        lambda **kwargs: {
            "action": "form4-market-backfill",
            "data_root": str(kwargs["data_root"]),
            "horizons": list(kwargs["horizons"]),
            "limit_events": kwargs["limit_events"],
            "include_controls": kwargs["include_controls"],
            "max_fetch_attempts": kwargs["max_fetch_attempts"],
            "rate_limit_pause_seconds": kwargs["rate_limit_pause_seconds"],
            "daily_symbol_batch_size": kwargs["daily_symbol_batch_size"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_label_curation",
        lambda **kwargs: {
            "action": "form4-labels",
            "data_root": str(kwargs["data_root"]),
            "horizons": kwargs["horizons"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
            "market_data_roots": [str(item) for item in kwargs["market_data_roots"]],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_event_study",
        lambda **kwargs: {
            "action": "form4-event-study",
            "data_root": str(kwargs["data_root"]),
            "primary_horizon": kwargs["primary_horizon"],
            "min_historical_sample": kwargs["min_historical_sample"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_form4_rework_study",
        lambda data_root: {
            "action": "form4-rework-study",
            "data_root": str(data_root),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_candidate_curation",
        lambda **kwargs: {
            "action": "sec8k-candidates",
            "data_root": str(kwargs["data_root"]),
            "filings_path": str(kwargs["filings_path"]),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_ingest_from_env",
        lambda **kwargs: {
            "action": "sec8k-ingest",
            "data_root": str(kwargs["data_root"]),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "limit": kwargs["limit"],
            "max_retrieval_attempts": kwargs["max_retrieval_attempts"],
            "rate_limit_pause_seconds": kwargs["rate_limit_pause_seconds"],
            "use_cache": kwargs["use_cache"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_market_backfill_from_env",
        lambda **kwargs: {
            "action": "sec8k-market-backfill",
            "data_root": str(kwargs["data_root"]),
            "horizons": list(kwargs["horizons"]),
            "limit_events": kwargs["limit_events"],
            "include_timestamp_placebo": kwargs["include_timestamp_placebo"],
            "max_fetch_attempts": kwargs["max_fetch_attempts"],
            "rate_limit_pause_seconds": kwargs["rate_limit_pause_seconds"],
            "daily_symbol_batch_size": kwargs["daily_symbol_batch_size"],
            "candidate_source": kwargs["candidate_source"],
            "target_items": list(kwargs["target_items"]),
            "accepted_from": kwargs["accepted_from"],
            "accepted_to": kwargs["accepted_to"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_event_study",
        lambda **kwargs: {
            "action": "sec8k-event-study",
            "data_root": str(kwargs["data_root"]),
            "primary_horizon": kwargs["primary_horizon"],
            "horizons": kwargs["horizons"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
            "market_data_roots": [str(item) for item in kwargs["market_data_roots"]],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_research_decision",
        lambda **kwargs: {
            "action": "sec8k-decision",
            "data_root": str(kwargs["data_root"]),
            "min_labeled_events": kwargs["min_labeled_events"],
            "min_family_events": kwargs["min_family_events"],
            "min_mean_abret": kwargs["min_mean_abret"],
            "min_control_separation": kwargs["min_control_separation"],
            "max_top5_abs_contribution": kwargs["max_top5_abs_contribution"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_coverage_audit",
        lambda **kwargs: {
            "action": "sec8k-coverage-audit",
            "data_root": str(kwargs["data_root"]),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "target_items": kwargs["target_items"],
            "fallback_target_items": kwargs["fallback_target_items"],
            "horizons": kwargs["horizons"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec8k_coverage_expand",
        lambda **kwargs: {
            "action": "sec8k-coverage-expand",
            "data_root": str(kwargs["data_root"]),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "target_items": kwargs["target_items"],
            "fallback_target_items": kwargs["fallback_target_items"],
            "limit_per_month": kwargs["limit_per_month"],
            "use_cache": kwargs["use_cache"],
            "rebuild_candidates": kwargs["rebuild_candidates"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_fixture_gate",
        lambda **kwargs: {
            "action": "sec-event-semantic-gate",
            "data_root": str(kwargs["data_root"]),
            "model": kwargs["model"],
            "base_url": kwargs["base_url"],
            "timeout_seconds": kwargs["timeout_seconds"],
            "response_format_mode": kwargs["response_format_mode"],
            "batch_size": kwargs["batch_size"],
            "limit": kwargs["limit"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_classification",
        lambda **kwargs: {
            "action": "sec-event-semantic-classify",
            "data_root": str(kwargs["data_root"]),
            "model": kwargs["model"],
            "base_url": kwargs["base_url"],
            "timeout_seconds": kwargs["timeout_seconds"],
            "response_format_mode": kwargs["response_format_mode"],
            "batch_size": kwargs["batch_size"],
            "limit": kwargs["limit"],
            "max_snippet_chars": kwargs["max_snippet_chars"],
            "routing_mode": kwargs["routing_mode"],
            "target_items": kwargs["target_items"],
            "accepted_from": kwargs["accepted_from"],
            "accepted_to": kwargs["accepted_to"],
            "snippet_kind": kwargs["snippet_kind"],
            "labelability_mode": kwargs["labelability_mode"],
            "resume": kwargs["resume"],
            "checkpoint_path": str(kwargs["checkpoint_path"]),
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_labelability_audit",
        lambda **kwargs: {
            "action": "sec-event-semantic-labelability-audit",
            "data_root": str(kwargs["data_root"]),
            "routing_mode": kwargs["routing_mode"],
            "target_items": kwargs["target_items"],
            "accepted_from": kwargs["accepted_from"],
            "accepted_to": kwargs["accepted_to"],
            "snippet_kind": kwargs["snippet_kind"],
            "horizons": kwargs["horizons"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_scaled_gate",
        lambda **kwargs: {
            "action": "sec-event-semantic-scaled-gate",
            "data_root": str(kwargs["data_root"]),
            "model": kwargs["model"],
            "base_url": kwargs["base_url"],
            "timeout_seconds": kwargs["timeout_seconds"],
            "response_format_mode": kwargs["response_format_mode"],
            "batch_size": kwargs["batch_size"],
            "target_items": kwargs["target_items"],
            "fallback_target_items": kwargs["fallback_target_items"],
            "years": kwargs["years"],
            "max_snippets": kwargs["max_snippets"],
            "resume": kwargs["resume"],
            "primary_horizon": kwargs["primary_horizon"],
            "min_sample": kwargs["min_sample"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_coverage_gate",
        lambda **kwargs: {
            "action": "sec-event-semantic-coverage-gate",
            "data_root": str(kwargs["data_root"]),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "model": kwargs["model"],
            "base_url": kwargs["base_url"],
            "timeout_seconds": kwargs["timeout_seconds"],
            "response_format_mode": kwargs["response_format_mode"],
            "batch_size": kwargs["batch_size"],
            "target_items": kwargs["target_items"],
            "fallback_target_items": kwargs["fallback_target_items"],
            "max_snippets": kwargs["max_snippets"],
            "resume": kwargs["resume"],
            "primary_horizon": kwargs["primary_horizon"],
            "min_sample": kwargs["min_sample"],
            "expand_missing_coverage": kwargs["expand_missing_coverage"],
            "repair_market_coverage": kwargs["repair_market_coverage"],
            "limit_per_month": kwargs["limit_per_month"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_sec_event_semantic_study",
        lambda **kwargs: {
            "action": "sec-event-semantic-study",
            "data_root": str(kwargs["data_root"]),
            "primary_horizon": kwargs["primary_horizon"],
            "horizons": kwargs["horizons"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
            "market_data_roots": [str(item) for item in kwargs["market_data_roots"]],
        },
    )
    monkeypatch.setattr(cli, "read_research_program_state", lambda local_state, program_id: {"action": "status", "program_id": program_id, "frontier": {"x": 1}})
    monkeypatch.setattr(cli, "pause_research_program", lambda local_state, program_id: {"action": "pause", "program_id": program_id})
    monkeypatch.setattr(cli, "resume_research_program", lambda program_id, **kwargs: {"action": "resume", "program_id": program_id})
    monkeypatch.setattr(cli, "stop_research_program", lambda local_state, program_id, **kwargs: {"action": "stop", "program_id": program_id})
    monkeypatch.setattr(cli, "latest_research_program_summary", lambda local_state: {"frontier": {"latest": True}})
    monkeypatch.setattr(cli, "write_research_review_packet", lambda program_id, **kwargs: {"action": "review", "program_id": program_id})
    monkeypatch.setattr(cli, "steer_research_program", lambda local_state, program_id, **kwargs: {"action": "steer", "program_id": program_id, "steering": kwargs})
    monkeypatch.setattr(cli, "install_research_launch_agent", lambda **kwargs: {"action": "install-launchd", "label": kwargs["label"], "load": kwargs["load"]})
    monkeypatch.setattr(cli, "launch_agent_status", lambda label: {"action": "launchd-status", "label": label})
    monkeypatch.setattr(cli, "unload_launch_agent", lambda label: {"action": "unload-launchd", "label": label})

    research_base = ["research", "--env-file", str(env_path)]

    assert cli.main([*research_base, "start", "--program", str(program_path), "--detach"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "start"
    assert cli.main([*research_base, "canary", "--program", str(program_path), "--detach"]) == 0
    canary_payload = json.loads(capsys.readouterr().out)
    assert canary_payload["action"] == "canary"
    assert canary_payload["detach"] is True
    assert cli.main([*research_base, "feature-canary", "--program", str(program_path), "--feature-version", "price_liquidity_v1", "--label-horizon", "5"]) == 0
    feature_canary_payload = json.loads(capsys.readouterr().out)
    assert feature_canary_payload["action"] == "feature-canary"
    assert feature_canary_payload["feature_versions"] == ["price_liquidity_v1"]
    assert feature_canary_payload["label_horizon"] == 5
    assert cli.main([*research_base, "paper-smoke", "--program", str(program_path)]) == 0
    paper_smoke_payload = json.loads(capsys.readouterr().out)
    assert paper_smoke_payload["action"] == "paper-smoke"
    assert paper_smoke_payload["api_key"] == "paper-key"
    assert paper_smoke_payload["program"] == str(program_path)
    assert cli.main(["research", "paper-submit", "--program", str(program_path), "--payloads", str(tmp_path / "payloads.json")]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "paper-submit"
    data_root = tmp_path / "nas"
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "form4-fixture-gate",
                "--limit",
                "2",
                "--user-agent",
                "TradeML test@example.com",
            ]
        )
        == 0
    )
    form4_payload = json.loads(capsys.readouterr().out)
    assert form4_payload["action"] == "form4-fixture-gate"
    assert form4_payload["data_root"] == str(data_root)
    assert form4_payload["limit"] == 2
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "form4-ingest",
                "--start-date",
                "2025-04-01",
                "--end-date",
                "2025-04-30",
                "--limit",
                "3",
                "--max-retrieval-attempts",
                "2",
                "--rate-limit-pause-seconds",
                "0.1",
            ]
        )
        == 0
    )
    form4_ingest_payload = json.loads(capsys.readouterr().out)
    assert form4_ingest_payload["action"] == "form4-ingest"
    assert form4_ingest_payload["start_date"] == "2025-04-01"
    assert form4_ingest_payload["limit"] == 3
    assert form4_ingest_payload["max_retrieval_attempts"] == 2
    assert form4_ingest_payload["rate_limit_pause_seconds"] == 0.1
    assert form4_ingest_payload["use_cache"] is True
    assert cli.main([*research_base, "--data-root", str(data_root), "form4-candidates"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "form4-candidates",
        "data_root": str(data_root),
    }
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "form4-market-backfill",
                "--horizon",
                "5",
                "--limit-events",
                "1",
                "--max-fetch-attempts",
                "3",
                "--rate-limit-pause-seconds",
                "0.2",
                "--daily-symbol-batch-size",
                "2",
            ]
        )
        == 0
    )
    form4_backfill_payload = json.loads(capsys.readouterr().out)
    assert form4_backfill_payload["action"] == "form4-market-backfill"
    assert form4_backfill_payload["horizons"] == [5]
    assert form4_backfill_payload["limit_events"] == 1
    assert form4_backfill_payload["include_controls"] is True
    assert form4_backfill_payload["max_fetch_attempts"] == 3
    assert form4_backfill_payload["rate_limit_pause_seconds"] == 0.2
    assert form4_backfill_payload["daily_symbol_batch_size"] == 2
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "form4-labels",
                "--horizon",
                "5",
                "--round-trip-cost-bps",
                "75",
                "--market-data-root",
                str(tmp_path / "market"),
            ]
        )
        == 0
    )
    form4_labels_payload = json.loads(capsys.readouterr().out)
    assert form4_labels_payload["action"] == "form4-labels"
    assert form4_labels_payload["horizons"] == [5]
    assert form4_labels_payload["round_trip_cost_bps"] == 75.0
    assert form4_labels_payload["market_data_roots"] == [str(tmp_path / "market")]
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "form4-event-study",
                "--primary-horizon",
                "5",
                "--min-historical-sample",
                "10",
            ]
        )
        == 0
    )
    form4_event_study_payload = json.loads(capsys.readouterr().out)
    assert form4_event_study_payload["action"] == "form4-event-study"
    assert form4_event_study_payload["min_historical_sample"] == 10
    assert (
        cli.main([*research_base, "--data-root", str(data_root), "form4-rework-study"])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "action": "form4-rework-study",
        "data_root": str(data_root),
    }
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-ingest",
                "--start-date",
                "2025-04-01",
                "--end-date",
                "2025-04-30",
                "--limit",
                "3",
                "--max-retrieval-attempts",
                "2",
                "--rate-limit-pause-seconds",
                "0.1",
            ]
        )
        == 0
    )
    sec8k_ingest_payload = json.loads(capsys.readouterr().out)
    assert sec8k_ingest_payload["action"] == "sec8k-ingest"
    assert sec8k_ingest_payload["start_date"] == "2025-04-01"
    assert sec8k_ingest_payload["limit"] == 3
    assert sec8k_ingest_payload["max_retrieval_attempts"] == 2
    assert sec8k_ingest_payload["use_cache"] is True
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-candidates",
                "--filings-path",
                str(tmp_path / "sec_filings.parquet"),
            ]
        )
        == 0
    )
    sec8k_candidates_payload = json.loads(capsys.readouterr().out)
    assert sec8k_candidates_payload["action"] == "sec8k-candidates"
    assert sec8k_candidates_payload["filings_path"] == str(tmp_path / "sec_filings.parquet")
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-market-backfill",
                "--horizon",
                "5",
                "--limit-events",
                "2",
                "--no-timestamp-placebo",
                "--max-fetch-attempts",
                "3",
                "--rate-limit-pause-seconds",
                "0.2",
                "--daily-symbol-batch-size",
                "2",
                "--candidate-source",
                "sec_event_semantic_candidates",
                "--target-item",
                "3.02",
                "--accepted-from",
                "2025-04-01",
                "--accepted-to",
                "2025-04-30",
            ]
        )
        == 0
    )
    sec8k_backfill_payload = json.loads(capsys.readouterr().out)
    assert sec8k_backfill_payload["action"] == "sec8k-market-backfill"
    assert sec8k_backfill_payload["horizons"] == [5]
    assert sec8k_backfill_payload["limit_events"] == 2
    assert sec8k_backfill_payload["include_timestamp_placebo"] is False
    assert sec8k_backfill_payload["max_fetch_attempts"] == 3
    assert sec8k_backfill_payload["rate_limit_pause_seconds"] == 0.2
    assert sec8k_backfill_payload["daily_symbol_batch_size"] == 2
    assert sec8k_backfill_payload["candidate_source"] == "sec_event_semantic_candidates"
    assert sec8k_backfill_payload["target_items"] == ["3.02"]
    assert sec8k_backfill_payload["accepted_from"] == "2025-04-01"
    assert sec8k_backfill_payload["accepted_to"] == "2025-04-30"
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-event-study",
                "--primary-horizon",
                "5",
                "--horizon",
                "5",
                "--round-trip-cost-bps",
                "75",
                "--market-data-root",
                str(tmp_path / "market"),
            ]
        )
        == 0
    )
    sec8k_study_payload = json.loads(capsys.readouterr().out)
    assert sec8k_study_payload["action"] == "sec8k-event-study"
    assert sec8k_study_payload["horizons"] == [5]
    assert sec8k_study_payload["round_trip_cost_bps"] == 75.0
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-decision",
                "--min-labeled-events",
                "300",
                "--min-family-events",
                "100",
                "--min-mean-abret",
                "0.005",
                "--min-control-separation",
                "0.0075",
                "--max-top5-abs-contribution",
                "0.35",
            ]
        )
        == 0
    )
    sec8k_decision_payload = json.loads(capsys.readouterr().out)
    assert sec8k_decision_payload["action"] == "sec8k-decision"
    assert sec8k_decision_payload["min_labeled_events"] == 300
    assert sec8k_decision_payload["min_family_events"] == 100
    assert sec8k_decision_payload["min_mean_abret"] == 0.005
    assert sec8k_decision_payload["min_control_separation"] == 0.0075
    assert sec8k_decision_payload["max_top5_abs_contribution"] == 0.35
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-coverage-audit",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2025-12-31",
                "--target-item",
                "3.02",
                "--fallback-target-item",
                "1.01",
                "--horizon",
                "5",
            ]
        )
        == 0
    )
    sec8k_coverage_payload = json.loads(capsys.readouterr().out)
    assert sec8k_coverage_payload["action"] == "sec8k-coverage-audit"
    assert sec8k_coverage_payload["target_items"] == ["3.02"]
    assert sec8k_coverage_payload["fallback_target_items"] == ["1.01"]
    assert sec8k_coverage_payload["horizons"] == [5]
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec8k-coverage-expand",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2025-12-31",
                "--target-item",
                "3.02",
                "--fallback-target-item",
                "1.01",
                "--limit-per-month",
                "10",
                "--no-cache",
                "--no-rebuild-candidates",
            ]
        )
        == 0
    )
    sec8k_expand_payload = json.loads(capsys.readouterr().out)
    assert sec8k_expand_payload["action"] == "sec8k-coverage-expand"
    assert sec8k_expand_payload["limit_per_month"] == 10
    assert sec8k_expand_payload["use_cache"] is False
    assert sec8k_expand_payload["rebuild_candidates"] is False
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-gate",
                "--model",
                "qwen3.5-9b-mlx",
                "--base-url",
                "http://127.0.0.1:1234/v1",
                "--timeout-seconds",
                "120",
                "--response-format-mode",
                "prompt_json",
                "--batch-size",
                "1",
                "--limit",
                "2",
            ]
        )
        == 0
    )
    semantic_gate_payload = json.loads(capsys.readouterr().out)
    assert semantic_gate_payload["action"] == "sec-event-semantic-gate"
    assert semantic_gate_payload["model"] == "qwen3.5-9b-mlx"
    assert semantic_gate_payload["base_url"] == "http://127.0.0.1:1234/v1"
    assert semantic_gate_payload["timeout_seconds"] == 120.0
    assert semantic_gate_payload["response_format_mode"] == "prompt_json"
    assert semantic_gate_payload["batch_size"] == 1
    assert semantic_gate_payload["limit"] == 2
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-classify",
                "--model",
                "qwen3.5-9b-mlx",
                "--base-url",
                "http://127.0.0.1:1234/v1",
                "--timeout-seconds",
                "120",
                "--response-format-mode",
                "prompt_json",
                "--batch-size",
                "4",
                "--limit",
                "40",
                "--max-snippet-chars",
                "3000",
                "--routing-mode",
                "targeted",
                "--target-item",
                "4.01",
                "--target-item",
                "2.04",
                "--accepted-from",
                "2025-04-01",
                "--accepted-to",
                "2025-04-30",
                "--snippet-kind",
                "item_section",
                "--labelability-mode",
                "labelable-only",
                "--resume",
                "--checkpoint-path",
                str(tmp_path / "semantic-checkpoint.parquet"),
            ]
        )
        == 0
    )
    semantic_classify_payload = json.loads(capsys.readouterr().out)
    assert semantic_classify_payload["action"] == "sec-event-semantic-classify"
    assert semantic_classify_payload["response_format_mode"] == "prompt_json"
    assert semantic_classify_payload["batch_size"] == 4
    assert semantic_classify_payload["limit"] == 40
    assert semantic_classify_payload["max_snippet_chars"] == 3000
    assert semantic_classify_payload["routing_mode"] == "targeted"
    assert semantic_classify_payload["target_items"] == ["4.01", "2.04"]
    assert semantic_classify_payload["accepted_from"] == "2025-04-01"
    assert semantic_classify_payload["accepted_to"] == "2025-04-30"
    assert semantic_classify_payload["snippet_kind"] == "item_section"
    assert semantic_classify_payload["labelability_mode"] == "labelable-only"
    assert semantic_classify_payload["resume"] is True
    assert semantic_classify_payload["checkpoint_path"] == str(tmp_path / "semantic-checkpoint.parquet")
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-labelability-audit",
                "--routing-mode",
                "targeted",
                "--target-item",
                "3.02",
                "--accepted-from",
                "2025-04-01",
                "--accepted-to",
                "2025-04-30",
                "--horizon",
                "5",
                "--round-trip-cost-bps",
                "50",
            ]
        )
        == 0
    )
    labelability_payload = json.loads(capsys.readouterr().out)
    assert labelability_payload["action"] == "sec-event-semantic-labelability-audit"
    assert labelability_payload["target_items"] == ["3.02"]
    assert labelability_payload["horizons"] == [5]
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-scaled-gate",
                "--model",
                "qwen3.5-9b-mlx",
                "--base-url",
                "http://127.0.0.1:1235/v1",
                "--timeout-seconds",
                "180",
                "--response-format-mode",
                "prompt_json",
                "--batch-size",
                "1",
                "--target-item",
                "3.02",
                "--fallback-target-item",
                "1.01",
                "--year",
                "2025",
                "--max-snippets",
                "10",
                "--no-resume",
                "--primary-horizon",
                "5",
                "--min-sample",
                "20",
            ]
        )
        == 0
    )
    scaled_payload = json.loads(capsys.readouterr().out)
    assert scaled_payload["action"] == "sec-event-semantic-scaled-gate"
    assert scaled_payload["base_url"] == "http://127.0.0.1:1235/v1"
    assert scaled_payload["target_items"] == ["3.02"]
    assert scaled_payload["fallback_target_items"] == ["1.01"]
    assert scaled_payload["years"] == [2025]
    assert scaled_payload["max_snippets"] == 10
    assert scaled_payload["resume"] is False
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-coverage-gate",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2025-12-31",
                "--model",
                "qwen3.5-9b-mlx",
                "--base-url",
                "http://127.0.0.1:1235/v1",
                "--timeout-seconds",
                "180",
                "--response-format-mode",
                "prompt_json",
                "--batch-size",
                "1",
                "--target-item",
                "3.02",
                "--fallback-target-item",
                "1.01",
                "--max-snippets",
                "10",
                "--no-resume",
                "--primary-horizon",
                "5",
                "--min-sample",
                "20",
                "--no-expand",
                "--no-market-repair",
                "--limit-per-month",
                "5",
            ]
        )
        == 0
    )
    coverage_gate_payload = json.loads(capsys.readouterr().out)
    assert coverage_gate_payload["action"] == "sec-event-semantic-coverage-gate"
    assert coverage_gate_payload["base_url"] == "http://127.0.0.1:1235/v1"
    assert coverage_gate_payload["target_items"] == ["3.02"]
    assert coverage_gate_payload["fallback_target_items"] == ["1.01"]
    assert coverage_gate_payload["resume"] is False
    assert coverage_gate_payload["expand_missing_coverage"] is False
    assert coverage_gate_payload["repair_market_coverage"] is False
    assert (
        cli.main(
            [
                *research_base,
                "--data-root",
                str(data_root),
                "sec-event-semantic-study",
                "--primary-horizon",
                "5",
                "--horizon",
                "5",
                "--round-trip-cost-bps",
                "75",
                "--market-data-root",
                str(tmp_path / "market"),
            ]
        )
        == 0
    )
    semantic_study_payload = json.loads(capsys.readouterr().out)
    assert semantic_study_payload["action"] == "sec-event-semantic-study"
    assert semantic_study_payload["horizons"] == [5]
    assert semantic_study_payload["round_trip_cost_bps"] == 75.0
    assert cli.main(["research", "status", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "status"
    assert cli.main(["research", "pause", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "pause"
    assert cli.main(["research", "resume", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "resume"
    assert cli.main(["research", "stop", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "stop"
    assert cli.main(["research", "install-launchd", "--program", str(program_path), "--load"]) == 0
    launchd_payload = json.loads(capsys.readouterr().out)
    assert launchd_payload["action"] == "install-launchd"
    assert launchd_payload["label"] == "com.trademl.research.perpetual"
    assert launchd_payload["load"] is True
    assert cli.main(["research", "launchd-status", "--program", str(program_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"action": "launchd-status", "label": "com.trademl.research.perpetual"}
    assert cli.main(["research", "unload-launchd", "--label", "com.trademl.research.perpetual"]) == 0
    assert json.loads(capsys.readouterr().out) == {"action": "unload-launchd", "label": "com.trademl.research.perpetual"}
    assert cli.main(["research", "frontier", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["x"] == 1
    assert cli.main(["research", "review-packet", "--program-id", "perpetual"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "review"
    assert cli.main(["research", "steer", "--program-id", "perpetual", "--prefer-architecture", "tree_challenger", "--force-pivot"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "steer"
