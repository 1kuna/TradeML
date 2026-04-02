from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml
import sqlite3

import trademl.dashboard.controller as dashboard_controller
from trademl.data_node.capabilities import default_macro_series
from trademl.dashboard.controller import (
    advance_collection_stage,
    collect_dashboard_snapshot,
    join_cluster,
    replan_coverage,
    reset_worker,
    run_vendor_audit,
    persist_node_settings,
    resolve_node_settings,
    start_node,
    stop_node,
    uninstall_worker,
    update_worker,
)
from trademl.data_node.db import DataNodeDB


def test_persist_node_settings_updates_config_env_stage_and_fstab(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    stage_path = workspace / "stage.yml"
    fstab_path = workspace / "fstab"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": "/mnt/trademl",
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
    stage_path.write_text(
        yaml.safe_dump(
            {
                "current": 0,
                "symbols": ["AAPL", "MSFT"],
                "years": 5,
                "schedule": {"collection_time_et": "16:30", "maintenance_hour_local": 2},
                "nas": {"share": "//nas/trademl", "mount": "/mnt/trademl"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    result = persist_node_settings(
        settings,
        nas_share="//192.168.1.20/trademl",
        nas_mount=str(tmp_path / "nas"),
        collection_time_et="17:05",
        maintenance_hour_local=4,
        fstab_path=fstab_path,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = yaml.safe_load(stage_path.read_text(encoding="utf-8"))
    env_text = (workspace / ".env").read_text(encoding="utf-8")

    assert config["node"]["nas_share"] == "//192.168.1.20/trademl"
    assert config["node"]["collection_time_et"] == "17:05"
    assert stage["schedule"]["maintenance_hour_local"] == 4
    assert "NAS_SHARE=//192.168.1.20/trademl" in env_text
    assert Path(result["fstab_path"]).exists()
    assert "//192.168.1.20/trademl" in fstab_path.read_text(encoding="utf-8")


def test_resolve_node_settings_autodetects_populated_worker_workspace(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    workspace = home / "trademl-node"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(tmp_path / "nas"),
                    "nas_share": "//127.0.0.1/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        f"LOCAL_STATE={workspace / 'control'}\nNAS_MOUNT={tmp_path / 'nas'}\nNAS_SHARE=//127.0.0.1/trademl\n",
        encoding="utf-8",
    )
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL"], "years": 1}, sort_keys=False),
        encoding="utf-8",
    )
    (workspace / "control").mkdir(parents=True, exist_ok=True)
    (workspace / "control" / "node_runtime.json").write_text('{"pid": 123, "running": true}', encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRADEML_WORKSPACE_ROOT", raising=False)

    settings = resolve_node_settings()

    assert settings.workspace_root == workspace
    assert settings.config_path == config_path
    assert settings.env_path == workspace / ".env"


def test_collect_dashboard_snapshot_reads_queue_qc_and_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nas_mount = tmp_path / "nas"
    local_state = workspace / "control"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas_mount),
                    "nas_share": "//127.0.0.1/trademl",
                    "local_state": str(local_state),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL", "MSFT"], "years": 1}, sort_keys=False),
        encoding="utf-8",
    )
    db = DataNodeDB(local_state / "node.sqlite")
    db.enqueue_task("equities_eod", "AAPL", "2025-01-01", "2025-01-02", "GAP", 1)
    leased = db.lease_next_task()
    assert leased is not None
    db.mark_task_failed(leased.id, "test", backoff_minutes=1)
    db.update_partition_status("alpaca", "equities_eod", "2025-01-02", "GREEN", 2, 2, "OK")

    qc_root = nas_mount / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"source": "alpaca", "dataset": "equities_eod", "date": "2025-01-02", "status": "GREEN"}]).to_parquet(
        qc_root / "partition_status.parquet",
        index=False,
    )
    pd.DataFrame([{"symbol": "AAPL", "massive_close": 1.0, "finnhub_close": 1.0}]).to_parquet(
        qc_root / "price_checks_2025-01-02.parquet",
        index=False,
    )
    for root in [nas_mount / "data" / "raw" / "equities_bars", nas_mount / "data" / "curated" / "equities_ohlcv_adj"]:
        partition = root / "date=2025-01-02"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(partition / "data.parquet", index=False)
    macro_partition = nas_mount / "data" / "raw" / "macros_fred" / "series=DGS10"
    macro_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2025-01-02", "value": 4.5}]).to_parquet(macro_partition / "data.parquet", index=False)
    reference_root = nas_mount / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(reference_root / "universe.parquet", index=False)

    local_state.mkdir(parents=True, exist_ok=True)
    (local_state / "node_runtime.json").write_text(
        '{"pid": %d, "started_at": "2026-03-31T20:00:00+00:00"}' % os.getpid(),
        encoding="utf-8",
    )
    (local_state / "logs").mkdir(parents=True, exist_ok=True)
    (local_state / "logs" / "node.log").write_text("node booted\ncycle done\n", encoding="utf-8")

    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    snapshot = collect_dashboard_snapshot(settings)

    assert snapshot["runtime"]["running"] is True
    assert snapshot["queue_counts"]["FAILED"] == 1
    assert snapshot["partition_summary"]["counts"]["GREEN"] == 1
    assert snapshot["raw_partitions"] == 1
    assert snapshot["raw_datapoints"] == 1
    assert snapshot["curated_datapoints"] == 1
    assert snapshot["latest_raw_date"] == "2025-01-02"
    assert snapshot["reference_file_count"] == 1
    assert snapshot["macro_series_count"] == 1
    assert snapshot["price_check_count"] == 1
    assert snapshot["data_readiness"]["missing"] == ["equities EOD backfill", "macro vintages"]
    assert snapshot["training_readiness"]["phase1"]["ready"] is False
    assert "corp_actions" in snapshot["training_readiness"]["phase1"]["blockers"]
    assert snapshot["vendor_attempt_summary"]["counts"] == {}
    assert "cycle done" in snapshot["log_tail"]


def test_run_vendor_audit_and_replan_coverage_persist_outputs(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    nas_mount = tmp_path / "nas"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "node.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas_mount),
                    "nas_share": "//127.0.0.1/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        f"NAS_MOUNT={nas_mount}\nNAS_SHARE=//127.0.0.1/trademl\nLOCAL_STATE={workspace / 'control'}\n",
        encoding="utf-8",
    )
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 1, "symbols": ["AAPL", "MSFT"], "years": 5}, sort_keys=False),
        encoding="utf-8",
    )
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)

    monkeypatch.setattr(
        dashboard_controller,
        "run_capability_audit",
        lambda **kwargs: {"checked_at": "2026-04-02T00:00:00+00:00", "summary": {"live_status": {"live_verified": 1}}},
    )
    monkeypatch.setattr(
        dashboard_controller,
        "_connectors_from_settings",
        lambda settings: {"alpaca": object()},
    )

    audit = run_vendor_audit(settings)
    plan = replan_coverage(settings)

    assert audit["summary"]["live_status"]["live_verified"] == 1
    assert (nas_mount / "control" / "cluster" / "state" / "coverage_plan.json").exists()
    assert plan["task_count"] >= 0


def test_start_and_stop_node_manage_runtime_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)

    runtime = start_node(
        settings,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
    )
    assert runtime["running"] is True
    assert Path(runtime["log_path"]).exists()

    stopped = stop_node(settings)
    assert stopped["running"] is False
    assert "last_pid" in stopped


def test_start_node_rotates_existing_log(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.write_text("old crash\n", encoding="utf-8")

    runtime = start_node(
        settings,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
    )

    rotated_logs = list(settings.log_path.parent.glob("node_*.log"))
    assert runtime["running"] is True
    assert settings.log_path.exists()
    assert rotated_logs
    assert rotated_logs[0].read_text(encoding="utf-8") == "old crash\n"

    stop_node(settings)


def test_start_node_persists_cluster_passphrase(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)

    runtime = start_node(
        settings,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        passphrase="pw123",
    )

    env_text = (workspace / ".env").read_text(encoding="utf-8")
    assert "TRADEML_CLUSTER_PASSPHRASE=pw123" in env_text


def test_advance_collection_stage_updates_stage_manifest_and_queue(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    nas_mount = tmp_path / "nas"
    local_state = workspace / "control"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas_mount),
                    "nas_share": "//127.0.0.1/trademl",
                    "local_state": str(local_state),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                    "worker_id": "worker-a",
                },
                "stage": {
                    "current": 0,
                    "stage_0": {"symbols": 100, "eod_years": 5},
                    "stage_1": {"symbols": 500, "eod_years": 10},
                },
                "vendors": {"alpaca": {"rpm": 150, "daily_cap": 10000}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        "\n".join(
            [
                f"NAS_MOUNT={nas_mount}",
                "NAS_SHARE=//127.0.0.1/trademl",
                f"LOCAL_STATE={local_state}",
                "EDGE_NODE_ID=worker-a",
                "ALPACA_API_KEY=test-key",
                "ALPACA_API_SECRET=test-secret",
                "TRADEML_CLUSTER_PASSPHRASE=pass123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL", "MSFT"], "years": 5}, sort_keys=False),
        encoding="utf-8",
    )
    db = DataNodeDB(local_state / "node.sqlite")
    reference_root = nas_mount / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
            {"symbol": "BBB", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
            {"symbol": "CCC", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
        ]
    ).to_parquet(reference_root / "listing_history.parquet", index=False)

    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    join_cluster(settings, passphrase="pass123")
    monkeypatch.setattr(dashboard_controller, "build_stage1_universe", lambda **kwargs: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(
        dashboard_controller,
        "_probe_available_history_years",
        lambda **kwargs: {"requested_years": 10, "effective_years": 5, "probe_symbol_count": 3},
    )

    result = advance_collection_stage(settings, target_stage=1, symbol_count=3, years=10, passphrase="pass123")

    stage = yaml.safe_load((workspace / "stage.yml").read_text(encoding="utf-8"))
    assert result["stage"]["current"] == 1
    assert stage["symbols"] == ["AAA", "BBB", "CCC"]
    assert stage["years"] == 5
    assert result["history_probe"]["effective_years"] == 5
    queued = sqlite3.connect(local_state / "node.sqlite").execute("SELECT COUNT(*) FROM backfill_queue").fetchone()[0]
    assert queued >= 3


def test_advance_collection_stage_prefers_tiingo_for_history_probe(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    nas_mount = tmp_path / "nas"
    local_state = workspace / "control"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas_mount),
                    "nas_share": "//127.0.0.1/trademl",
                    "local_state": str(local_state),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                    "worker_id": "worker-a",
                },
                "stage": {
                    "current": 0,
                    "stage_0": {"symbols": 100, "eod_years": 5},
                    "stage_1": {"symbols": 500, "eod_years": 10},
                },
                "vendors": {
                    "alpaca": {"rpm": 150, "daily_cap": 10000},
                    "tiingo": {"rpm": 40, "daily_cap": 400},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / ".env").write_text(
        "\n".join(
            [
                f"NAS_MOUNT={nas_mount}",
                "NAS_SHARE=//127.0.0.1/trademl",
                f"LOCAL_STATE={local_state}",
                "EDGE_NODE_ID=worker-a",
                "ALPACA_API_KEY=test-key",
                "ALPACA_API_SECRET=test-secret",
                "TIINGO_API_KEY=tiingo-key",
                "TRADEML_CLUSTER_PASSPHRASE=pass123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL", "MSFT"], "years": 5}, sort_keys=False),
        encoding="utf-8",
    )
    reference_root = nas_mount / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
            {"symbol": "BBB", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
            {"symbol": "CCC", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2000-01-01", "delist_date": None},
        ]
    ).to_parquet(reference_root / "listing_history.parquet", index=False)

    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    join_cluster(settings, passphrase="pass123")
    monkeypatch.setattr(dashboard_controller, "build_stage1_universe", lambda **kwargs: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(
        dashboard_controller,
        "_probe_available_history_years",
        lambda **kwargs: {
            "requested_years": 10,
            "effective_years": 10,
            "probe_symbol_count": 3,
            "connector": kwargs["connector"].vendor_name,
        },
    )

    result = advance_collection_stage(settings, target_stage=1, symbol_count=3, years=10, passphrase="pass123")

    assert result["history_probe"]["connector"] == "tiingo"


def test_join_cluster_persists_cluster_passphrase(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)

    class FakeCoordinator:
        worker_id = "worker-a"

        def ensure_cluster_ready(self, *, passphrase=None):  # noqa: ANN001
            return {"ok": True, "passphrase": passphrase}

        def rebuild_local_state(self, *, local_db_path):  # noqa: ANN001
            return {"db": str(local_db_path)}

    monkeypatch.setattr("trademl.dashboard.controller._coordinator", lambda _settings: FakeCoordinator())

    joined = join_cluster(settings, passphrase="pw123")

    env_text = (workspace / ".env").read_text(encoding="utf-8")
    assert "TRADEML_CLUSTER_PASSPHRASE=pw123" in env_text
    assert joined["manifest"]["passphrase"] == "pw123"


def test_collect_dashboard_snapshot_tolerates_locked_queue_db(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    (workspace / "stage.yml").write_text(
        yaml.safe_dump({"current": 0, "symbols": ["AAPL"], "years": 1}, sort_keys=False),
        encoding="utf-8",
    )

    real_connect = dashboard_controller.sqlite3.connect

    def locked_connect(*args, **kwargs):
        if args and str(args[0]).endswith("node.sqlite"):
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dashboard_controller.sqlite3, "connect", locked_connect)

    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    snapshot = collect_dashboard_snapshot(settings)

    assert snapshot["queue_counts"]["PENDING"] == 0
    assert snapshot["queue_counts"]["FAILED"] == 0


def test_update_worker_refreshes_wrapper_and_reports_paths(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr("trademl.dashboard.controller.collect_dashboard_snapshot", lambda _settings: {"runtime": {"running": False}})
    commands: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_run(command, check=True, **_kwargs):  # noqa: ANN001
        commands.append([str(part) for part in command])
        return _Result()

    monkeypatch.setattr("trademl.dashboard.controller.subprocess.run", fake_run)
    monkeypatch.setattr("trademl.dashboard.controller.install_service", lambda _settings: {"service_path": str(workspace / "svc")})

    result = update_worker(settings)

    assert result["wrapper_path"].endswith("/.local/bin/trademl")
    assert Path(result["wrapper_path"]).exists()
    assert any("venv" in " ".join(command) for command in commands)
    assert any("install" in command for command in commands)


def test_reset_and_uninstall_worker_manage_local_artifacts(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    config_path = workspace / "node.yml"
    workspace.mkdir(parents=True, exist_ok=True)
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
    (workspace / ".env").write_text("NAS_MOUNT=/tmp/nas\n", encoding="utf-8")
    (workspace / "stage.yml").write_text("current: 0\n", encoding="utf-8")
    (workspace / "control").mkdir(parents=True, exist_ok=True)
    settings = resolve_node_settings(workspace_root=workspace, config_path=config_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr("trademl.dashboard.controller.collect_dashboard_snapshot", lambda _settings: {"runtime": {"running": False}})
    monkeypatch.setattr("trademl.dashboard.controller.stop_node", lambda _settings: {"running": False})
    monkeypatch.setattr("trademl.dashboard.controller.leave_cluster", lambda _settings: {"ok": True})
    monkeypatch.setattr(
        "trademl.dashboard.controller.join_cluster",
        lambda _settings, passphrase=None: {"worker_id": _settings.worker_id, "passphrase_used": passphrase},
    )

    reset_result = reset_worker(settings, passphrase="pass123")

    assert reset_result["joined"]["worker_id"] == settings.worker_id
    assert not (workspace / ".env").exists()

    wrapper = Path.home() / ".local" / "bin" / "trademl"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("test", encoding="utf-8")
    uninstall_result = uninstall_worker(settings)

    assert str(workspace) in uninstall_result["removed_paths"]
    assert not workspace.exists()
