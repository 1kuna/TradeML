from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from trademl.dashboard.controller import (
    collect_dashboard_snapshot,
    persist_node_settings,
    resolve_node_settings,
    start_node,
    stop_node,
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
    for root in [nas_mount / "data" / "raw" / "equities_bars", nas_mount / "data" / "curated" / "equities_ohlcv_adj"]:
        partition = root / "date=2025-01-02"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"symbol": "AAPL"}]).to_parquet(partition / "data.parquet", index=False)

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
    assert snapshot["latest_raw_date"] == "2025-01-02"
    assert "cycle done" in snapshot["log_tail"]


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
