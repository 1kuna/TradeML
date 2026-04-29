from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from trademl.fleet.cluster import (
    ClusterCoordinator,
    SecretBundleError,
    build_shard_map,
    decrypt_secret_bundle,
    encrypt_secret_bundle,
    install_systemd_service,
    render_systemd_unit,
)
from trademl.data_node.db import DataNodeDB


def _seed_workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    workspace = tmp_path / "workspace"
    nas = tmp_path / "nas"
    config_path = workspace / "node.yml"
    env_path = workspace / ".env"
    stage_path = workspace / "stage.yml"
    workspace.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                "current": 0,
                "symbols": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
                "years": 2,
                "schedule": {"collection_time_et": "16:30", "maintenance_hour_local": 2},
                "nas": {"share": "//nas/trademl", "mount": str(nas)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "node": {
                    "nas_mount": str(nas),
                    "nas_share": "//nas/trademl",
                    "local_state": str(workspace / "control"),
                    "collection_time_et": "16:30",
                    "maintenance_hour_local": 2,
                },
                "stage": {"current": 0},
                "collection": {"saturation": {"enabled": True, "target_utilization": 0.98}},
                "vendors": {"alpaca": {"rpm": 150, "daily_cap": 10000}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                f"NAS_MOUNT={nas}",
                "NAS_SHARE=//nas/trademl",
                f"LOCAL_STATE={workspace / 'control'}",
                "ALPACA_API_KEY=test-key",
                "ALPACA_API_SECRET=test-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace, nas, config_path, env_path


def test_encrypt_and_decrypt_secret_bundle(tmp_path: Path) -> None:
    path = tmp_path / "secrets.enc.json"
    encrypt_secret_bundle(path, "pass123", {"ALPACA_API_KEY": "abc", "ALPACA_API_SECRET": "def"})

    decrypted = decrypt_secret_bundle(path, "pass123")

    assert decrypted["ALPACA_API_KEY"] == "abc"
    with pytest.raises(SecretBundleError):
        decrypt_secret_bundle(path, "wrong-pass")


def test_build_shard_map_is_deterministic() -> None:
    symbols = ["MSFT", "AAPL", "NVDA", "META"]
    first = build_shard_map("equities_eod", symbols, 4)
    second = build_shard_map("equities_eod", list(reversed(symbols)), 4)

    assert [spec.symbols for spec in first] == [spec.symbols for spec in second]
    assert sum(len(spec.symbols) for spec in first) == 4


def test_coordinator_bootstrap_and_rebuild_state(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    qc_root = nas / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"source": "alpaca", "dataset": "equities_eod", "date": "2025-01-02", "status": "GREEN", "row_count": 6, "expected_rows": 6}
        ]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)
    raw_partition = nas / "data" / "raw" / "equities_bars" / "date=2025-01-02"
    raw_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL", "close": 100.0}]).to_parquet(raw_partition / "data.parquet", index=False)

    coordinator = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(count)],
    )
    manifest = coordinator.ensure_cluster_ready(passphrase="pass123")
    node_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert manifest["collection"]["saturation"]["enabled"] is True
    assert node_config["collection"]["saturation"]["target_utilization"] == 0.98
    rebuilt = coordinator.rebuild_local_state(local_db_path=workspace / "control" / "node.sqlite", current_date="2025-01-06")

    assert manifest["stage"]["symbols"]
    assert rebuilt["qc_rows"] == 1
    assert rebuilt["gap_tasks"] >= 1
    assert (nas / "control" / "cluster" / "manifest.yml").exists()
    assert (nas / "control" / "cluster" / "shards" / "equities_eod.json").exists()


def test_recreate_local_db_removes_sqlite_sidecars(tmp_path: Path) -> None:
    db_path = tmp_path / "control" / "node.sqlite"
    db = DataNodeDB(db_path)
    db.enqueue_task("equities_eod", None, "2025-01-01", "2025-01-01", "GAP", 1)
    wal_path = db_path.with_name(f"{db_path.name}-wal")
    shm_path = db_path.with_name(f"{db_path.name}-shm")
    wal_path.write_text("stale", encoding="utf-8")
    shm_path.write_text("stale", encoding="utf-8")

    recreated = DataNodeDB.recreate(db_path)

    assert not wal_path.exists()
    assert not shm_path.exists()
    recreated.update_partition_status("alpaca", "equities_eod", "2025-01-01", "GREEN", 1, 1, "OK")
    rows = recreated.fetch_partition_status()
    assert len(rows) == 1


def test_rebuild_local_state_requeues_underfilled_partitions_after_stage_promotion(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    qc_root = nas / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "source": "alpaca",
                "dataset": "equities_eod",
                "date": "2025-01-02",
                "status": "GREEN",
                "row_count": 6,
                "expected_rows": 6,
            }
        ]
    ).to_parquet(qc_root / "partition_status.parquet", index=False)
    raw_partition = nas / "data" / "raw" / "equities_bars" / "date=2025-01-02"
    raw_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL", "close": 100.0}]).to_parquet(raw_partition / "data.parquet", index=False)

    stage_symbols = [f"SYM{index:03d}" for index in range(20)]
    coordinator = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        universe_builder=lambda count: stage_symbols[:count],
    )
    coordinator.ensure_cluster_ready(passphrase="pass123")
    coordinator.update_stage(current_stage=1, symbols=stage_symbols, years=1)

    rebuilt = coordinator.rebuild_local_state(local_db_path=workspace / "control" / "node.sqlite", current_date="2025-01-06")

    assert rebuilt["gap_tasks"] >= 1
    import sqlite3

    conn = sqlite3.connect(workspace / "control" / "node.sqlite")
    queued = conn.execute(
        "SELECT COUNT(*) FROM backfill_queue WHERE dataset='equities_eod' AND start_date='2025-01-02' AND kind='GAP'"
    ).fetchone()[0]
    status_row = conn.execute(
        "SELECT status, row_count, expected_rows FROM partition_status WHERE source='alpaca' AND dataset='equities_eod' AND date='2025-01-02'"
    ).fetchone()
    conn.close()
    assert queued == 0
    assert status_row == ("AMBER", 6, 20)


def test_coordinator_bootstraps_from_config_when_stage_file_is_missing(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    (workspace / "stage.yml").unlink()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["stage"] = {"current": 0, "stage_0": {"symbols": 8, "eod_years": 3}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    coordinator = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(count)],
    )

    manifest = coordinator.ensure_cluster_ready(passphrase="pass123")

    assert manifest["stage"]["current"] == 0
    assert len(manifest["stage"]["symbols"]) == 8
    stage_payload = yaml.safe_load((workspace / "stage.yml").read_text(encoding="utf-8"))
    assert stage_payload["symbols"] == manifest["stage"]["symbols"]
    assert stage_payload["years"] == 3


def test_stale_lease_can_be_taken_over(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    coordinator_a = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control-a",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        lease_ttl_seconds=1,
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(max(count, 6))],
    )
    coordinator_b = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control-b",
        nas_share="//nas/trademl",
        worker_id="worker-b",
        lease_ttl_seconds=1,
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(max(count, 6))],
    )
    coordinator_a.ensure_cluster_ready(passphrase="pass123")
    coordinator_b.ensure_cluster_ready(passphrase="pass123")
    assert coordinator_a.acquire_or_renew_lease("equities_eod::shard-00")
    lease_path = nas / "control" / "cluster" / "leases" / "equities_eod__shard-00.json"
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    payload["expires_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(payload), encoding="utf-8")

    acquired = coordinator_b.acquire_or_renew_lease("equities_eod::shard-00")

    assert acquired is True
    updated = json.loads(lease_path.read_text(encoding="utf-8"))
    assert updated["owner"] == "worker-b"
    assert updated["epoch"] > payload["epoch"]


def test_render_and_install_systemd_service(tmp_path: Path) -> None:
    workspace, _nas, config_path, env_path = _seed_workspace(tmp_path)
    unit = render_systemd_unit(
        python_executable="/usr/bin/python3",
        config_path=config_path,
        workspace_root=workspace,
        env_path=env_path,
    )
    assert "ExecStart=/usr/bin/python3 -m trademl.data_node" in unit
    assert "Restart=always" in unit
    assert "StartLimitIntervalSec=0" in unit
    assert "KillSignal=SIGINT" in unit

    result = install_systemd_service(
        python_executable="/usr/bin/python3",
        config_path=config_path,
        workspace_root=workspace,
        env_path=env_path,
        service_path=workspace / "trademl-node.service",
    )
    assert Path(result["service_path"]).exists()


def test_install_systemd_service_falls_back_to_user_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, _nas, config_path, env_path = _seed_workspace(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    real_write_text = Path.write_text
    system_target = Path("/etc/systemd/system/trademl-node.service")
    systemctl_calls: list[list[str]] = []

    def _fake_write_text(self: Path, data: str, *args, **kwargs):
        if self == system_target:
            raise PermissionError("permission denied")
        return real_write_text(self, data, *args, **kwargs)

    def _fake_run(command: list[str], **kwargs):
        systemctl_calls.append(command)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    monkeypatch.setattr("trademl.fleet.cluster.subprocess.run", _fake_run)
    monkeypatch.setattr("trademl.fleet.cluster.shutil_which", lambda command: "/bin/systemctl" if command == "systemctl" else None)
    monkeypatch.setattr("trademl.fleet.cluster.platform.system", lambda: "Linux")

    result = install_systemd_service(
        python_executable="/usr/bin/python3",
        config_path=config_path,
        workspace_root=workspace,
        env_path=env_path,
    )

    expected_path = home / ".config" / "systemd" / "user" / "trademl-node.service"
    assert Path(result["service_path"]) == expected_path
    assert result["service_scope"] == "user"
    unit_text = expected_path.read_text(encoding="utf-8")
    assert "WantedBy=default.target" in unit_text
    assert ["systemctl", "--user", "daemon-reload"] in systemctl_calls
    assert ["systemctl", "--user", "enable", "trademl-node.service"] in systemctl_calls


def test_sync_shard_leases_does_not_release_singleton_leases(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    coordinator = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(count)],
    )
    coordinator.ensure_cluster_ready(passphrase="pass123")
    assert coordinator.acquire_singleton("backfill", "2026-04-01")

    coordinator.sync_shard_leases()

    singleton_path = nas / "control" / "cluster" / "leases" / "singleton__backfill__2026-04-01.json"
    assert singleton_path.exists()


def test_update_stage_rewrites_manifest_and_shards(tmp_path: Path) -> None:
    workspace, nas, config_path, env_path = _seed_workspace(tmp_path)
    coordinator = ClusterCoordinator(
        nas_root=nas,
        workspace_root=workspace,
        config_path=config_path,
        env_path=env_path,
        local_state=workspace / "control",
        nas_share="//nas/trademl",
        worker_id="worker-a",
        universe_builder=lambda count: [f"SYM{index:03d}" for index in range(count)],
    )
    coordinator.ensure_cluster_ready(passphrase="pass123")

    manifest = coordinator.update_stage(current_stage=1, symbols=["AAA", "BBB", "CCC", "DDD"], years=10)

    assert manifest["stage"]["current"] == 1
    assert manifest["stage"]["years"] == 10
    shard_payload = json.loads((nas / "control" / "cluster" / "shards" / "equities_eod.json").read_text(encoding="utf-8"))
    assert sum(len(item["symbols"]) for item in shard_payload["shards"]) == 4
