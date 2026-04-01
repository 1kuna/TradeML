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
    )
    manifest = coordinator.ensure_cluster_ready(passphrase="pass123")
    rebuilt = coordinator.rebuild_local_state(local_db_path=workspace / "control" / "node.sqlite", current_date="2025-01-06")

    assert manifest["stage"]["symbols"]
    assert rebuilt["qc_rows"] == 1
    assert rebuilt["gap_tasks"] >= 1
    assert (nas / "control" / "cluster" / "manifest.yml").exists()
    assert (nas / "control" / "cluster" / "shards" / "equities_eod.json").exists()


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

    result = install_systemd_service(
        python_executable="/usr/bin/python3",
        config_path=config_path,
        workspace_root=workspace,
        env_path=env_path,
        service_path=workspace / "trademl-node.service",
    )
    assert Path(result["service_path"]).exists()


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
    )
    coordinator.ensure_cluster_ready(passphrase="pass123")
    assert coordinator.acquire_singleton("backfill", "2026-04-01")

    coordinator.sync_shard_leases()

    singleton_path = nas / "control" / "cluster" / "leases" / "singleton__backfill__2026-04-01.json"
    assert singleton_path.exists()
