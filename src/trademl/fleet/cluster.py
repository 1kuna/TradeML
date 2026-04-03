"""Cluster coordination and rebuild helpers for NAS-backed worker fleets."""

from __future__ import annotations

import base64
import contextlib
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import yaml
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from trademl import __version__
from trademl.calendars.exchange import get_trading_days
from trademl.data_node.bootstrap import resolve_bootstrap_stage
from trademl.data_node.db import DataNodeDB


SHARED_SECRET_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "ALPACA_BASE_URL",
    "ALPACA_DATA_BASE_URL",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "FRED_API_KEY",
    "FMP_API_KEY",
    "MASSIVE_API_KEY",
    "SEC_EDGAR_USER_AGENT",
]
MACHINE_LOCAL_KEYS = [
    "TRADEML_ENV",
    "NAS_MOUNT",
    "NAS_SHARE",
    "LOCAL_STATE",
    "EDGE_NODE_ID",
    "COLLECTION_TIME_ET",
    "MAINTENANCE_HOUR_LOCAL",
]
SINGLETON_TASKS = ["audit_curate", "backfill", "macro", "reference", "price_checks"]


@dataclass(slots=True)
class ClusterPaths:
    """Filesystem layout for the NAS-backed cluster control plane."""

    nas_root: Path

    @property
    def control_root(self) -> Path:
        return self.nas_root / "control" / "cluster"

    @property
    def manifest_path(self) -> Path:
        return self.control_root / "manifest.yml"

    @property
    def workers_root(self) -> Path:
        return self.control_root / "workers"

    @property
    def leases_root(self) -> Path:
        return self.control_root / "leases"

    @property
    def shards_root(self) -> Path:
        return self.control_root / "shards"

    @property
    def state_root(self) -> Path:
        return self.control_root / "state"

    @property
    def events_root(self) -> Path:
        return self.control_root / "events"

    @property
    def secrets_path(self) -> Path:
        return self.control_root / "secrets.enc.json"

    @property
    def last_success_path(self) -> Path:
        return self.state_root / "last_success.json"

    def ensure_dirs(self) -> None:
        for path in [self.workers_root, self.leases_root, self.shards_root, self.state_root, self.events_root]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ShardSpec:
    """Deterministic shard assignment for symbol subsets."""

    dataset: str
    shard_id: str
    symbols: list[str]
    index: int

    @property
    def lease_id(self) -> str:
        return f"{self.dataset}::{self.shard_id}"


class SecretBundleError(RuntimeError):
    """Raised when the encrypted secret bundle cannot be decrypted."""


class ClusterCoordinator:
    """Coordinate workers, shard leases, manifest sync, and rebuild behavior."""

    def __init__(
        self,
        *,
        nas_root: Path,
        workspace_root: Path,
        config_path: Path,
        env_path: Path,
        local_state: Path,
        nas_share: str,
        worker_id: str | None = None,
        lease_ttl_seconds: int = 90,
        heartbeat_interval_seconds: int = 30,
        universe_builder: Callable[[int], list[str]] | None = None,
    ) -> None:
        self.paths = ClusterPaths(nas_root=nas_root)
        self.workspace_root = workspace_root
        self.config_path = config_path
        self.env_path = env_path
        self.local_state = local_state
        self.nas_share = nas_share
        self.worker_id = worker_id or os.getenv("EDGE_NODE_ID") or socket.gethostname()
        self.lease_ttl_seconds = lease_ttl_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.universe_builder = universe_builder

    def ensure_cluster_ready(self, *, passphrase: str | None = None) -> dict[str, Any]:
        """Bootstrap manifest/secrets if needed, then sync local config and registration."""
        self.paths.ensure_dirs()
        manifest = self._ensure_manifest()
        self._ensure_shards(manifest)
        self._ensure_secret_bundle(passphrase=passphrase)
        secrets = self.decrypt_cluster_secrets(passphrase=passphrase)
        self.materialize_local_workspace(manifest=manifest, secrets=secrets)
        self.register_worker(active=True)
        append_cluster_event(self.paths, "worker_joined", {"worker_id": self.worker_id})
        return manifest

    def decrypt_cluster_secrets(self, *, passphrase: str | None = None) -> dict[str, str]:
        """Return decrypted shared secrets, prompting only if needed."""
        if not self.paths.secrets_path.exists():
            return {}
        resolved_passphrase = self._resolve_passphrase(passphrase)
        return decrypt_secret_bundle(self.paths.secrets_path, resolved_passphrase)

    def materialize_local_workspace(self, *, manifest: dict[str, Any], secrets: dict[str, str]) -> None:
        """Write local stage/config/env files from the NAS manifest and secret bundle."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.local_state.mkdir(parents=True, exist_ok=True)
        stage_payload = {
            "current": manifest["stage"]["current"],
            "symbols": manifest["stage"]["symbols"],
            "years": manifest["stage"]["years"],
            "schedule": manifest["schedule"],
            "nas": {"share": manifest["nas_share"], "mount": str(self.paths.nas_root)},
            "cluster": {
                "worker_id": self.worker_id,
                "shard_count": manifest["datasets"]["equities_eod"]["shard_count"],
            },
        }
        _write_yaml(self.workspace_root / "stage.yml", stage_payload)
        node_payload = {
            "node": {
                "nas_mount": str(self.paths.nas_root),
                "nas_share": manifest["nas_share"],
                "local_state": str(self.local_state),
                "collection_time_et": manifest["schedule"]["collection_time_et"],
                "maintenance_hour_local": manifest["schedule"]["maintenance_hour_local"],
                "worker_id": self.worker_id,
                "lease_ttl_seconds": self.lease_ttl_seconds,
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            },
            "stage": manifest["stage_config"],
            "vendors": manifest["vendors"],
        }
        _write_yaml(self.config_path, node_payload)
        env_values = _read_env_file(self.env_path)
        env_values.update(
            {
                "TRADEML_ENV": env_values.get("TRADEML_ENV", "local"),
                "NAS_MOUNT": str(self.paths.nas_root),
                "NAS_SHARE": manifest["nas_share"],
                "LOCAL_STATE": str(self.local_state),
                "EDGE_NODE_ID": self.worker_id,
                "COLLECTION_TIME_ET": manifest["schedule"]["collection_time_et"],
                "MAINTENANCE_HOUR_LOCAL": str(manifest["schedule"]["maintenance_hour_local"]),
            }
        )
        for key, value in secrets.items():
            if value:
                env_values[key] = value
        _write_env_file(self.env_path, env_values)

    def rebuild_local_state(self, *, local_db_path: Path, current_date: str | None = None) -> dict[str, Any]:
        """Rebuild the local disposable DB from NAS data and cluster manifest."""
        manifest = self.load_manifest()
        current_ts = pd.Timestamp(current_date or datetime.now(tz=UTC).date())
        expected_rows = len(manifest["stage"]["symbols"])
        db = DataNodeDB.recreate(local_db_path)
        qc_path = self.paths.nas_root / "data" / "qc" / "partition_status.parquet"
        qc_rows = 0
        if qc_path.exists():
            qc = pd.read_parquet(qc_path)
            qc_rows = len(qc)
            for row in qc.itertuples(index=False):
                db.update_partition_status(
                    source=str(row.source),
                    dataset=str(row.dataset),
                    date=pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    status=str(row.status),
                    row_count=getattr(row, "row_count", None),
                    expected_rows=getattr(row, "expected_rows", None),
                    qc_code=getattr(row, "qc_code", None),
                    note=getattr(row, "note", None),
                )
        stage_years = int(manifest["stage"]["years"])
        start_ts = current_ts - pd.DateOffset(years=stage_years)
        expected_days = [day.isoformat() for day in get_trading_days("XNYS", start_ts.date(), current_ts.date())]
        raw_root = self.paths.nas_root / "data" / "raw" / "equities_bars"
        statuses = {
            (row["source"], row["dataset"], row["date"]): row
            for row in db.fetch_partition_status()
        }
        queue_count = 0
        for day in expected_days:
            partition = raw_root / f"date={day}" / "data.parquet"
            existing = statuses.get(("alpaca", "equities_eod", day))
            needs_gap = not partition.exists()
            row_count = int(existing["row_count"] or 0) if existing else 0
            if partition.exists() and expected_rows and row_count < expected_rows:
                needs_gap = True
                db.update_partition_status(
                    source="alpaca",
                    dataset="equities_eod",
                    date=day,
                    status="AMBER",
                    row_count=row_count,
                    expected_rows=expected_rows,
                    qc_code="LOW_ROW_COUNT",
                    note="rebuild detected partition below current stage symbol count",
                )
            elif partition.exists() and existing and int(existing["expected_rows"] or 0) < expected_rows:
                db.update_partition_status(
                    source="alpaca",
                    dataset="equities_eod",
                    date=day,
                    status=str(existing["status"]),
                    row_count=row_count,
                    expected_rows=expected_rows,
                    qc_code=existing["qc_code"],
                    note=existing["note"],
                )
            if not needs_gap:
                continue
            queue_count += 1
        self.register_worker(active=True)
        append_cluster_event(self.paths, "worker_rebuilt", {"worker_id": self.worker_id, "gap_tasks": queue_count})
        return {"qc_rows": qc_rows, "gap_tasks": queue_count, "worker_id": self.worker_id}

    def register_worker(self, *, active: bool) -> dict[str, Any]:
        """Register or update the worker metadata file."""
        self.paths.ensure_dirs()
        payload = {
            "worker_id": self.worker_id,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "version": __version__,
            "active": active,
            "joined_at": self._existing_worker_field("joined_at") or _utcnow().isoformat(),
            "last_heartbeat": _utcnow().isoformat(),
            "workspace_root": str(self.workspace_root),
            "local_state": str(self.local_state),
            "capabilities": {
                "equities_shards": True,
                "reference": True,
                "macro": True,
                "price_checks": True,
            },
        }
        _write_json(self.paths.workers_root / f"{self.worker_id}.json", payload)
        return payload

    def heartbeat_worker(self) -> dict[str, Any]:
        """Refresh the worker heartbeat and any leases already owned by this worker."""
        worker = self.register_worker(active=True)
        for lease_id in self.list_owned_lease_ids():
            self.acquire_or_renew_lease(lease_id)
        return worker

    def mark_worker_inactive(self) -> dict[str, Any]:
        worker = self.register_worker(active=False)
        append_cluster_event(self.paths, "worker_left", {"worker_id": self.worker_id})
        return worker

    def load_manifest(self) -> dict[str, Any]:
        return _read_yaml(self.paths.manifest_path)

    def load_shard_specs(self) -> list[ShardSpec]:
        payload = _read_json(self.paths.shards_root / "equities_eod.json")
        return [ShardSpec(**item) for item in payload.get("shards", [])]

    def active_workers(self) -> list[dict[str, Any]]:
        now = _utcnow()
        active: list[dict[str, Any]] = []
        for path in sorted(self.paths.workers_root.glob("*.json")):
            payload = _read_json(path)
            if not payload:
                continue
            last = payload.get("last_heartbeat")
            if not payload.get("active", False) or not last:
                continue
            if (now - datetime.fromisoformat(str(last))).total_seconds() > self.lease_ttl_seconds:
                continue
            active.append(payload)
        return sorted(active, key=lambda item: str(item["worker_id"]))

    def desired_shards_for_worker(self) -> list[ShardSpec]:
        workers = [item["worker_id"] for item in self.active_workers()]
        if self.worker_id not in workers:
            workers.append(self.worker_id)
            workers = sorted(set(workers))
        shards = self.load_shard_specs()
        if not workers:
            return []
        owned: list[ShardSpec] = []
        for index, shard in enumerate(shards):
            if workers[index % len(workers)] == self.worker_id:
                owned.append(shard)
        return owned

    def assigned_singleton_tasks(self) -> list[str]:
        workers = [item["worker_id"] for item in self.active_workers()]
        if self.worker_id not in workers:
            workers.append(self.worker_id)
            workers = sorted(set(workers))
        if not workers:
            return []
        assignments: list[str] = []
        for index, task_name in enumerate(SINGLETON_TASKS):
            if workers[index % len(workers)] == self.worker_id:
                assignments.append(task_name)
        return assignments

    def sync_shard_leases(self) -> list[ShardSpec]:
        """Acquire or renew the shards this worker should own, and release others."""
        target = {spec.lease_id: spec for spec in self.desired_shards_for_worker()}
        current = {lease_id for lease_id in self.list_owned_lease_ids() if lease_id.startswith("equities_eod::")}
        for lease_id in current - set(target):
            self.release_lease(lease_id)
        owned_specs: list[ShardSpec] = []
        for lease_id, spec in target.items():
            if self.acquire_or_renew_lease(lease_id):
                owned_specs.append(spec)
        return owned_specs

    def acquire_or_renew_lease(self, lease_id: str) -> bool:
        """Acquire a stale/free lease or renew an existing lease held by this worker."""
        lease_path = self.paths.leases_root / f"{_safe_name(lease_id)}.json"
        lock_path = lease_path.with_suffix(".lock")
        with _exclusive_lock(lock_path):
            current = _read_json(lease_path)
            now = _utcnow()
            expires_at = now + pd.Timedelta(seconds=self.lease_ttl_seconds)
            if current:
                owner = current.get("owner")
                current_expiry = datetime.fromisoformat(str(current["expires_at"])) if current.get("expires_at") else None
                if owner not in {self.worker_id, None} and current_expiry and current_expiry > now:
                    return False
                epoch = int(current.get("epoch", 0)) + (1 if owner != self.worker_id else 0)
            else:
                epoch = 1
            payload = {
                "lease_id": lease_id,
                "owner": self.worker_id,
                "epoch": epoch,
                "acquired_at": current.get("acquired_at") if current and current.get("owner") == self.worker_id else now.isoformat(),
                "heartbeat_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
            _write_json(lease_path, payload)
        return True

    def release_lease(self, lease_id: str, *, force: bool = False) -> bool:
        lease_path = self.paths.leases_root / f"{_safe_name(lease_id)}.json"
        lock_path = lease_path.with_suffix(".lock")
        with _exclusive_lock(lock_path):
            current = _read_json(lease_path)
            if not current:
                return True
            if not force and current.get("owner") not in {self.worker_id, None}:
                return False
            if lease_path.exists():
                lease_path.unlink()
        append_cluster_event(self.paths, "lease_released", {"worker_id": self.worker_id, "lease_id": lease_id, "force": force})
        return True

    def force_release_lease(self, lease_id: str) -> bool:
        return self.release_lease(lease_id, force=True)

    def list_owned_lease_ids(self) -> list[str]:
        now = _utcnow()
        lease_ids: list[str] = []
        for path in self.paths.leases_root.glob("*.json"):
            payload = _read_json(path)
            if payload.get("owner") != self.worker_id:
                continue
            expires_at = payload.get("expires_at")
            if expires_at and datetime.fromisoformat(str(expires_at)) > now:
                lease_ids.append(str(payload["lease_id"]))
        return sorted(lease_ids)

    def singleton_should_run(self, task_name: str, bucket_key: str) -> bool:
        """Return whether this worker is assigned the singleton and has not completed it."""
        if task_name not in self.assigned_singleton_tasks():
            return False
        success = self.read_last_success()
        task_state = success.get(task_name, {})
        return str(task_state.get("bucket")) != bucket_key

    def acquire_singleton(self, task_name: str, bucket_key: str) -> bool:
        if not self.singleton_should_run(task_name, bucket_key):
            return False
        return self.acquire_or_renew_lease(f"singleton::{task_name}::{bucket_key}")

    def mark_singleton_success(self, task_name: str, bucket_key: str, metadata: dict[str, Any] | None = None) -> None:
        state = self.read_last_success()
        state[task_name] = {
            "bucket": bucket_key,
            "worker_id": self.worker_id,
            "updated_at": _utcnow().isoformat(),
            "metadata": metadata or {},
        }
        _write_json(self.paths.last_success_path, state)
        append_cluster_event(self.paths, "singleton_success", {"worker_id": self.worker_id, "task": task_name, "bucket": bucket_key})

    def read_last_success(self) -> dict[str, Any]:
        return _read_json(self.paths.last_success_path)

    def leave_cluster(self) -> dict[str, Any]:
        for lease_id in self.list_owned_lease_ids():
            self.release_lease(lease_id, force=True)
        worker = self.mark_worker_inactive()
        return {"worker": worker, "released_leases": []}

    def _ensure_manifest(self) -> dict[str, Any]:
        manifest = _read_yaml(self.paths.manifest_path)
        if manifest:
            return manifest
        local_config = _read_yaml(self.config_path)
        local_stage = _read_yaml(self.workspace_root / "stage.yml")
        current_stage, stage_symbols, stage_years = resolve_bootstrap_stage(
            local_config,
            local_stage,
            universe_builder=self.universe_builder,
        )
        if not stage_symbols:
            raise RuntimeError("cluster bootstrap requires explicit stage symbols or a Stage 0 universe builder")
        manifest = {
            "version": 1,
            "nas_share": self.nas_share,
            "nas_mount": str(self.paths.nas_root),
            "schedule": {
                "collection_time_et": str(
                    local_config.get("node", {}).get("collection_time_et", local_stage.get("schedule", {}).get("collection_time_et", "16:30"))
                ),
                "maintenance_hour_local": int(
                    local_config.get("node", {}).get("maintenance_hour_local", local_stage.get("schedule", {}).get("maintenance_hour_local", 2))
                ),
            },
            "stage": {
                "current": current_stage,
                "symbols": stage_symbols,
                "years": stage_years,
            },
            "stage_config": local_config.get("stage", {}),
            "vendors": local_config.get("vendors", {}),
            "datasets": {
                "equities_eod": {
                    "shard_count": int(local_config.get("node", {}).get("shard_count", 8)),
                    "exchange": "XNYS",
                }
            },
        }
        _write_yaml(self.paths.manifest_path, manifest)
        append_cluster_event(self.paths, "cluster_bootstrapped", {"worker_id": self.worker_id, "symbol_count": len(stage_symbols)})
        return manifest

    def _ensure_shards(self, manifest: dict[str, Any]) -> None:
        shard_path = self.paths.shards_root / "equities_eod.json"
        if shard_path.exists():
            return
        self.write_shards(manifest)

    def write_shards(self, manifest: dict[str, Any]) -> None:
        """Write the current equities shard map from the manifest stage symbols."""
        shard_count = int(manifest["datasets"]["equities_eod"]["shard_count"])
        shard_specs = build_shard_map("equities_eod", manifest["stage"]["symbols"], shard_count)
        _write_json(
            self.paths.shards_root / "equities_eod.json",
            {
                "dataset": "equities_eod",
                "shard_count": shard_count,
                "shards": [
                    {
                        "dataset": spec.dataset,
                        "shard_id": spec.shard_id,
                        "symbols": spec.symbols,
                        "index": spec.index,
                    }
                    for spec in shard_specs
                ],
            },
        )

    def update_stage(self, *, current_stage: int, symbols: list[str], years: int) -> dict[str, Any]:
        """Persist a new active stage into the manifest and refresh shard layout."""
        manifest = self.load_manifest()
        if not manifest:
            raise RuntimeError("cluster manifest missing")
        manifest["stage"] = {"current": int(current_stage), "symbols": list(symbols), "years": int(years)}
        _write_yaml(self.paths.manifest_path, manifest)
        self.write_shards(manifest)
        append_cluster_event(
            self.paths,
            "stage_updated",
            {"worker_id": self.worker_id, "current_stage": int(current_stage), "symbol_count": len(symbols), "years": int(years)},
        )
        return manifest

    def _ensure_secret_bundle(self, *, passphrase: str | None) -> None:
        if self.paths.secrets_path.exists():
            return
        env_values = _read_env_file(self.env_path)
        shared = {key: env_values.get(key, os.getenv(key, "")) for key in SHARED_SECRET_KEYS if env_values.get(key, os.getenv(key, ""))}
        if not shared:
            return
        encrypt_secret_bundle(self.paths.secrets_path, self._resolve_passphrase(passphrase), shared)
        append_cluster_event(self.paths, "secrets_bootstrapped", {"worker_id": self.worker_id, "keys": sorted(shared)})

    def _existing_worker_field(self, field_name: str) -> str | None:
        payload = _read_json(self.paths.workers_root / f"{self.worker_id}.json")
        value = payload.get(field_name)
        return str(value) if value else None

    @staticmethod
    def _resolve_passphrase(passphrase: str | None) -> str:
        if passphrase:
            return passphrase
        env_value = os.getenv("TRADEML_CLUSTER_PASSPHRASE")
        if env_value:
            return env_value
        if os.isatty(0):
            return getpass.getpass("TradeML cluster passphrase: ")
        raise RuntimeError("cluster passphrase required via TTY or TRADEML_CLUSTER_PASSPHRASE")


def build_shard_map(dataset: str, symbols: list[str], shard_count: int) -> list[ShardSpec]:
    """Build a deterministic symbol-to-shard map."""
    buckets: dict[int, list[str]] = {index: [] for index in range(shard_count)}
    for symbol in sorted(set(symbols)):
        digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % shard_count
        buckets[bucket].append(symbol)
    return [
        ShardSpec(dataset=dataset, shard_id=f"shard-{index:02d}", symbols=buckets[index], index=index)
        for index in range(shard_count)
    ]


def encrypt_secret_bundle(path: Path, passphrase: str, payload: dict[str, str]) -> None:
    """Encrypt a JSON secret bundle with scrypt + AES-GCM."""
    salt = os.urandom(16)
    nonce = os.urandom(12)
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    key = kdf.derive(passphrase.encode("utf-8"))
    cipher = AESGCM(key)
    ciphertext = cipher.encrypt(nonce, json.dumps(payload, sort_keys=True).encode("utf-8"), None)
    bundle = {
        "version": 1,
        "kdf": {
            "name": "scrypt",
            "salt": base64.b64encode(salt).decode("ascii"),
            "n": 2**14,
            "r": 8,
            "p": 1,
        },
        "cipher": {
            "name": "AESGCM",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        },
    }
    _write_json(path, bundle)


def decrypt_secret_bundle(path: Path, passphrase: str) -> dict[str, str]:
    """Decrypt the NAS-backed secret bundle."""
    payload = _read_json(path)
    if not payload:
        return {}
    salt = base64.b64decode(payload["kdf"]["salt"])
    nonce = base64.b64decode(payload["cipher"]["nonce"])
    ciphertext = base64.b64decode(payload["cipher"]["ciphertext"])
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=int(payload["kdf"]["n"]),
        r=int(payload["kdf"]["r"]),
        p=int(payload["kdf"]["p"]),
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise SecretBundleError("invalid cluster passphrase") from exc
    return json.loads(plaintext.decode("utf-8"))


def append_cluster_event(paths: ClusterPaths, event_type: str, payload: dict[str, Any]) -> None:
    """Append a fleet event to the monthly JSONL log."""
    paths.ensure_dirs()
    timestamp = _utcnow()
    event_path = paths.events_root / f"{timestamp:%Y-%m}.jsonl"
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": timestamp.isoformat(), "event_type": event_type, "payload": payload}, sort_keys=True) + "\n")


def rebuild_local_state(
    *,
    nas_root: Path,
    workspace_root: Path,
    config_path: Path,
    env_path: Path,
    local_state: Path,
    nas_share: str,
    worker_id: str | None = None,
    passphrase: str | None = None,
    current_date: str | None = None,
    universe_builder: Callable[[int], list[str]] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one-shot cluster join + rebuild."""
    coordinator = ClusterCoordinator(
        nas_root=nas_root,
        workspace_root=workspace_root,
        config_path=config_path,
        env_path=env_path,
        local_state=local_state,
        nas_share=nas_share,
        worker_id=worker_id,
        universe_builder=universe_builder,
    )
    manifest = coordinator.ensure_cluster_ready(passphrase=passphrase)
    rebuilt = coordinator.rebuild_local_state(local_db_path=local_state / "node.sqlite", current_date=current_date)
    return {"manifest": manifest, "rebuilt": rebuilt, "worker_id": coordinator.worker_id}


def read_cluster_snapshot(*, nas_root: Path, worker_id: str | None = None) -> dict[str, Any]:
    """Read fleet state from NAS for dashboard/status views."""
    paths = ClusterPaths(nas_root=nas_root)
    paths.ensure_dirs()
    workers = [_read_json(path) for path in sorted(paths.workers_root.glob("*.json"))]
    leases = [_read_json(path) for path in sorted(paths.leases_root.glob("*.json"))]
    events = []
    latest_event_log = sorted(paths.events_root.glob("*.jsonl"))[-1:] if paths.events_root.exists() else []
    for path in latest_event_log:
        lines = path.read_text(encoding="utf-8").splitlines()[-50:]
        events.extend(json.loads(line) for line in lines if line.strip())
    manifest = _read_yaml(paths.manifest_path)
    shards = _read_json(paths.shards_root / "equities_eod.json")
    last_success = _read_json(paths.last_success_path)
    active_workers = []
    now = _utcnow()
    for payload in workers:
        last = payload.get("last_heartbeat")
        is_active = bool(payload.get("active")) and bool(last) and (now - datetime.fromisoformat(str(last))).total_seconds() <= 90
        payload["is_stale"] = not is_active
        if is_active:
            active_workers.append(payload)
    owned = [lease for lease in leases if worker_id and lease.get("owner") == worker_id]
    return {
        "manifest": manifest,
        "workers": workers,
        "active_workers": active_workers,
        "leases": leases,
        "owned_leases": owned,
        "shards": shards.get("shards", []),
        "last_success": last_success,
        "recent_events": events,
    }


def render_systemd_unit(*, python_executable: str, config_path: Path, workspace_root: Path, env_path: Path) -> str:
    """Render the systemd unit content for the worker service."""
    return "\n".join(
        [
            "[Unit]",
            "Description=TradeML NAS-backed data worker",
            "After=network-online.target remote-fs.target",
            "Wants=network-online.target remote-fs.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={workspace_root}",
            f"Environment=PYTHONUNBUFFERED=1",
            f"ExecStart={python_executable} -m trademl.data_node --config {config_path} --root {workspace_root} --env-file {env_path}",
            "Restart=on-failure",
            "RestartSec=15",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )


def install_systemd_service(
    *,
    python_executable: str,
    config_path: Path,
    workspace_root: Path,
    env_path: Path,
    service_path: Path | None = None,
) -> dict[str, str]:
    """Write the worker systemd unit, falling back to a local file if needed."""
    target = service_path or Path("/etc/systemd/system/trademl-node.service")
    unit = render_systemd_unit(
        python_executable=python_executable,
        config_path=config_path,
        workspace_root=workspace_root,
        env_path=env_path,
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(unit, encoding="utf-8")
    except OSError:
        target = workspace_root / "trademl-node.service"
        target.write_text(unit, encoding="utf-8")
    return {"service_path": str(target), "service_name": "trademl-node.service"}


def systemd_status(service_name: str = "trademl-node.service") -> dict[str, Any]:
    """Return a lightweight `systemctl status` summary when available."""
    if platform.system() != "Linux":
        return {"supported": False, "reason": "systemd is only supported on Linux"}
    if not shutil_which("systemctl"):
        return {"supported": False, "reason": "systemctl not found"}
    result = subprocess.run(
        ["systemctl", "show", service_name, "--no-page", "--property=LoadState,ActiveState,SubState,UnitFileState"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = {"supported": True, "returncode": result.returncode, "service_name": service_name, "raw": result.stdout.strip()}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            payload[key] = value
    return payload


def systemd_journal_tail(service_name: str = "trademl-node.service", *, lines: int = 100) -> str:
    """Return recent journal lines for the worker service when available."""
    if platform.system() != "Linux" or not shutil_which("journalctl"):
        return ""
    result = subprocess.run(
        ["journalctl", "-u", service_name, "-n", str(lines), "--no-pager"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def shutil_which(command: str) -> str | None:
    return subprocess.run(["/usr/bin/which", command], check=False, capture_output=True, text=True).stdout.strip() or None


@contextlib.contextmanager
def _exclusive_lock(path: Path, *, stale_after_seconds: int = 15) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if path.exists() and time.time() - path.stat().st_mtime > stale_after_seconds:
                with contextlib.suppress(OSError):
                    path.unlink()
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: values[key] for key in sorted(values)}
    path.write_text("\n".join(f"{key}={value}" for key, value in ordered.items()) + "\n", encoding="utf-8")


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)
