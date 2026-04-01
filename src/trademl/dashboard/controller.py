"""Backend helpers for the operator dashboard."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trademl.calendars.exchange import get_trading_days
from trademl.fleet.cluster import (
    ClusterCoordinator,
    ClusterPaths,
    append_cluster_event,
    encrypt_secret_bundle,
    install_systemd_service,
    read_cluster_snapshot,
    rebuild_local_state as rebuild_cluster_local_state,
    systemd_journal_tail,
    systemd_status,
)


@dataclass(slots=True)
class NodeSettings:
    """Resolved paths and settings for a dashboard-managed node."""

    repo_root: Path
    workspace_root: Path
    config_path: Path
    env_path: Path
    local_state: Path
    nas_mount: Path
    nas_share: str
    collection_time_et: str
    maintenance_hour_local: int
    worker_id: str

    @property
    def stage_path(self) -> Path:
        return self.workspace_root / "stage.yml"

    @property
    def runtime_path(self) -> Path:
        return self.local_state / "node_runtime.json"

    @property
    def log_path(self) -> Path:
        return self.local_state / "logs" / "node.log"

    @property
    def db_path(self) -> Path:
        return self.local_state / "node.sqlite"

    @property
    def qc_path(self) -> Path:
        return self.nas_mount / "data" / "qc" / "partition_status.parquet"

    @property
    def raw_equities_root(self) -> Path:
        return self.nas_mount / "data" / "raw" / "equities_bars"

    @property
    def curated_equities_root(self) -> Path:
        return self.nas_mount / "data" / "curated" / "equities_ohlcv_adj"

    @property
    def reference_root(self) -> Path:
        return self.nas_mount / "data" / "reference"

    @property
    def macro_root(self) -> Path:
        return self.nas_mount / "data" / "raw" / "macros_fred"

    @property
    def price_checks_root(self) -> Path:
        return self.nas_mount / "data" / "qc"

    @property
    def cluster_paths(self) -> ClusterPaths:
        return ClusterPaths(self.nas_mount)


def resolve_node_settings(
    *,
    workspace_root: str | Path | None = None,
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> NodeSettings:
    """Resolve dashboard settings from workspace files and defaults."""
    repo_root = Path(__file__).resolve().parents[3]
    resolved_workspace = Path(workspace_root or os.getenv("TRADEML_WORKSPACE_ROOT", "~/trademl")).expanduser()
    resolved_config = (
        Path(config_path).expanduser()
        if config_path
        else (resolved_workspace / "node.yml" if (resolved_workspace / "node.yml").exists() else repo_root / "configs" / "node.yml")
    )
    resolved_env = Path(env_path).expanduser() if env_path else resolved_workspace / ".env"

    config = _read_yaml(resolved_config)
    env_values = _read_env_file(resolved_env)
    stage = _read_yaml(resolved_workspace / "stage.yml")
    node = config.get("node", {})
    local_state = Path(env_values.get("LOCAL_STATE") or node.get("local_state") or (resolved_workspace / "control")).expanduser()
    nas_mount = Path(env_values.get("NAS_MOUNT") or node.get("nas_mount") or "/mnt/trademl").expanduser()
    nas_share = env_values.get("NAS_SHARE") or node.get("nas_share") or stage.get("nas", {}).get("share") or "//nas/trademl"
    collection_time_et = env_values.get("COLLECTION_TIME_ET") or node.get("collection_time_et") or "16:30"
    maintenance_hour_local = int(
        env_values.get("MAINTENANCE_HOUR_LOCAL") or node.get("maintenance_hour_local") or 2
    )
    worker_id = env_values.get("EDGE_NODE_ID") or node.get("worker_id") or socket.gethostname()
    return NodeSettings(
        repo_root=repo_root,
        workspace_root=resolved_workspace,
        config_path=resolved_config,
        env_path=resolved_env,
        local_state=local_state,
        nas_mount=nas_mount,
        nas_share=nas_share,
        collection_time_et=collection_time_et,
        maintenance_hour_local=maintenance_hour_local,
        worker_id=worker_id,
    )


def persist_node_settings(
    settings: NodeSettings,
    *,
    nas_share: str,
    nas_mount: str,
    collection_time_et: str,
    maintenance_hour_local: int,
    fstab_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist dashboard-edited settings back to config, env, and stage files."""
    config = _read_yaml(settings.config_path)
    node = config.setdefault("node", {})
    node["nas_mount"] = nas_mount
    node["nas_share"] = nas_share
    node["local_state"] = str(settings.local_state)
    node["collection_time_et"] = collection_time_et
    node["maintenance_hour_local"] = int(maintenance_hour_local)
    node["worker_id"] = settings.worker_id
    _write_yaml(settings.config_path, config)

    env_values = _read_env_file(settings.env_path)
    env_values.update(
        {
            "TRADEML_ENV": env_values.get("TRADEML_ENV", "local"),
            "NAS_MOUNT": nas_mount,
            "NAS_SHARE": nas_share,
            "LOCAL_STATE": str(settings.local_state),
            "COLLECTION_TIME_ET": collection_time_et,
            "MAINTENANCE_HOUR_LOCAL": str(maintenance_hour_local),
        }
    )
    _write_env_file(settings.env_path, env_values)

    stage = _read_yaml(settings.stage_path)
    if stage:
        stage.setdefault("schedule", {})
        stage["schedule"]["collection_time_et"] = collection_time_et
        stage["schedule"]["maintenance_hour_local"] = int(maintenance_hour_local)
        stage.setdefault("nas", {})
        stage["nas"]["share"] = nas_share
        stage["nas"]["mount"] = nas_mount
        _write_yaml(settings.stage_path, stage)

    manifest = _read_yaml(settings.cluster_paths.manifest_path)
    if manifest:
        manifest["nas_share"] = nas_share
        manifest.setdefault("schedule", {})
        manifest["schedule"]["collection_time_et"] = collection_time_et
        manifest["schedule"]["maintenance_hour_local"] = int(maintenance_hour_local)
        _write_yaml(settings.cluster_paths.manifest_path, manifest)

    persisted_fstab = persist_fstab_entry(
        path=Path(fstab_path).expanduser() if fstab_path else Path("/etc/fstab"),
        nas_share=nas_share,
        nas_mount=nas_mount,
    )
    return {"config_path": str(settings.config_path), "env_path": str(settings.env_path), "fstab_path": str(persisted_fstab)}


def persist_fstab_entry(*, path: Path, nas_share: str, nas_mount: str) -> Path:
    """Persist a CIFS mount line, falling back to a local file if privileged write fails."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        target_path = path
    except OSError:
        target_path = Path.cwd() / "fstab.tradeML"
        target_path.parent.mkdir(parents=True, exist_ok=True)

    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    entry = f"{nas_share} {nas_mount} cifs credentials=/etc/nas-creds,uid=pi,gid=pi 0 0"
    lines = [line for line in existing.splitlines() if line.strip() and nas_share not in line and nas_mount not in line]
    lines.append(entry)
    try:
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        target_path = Path.cwd() / "fstab.tradeML"
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def start_node(
    settings: NodeSettings,
    *,
    command: list[str] | None = None,
    passphrase: str | None = None,
) -> dict[str, Any]:
    """Start the data-node process in the background if it is not already running."""
    if passphrase:
        persist_cluster_passphrase(settings, passphrase)
    svc = systemd_status()
    if svc.get("supported") and svc.get("UnitFileState") not in {None, "not-found", "", "masked"}:
        subprocess.run(["systemctl", "start", "trademl-node.service"], check=False)
        snapshot = collect_dashboard_snapshot(settings)
        snapshot["runtime"]["managed_by"] = "systemd"
        return snapshot["runtime"]
    runtime = _read_runtime_state(settings)
    pid = runtime.get("pid")
    if isinstance(pid, int) and _is_process_running(pid):
        runtime["running"] = True
        return runtime

    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.local_state.mkdir(parents=True, exist_ok=True)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_command = command or [
        sys.executable,
        "-m",
        "trademl.data_node",
        "--config",
        str(settings.config_path),
        "--root",
        str(settings.workspace_root),
        "--env-file",
        str(settings.env_path),
    ]
    env = os.environ.copy()
    env.update(_read_env_file(settings.env_path))
    with settings.log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            launch_command,
            cwd=settings.workspace_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runtime = {
        "pid": process.pid,
        "started_at": datetime.now(tz=UTC).isoformat(),
        "log_path": str(settings.log_path),
        "command": launch_command,
        "workspace_root": str(settings.workspace_root),
        "config_path": str(settings.config_path),
        "env_path": str(settings.env_path),
        "running": True,
    }
    _write_runtime_state(settings, runtime)
    return runtime


def stop_node(settings: NodeSettings, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Stop the managed node process if it is running."""
    svc = systemd_status()
    if svc.get("supported") and svc.get("UnitFileState") not in {None, "not-found", "", "masked"}:
        subprocess.run(["systemctl", "stop", "trademl-node.service"], check=False)
        runtime = _read_runtime_state(settings)
        runtime["running"] = False
        runtime["managed_by"] = "systemd"
        _write_runtime_state(settings, runtime)
        return runtime
    runtime = _read_runtime_state(settings)
    pid = runtime.get("pid")
    if not isinstance(pid, int) or not _is_process_running(pid):
        runtime["running"] = False
        runtime["stopped_at"] = datetime.now(tz=UTC).isoformat()
        _write_runtime_state(settings, runtime)
        return runtime

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_process_running(pid):
            break
        time.sleep(0.1)
    if _is_process_running(pid):
        os.kill(pid, signal.SIGKILL)
    runtime["running"] = False
    runtime["last_pid"] = pid
    runtime["stopped_at"] = datetime.now(tz=UTC).isoformat()
    runtime.pop("pid", None)
    _write_runtime_state(settings, runtime)
    return runtime


def restart_node(settings: NodeSettings, *, passphrase: str | None = None) -> dict[str, Any]:
    """Restart the managed node and return the new runtime state."""
    stop_node(settings)
    return start_node(settings, passphrase=passphrase)


def join_cluster(settings: NodeSettings, *, passphrase: str | None = None) -> dict[str, Any]:
    """Bootstrap or join the NAS-backed worker cluster."""
    if passphrase:
        persist_cluster_passphrase(settings, passphrase)
    coordinator = _coordinator(settings)
    manifest = coordinator.ensure_cluster_ready(passphrase=passphrase)
    rebuilt = coordinator.rebuild_local_state(local_db_path=settings.db_path)
    return {"manifest": manifest, "rebuilt": rebuilt, "worker_id": coordinator.worker_id}


def rebuild_cluster_state(settings: NodeSettings, *, passphrase: str | None = None) -> dict[str, Any]:
    """Rebuild disposable local state from NAS-backed truth."""
    if passphrase:
        persist_cluster_passphrase(settings, passphrase)
    return rebuild_cluster_local_state(
        nas_root=settings.nas_mount,
        workspace_root=settings.workspace_root,
        config_path=settings.config_path,
        env_path=settings.env_path,
        local_state=settings.local_state,
        nas_share=settings.nas_share,
        worker_id=settings.worker_id,
        passphrase=passphrase,
    )


def leave_cluster(settings: NodeSettings) -> dict[str, Any]:
    """Leave the cluster and release any owned leases."""
    coordinator = _coordinator(settings)
    return coordinator.leave_cluster()


def persist_cluster_passphrase(settings: NodeSettings, passphrase: str) -> None:
    """Persist the cluster passphrase for non-interactive worker starts."""
    env_values = _read_env_file(settings.env_path)
    env_values["TRADEML_CLUSTER_PASSPHRASE"] = passphrase
    _write_env_file(settings.env_path, env_values)


def install_service(settings: NodeSettings, *, service_path: str | None = None) -> dict[str, Any]:
    """Install the Linux systemd service definition for the worker."""
    result = install_systemd_service(
        python_executable=sys.executable,
        config_path=settings.config_path,
        workspace_root=settings.workspace_root,
        env_path=settings.env_path,
        service_path=Path(service_path).expanduser() if service_path else None,
    )
    append_cluster_event(settings.cluster_paths, "service_installed", {"worker_id": settings.worker_id, **result})
    return result


def update_worker(settings: NodeSettings) -> dict[str, Any]:
    """Update the local worker installation and optionally restart it."""
    runtime_before = collect_dashboard_snapshot(settings)["runtime"]
    was_running = bool(runtime_before.get("running"))
    if was_running:
        stop_node(settings)

    venv_path = settings.repo_root / ".venv"
    bin_dir = Path.home() / ".local" / "bin"
    wrapper_path = bin_dir / "trademl"
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    pip_path = venv_path / "bin" / "pip"
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_path), "install", "-e", f"{settings.repo_root}[dev,dashboard]"], check=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        "#!/usr/bin/env sh\n"
        f'exec "{venv_path / "bin" / "trademl"}" "$@"\n',
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)

    service_result = None
    if settings.cluster_paths.manifest_path.exists():
        service_result = install_service(settings)
    if was_running:
        start_node(settings)
    append_cluster_event(settings.cluster_paths, "worker_updated", {"worker_id": settings.worker_id, "wrapper_path": str(wrapper_path)})
    return {
        "venv_path": str(venv_path),
        "wrapper_path": str(wrapper_path),
        "service": service_result,
        "restarted": was_running,
    }


def reset_worker(settings: NodeSettings, *, passphrase: str | None = None) -> dict[str, Any]:
    """Wipe local disposable worker state, then rejoin/rebuild from NAS truth."""
    runtime_before = collect_dashboard_snapshot(settings)["runtime"]
    was_running = bool(runtime_before.get("running"))
    stop_node(settings)
    leave_cluster(settings)
    _remove_worker_state(settings)
    joined = join_cluster(settings, passphrase=passphrase)
    if was_running:
        start_node(settings, passphrase=passphrase)
    append_cluster_event(settings.cluster_paths, "worker_reset", {"worker_id": settings.worker_id})
    return {"joined": joined, "restarted": was_running, "workspace_root": str(settings.workspace_root)}


def uninstall_worker(settings: NodeSettings) -> dict[str, Any]:
    """Remove local worker artifacts and detach the machine from the cluster."""
    stop_node(settings)
    leave_cluster(settings)
    removed_paths: list[str] = []

    if platform.system() == "Linux" and shutil.which("systemctl"):
        subprocess.run(["systemctl", "disable", "--now", "trademl-node.service"], check=False)
        for candidate in [Path("/etc/systemd/system/trademl-node.service"), settings.workspace_root / "trademl-node.service"]:
            if candidate.exists():
                try:
                    candidate.unlink()
                    removed_paths.append(str(candidate))
                except OSError:
                    pass
        subprocess.run(["systemctl", "daemon-reload"], check=False)

    wrapper_path = Path.home() / ".local" / "bin" / "trademl"
    if wrapper_path.exists():
        try:
            wrapper_path.unlink()
            removed_paths.append(str(wrapper_path))
        except OSError:
            pass

    if settings.workspace_root.exists():
        shutil.rmtree(settings.workspace_root, ignore_errors=True)
        removed_paths.append(str(settings.workspace_root))

    runtime_path = settings.local_state / "node_runtime.json"
    if runtime_path.exists():
        with contextlib.suppress(OSError):
            runtime_path.unlink()

    append_cluster_event(settings.cluster_paths, "worker_uninstalled", {"worker_id": settings.worker_id, "removed_paths": removed_paths})
    return {"removed_paths": removed_paths, "worker_id": settings.worker_id}


def rotate_cluster_passphrase(
    settings: NodeSettings,
    *,
    old_passphrase: str,
    new_passphrase: str,
) -> dict[str, Any]:
    """Rotate the NAS-backed encrypted secret bundle passphrase."""
    coordinator = _coordinator(settings)
    secrets = coordinator.decrypt_cluster_secrets(passphrase=old_passphrase)
    encrypt_secret_bundle(settings.cluster_paths.secrets_path, new_passphrase, secrets)
    persist_cluster_passphrase(settings, new_passphrase)
    append_cluster_event(settings.cluster_paths, "passphrase_rotated", {"worker_id": settings.worker_id})
    return {"rotated_keys": sorted(secrets)}


def update_cluster_secrets(
    settings: NodeSettings,
    *,
    passphrase: str,
    updates: dict[str, str],
) -> dict[str, Any]:
    """Update the encrypted NAS-backed shared secret bundle."""
    coordinator = _coordinator(settings)
    secrets = coordinator.decrypt_cluster_secrets(passphrase=passphrase)
    secrets.update({key: value for key, value in updates.items() if key})
    encrypt_secret_bundle(settings.cluster_paths.secrets_path, passphrase, secrets)
    persist_cluster_passphrase(settings, passphrase)
    append_cluster_event(settings.cluster_paths, "secrets_updated", {"worker_id": settings.worker_id, "keys": sorted(updates)})
    return {"keys": sorted(secrets)}


def force_release_lease(settings: NodeSettings, lease_id: str) -> bool:
    """Force-release a lease from the dashboard/CLI."""
    coordinator = _coordinator(settings)
    released = coordinator.force_release_lease(lease_id)
    if released:
        append_cluster_event(settings.cluster_paths, "lease_force_released", {"worker_id": settings.worker_id, "lease_id": lease_id})
    return released


def _remove_worker_state(settings: NodeSettings) -> None:
    for path in [
        settings.local_state,
        settings.workspace_root / "stage.yml",
        settings.workspace_root / "node.yml",
        settings.workspace_root / ".env",
        settings.workspace_root / "bookmarks.json",
        settings.workspace_root / "fstab.tradeML",
    ]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            with contextlib.suppress(OSError):
                path.unlink()


def collect_dashboard_snapshot(settings: NodeSettings) -> dict[str, Any]:
    """Collect a restart-safe snapshot for the dashboard UI."""
    runtime = _read_runtime_state(settings)
    pid = runtime.get("pid")
    running = isinstance(pid, int) and _is_process_running(pid)
    runtime["running"] = running
    if running and runtime.get("started_at"):
        started_at = datetime.fromisoformat(str(runtime["started_at"]))
        runtime["uptime_seconds"] = max(0.0, (datetime.now(tz=UTC) - started_at).total_seconds())
    queue_counts = _read_queue_counts(settings.db_path)
    partition_summary = _read_partition_summary(settings.qc_path)
    raw_dates = _partition_dates(settings.raw_equities_root)
    curated_dates = _partition_dates(settings.curated_equities_root)
    reference_files = sorted(path.name for path in settings.reference_root.glob("*.parquet")) if settings.reference_root.exists() else []
    macro_series = _macro_series(settings.macro_root)
    price_check_files = _price_check_files(settings.price_checks_root)
    stage = _read_yaml(settings.stage_path)
    stage_symbols = stage.get("symbols", [])
    stage_years = stage.get("years")
    expected_sessions = None
    progress_ratio = None
    if stage_years:
        end = date.today()
        start = (pd.Timestamp(end) - pd.DateOffset(years=int(stage_years))).date()
        expected_sessions = len(get_trading_days("XNYS", start, end))
        if expected_sessions:
            progress_ratio = len(raw_dates) / expected_sessions
    host = _extract_share_host(settings.nas_share)
    data_readiness = _summarize_data_readiness(
        raw_dates=len(raw_dates),
        expected_sessions=expected_sessions,
        partition_summary=partition_summary,
        reference_files=reference_files,
        macro_series=macro_series,
        price_check_files=price_check_files,
    )
    snapshot = {
        "settings": asdict(settings),
        "runtime": runtime,
        "queue_counts": queue_counts,
        "partition_summary": partition_summary,
        "raw_partitions": len(raw_dates),
        "curated_partitions": len(curated_dates),
        "latest_raw_date": max(raw_dates) if raw_dates else None,
        "latest_curated_date": max(curated_dates) if curated_dates else None,
        "reference_files": reference_files,
        "reference_file_count": len(reference_files),
        "macro_series": macro_series,
        "macro_series_count": len(macro_series),
        "price_check_files": price_check_files,
        "price_check_count": len(price_check_files),
        "stage_symbol_count": len(stage_symbols),
        "stage_years": stage_years,
        "expected_stage_sessions": expected_sessions,
        "stage_progress_ratio": progress_ratio,
        "data_readiness": data_readiness,
        "nas": {
            "share": settings.nas_share,
            "host": host,
            "host_reachable": _check_host_reachable(host) if host else None,
            "mount_path": str(settings.nas_mount),
            "mount_writable": _check_mount_writable(settings.nas_mount),
        },
        "log_tail": _tail_file(settings.log_path),
        "cluster": read_cluster_snapshot(nas_root=settings.nas_mount, worker_id=settings.worker_id),
        "systemd": systemd_status(),
        "journal_tail": systemd_journal_tail(),
    }
    return snapshot


def _coordinator(settings: NodeSettings) -> ClusterCoordinator:
    return ClusterCoordinator(
        nas_root=settings.nas_mount,
        workspace_root=settings.workspace_root,
        config_path=settings.config_path,
        env_path=settings.env_path,
        local_state=settings.local_state,
        nas_share=settings.nas_share,
        worker_id=settings.worker_id,
    )


def _read_queue_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {"PENDING": 0, "LEASED": 0, "FAILED": 0, "DONE": 0, "BOOTSTRAP": 0, "FORWARD": 0, "GAP": 0}
    try:
        with sqlite3.connect(db_path, timeout=5.0) as connection:
            connection.execute("PRAGMA busy_timeout = 5000")
            status_rows = connection.execute("SELECT status, COUNT(*) FROM backfill_queue GROUP BY status").fetchall()
            kind_rows = connection.execute("SELECT kind, COUNT(*) FROM backfill_queue GROUP BY kind").fetchall()
    except sqlite3.OperationalError:
        return {"PENDING": 0, "LEASED": 0, "FAILED": 0, "DONE": 0, "BOOTSTRAP": 0, "FORWARD": 0, "GAP": 0}
    counts = {str(status): int(count) for status, count in status_rows}
    counts.update({str(kind): int(count) for kind, count in kind_rows})
    return counts


def _read_partition_summary(qc_path: Path) -> dict[str, Any]:
    if not qc_path.exists():
        return {"counts": {}, "coverage_green": None, "latest_date": None}
    frame = pd.read_parquet(qc_path)
    if frame.empty:
        return {"counts": {}, "coverage_green": None, "latest_date": None}
    counts = {str(key): int(value) for key, value in frame["status"].value_counts().to_dict().items()}
    coverage_green = float(counts.get("GREEN", 0) / len(frame))
    latest_date = pd.to_datetime(frame["date"]).max().date().isoformat()
    return {"counts": counts, "coverage_green": coverage_green, "latest_date": latest_date}


def _partition_dates(root: Path) -> list[str]:
    if not root.exists():
        return []
    dates: list[str] = []
    for path in root.glob("date=*"):
        _, _, value = path.name.partition("=")
        if value:
            dates.append(value)
    return sorted(set(dates))


def _macro_series(root: Path) -> list[str]:
    if not root.exists():
        return []
    series: list[str] = []
    for path in root.glob("series=*"):
        _, _, value = path.name.partition("=")
        if value:
            series.append(value)
    return sorted(set(series))


def _price_check_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.glob("price_checks_*.parquet"))


def _summarize_data_readiness(
    *,
    raw_dates: int,
    expected_sessions: int | None,
    partition_summary: dict[str, Any],
    reference_files: list[str],
    macro_series: list[str],
    price_check_files: list[str],
) -> dict[str, Any]:
    eod_complete = bool(
        expected_sessions
        and raw_dates >= expected_sessions
        and partition_summary.get("coverage_green") == 1.0
    )
    missing: list[str] = []
    if not eod_complete:
        missing.append("equities EOD backfill")
    if not reference_files:
        missing.append("reference datasets")
    if not macro_series:
        missing.append("macro series")
    if not price_check_files:
        missing.append("cross-vendor price checks")

    if not missing:
        headline = "All tracked Stage 0 datasets present"
        state = "complete"
    elif eod_complete:
        headline = "Equities EOD complete; additional datasets still pending"
        state = "partial"
    else:
        headline = "Initial collection still in progress"
        state = "collecting"
    return {
        "state": state,
        "headline": headline,
        "missing": missing,
        "eod_complete": eod_complete,
    }


def _extract_share_host(nas_share: str) -> str | None:
    share = nas_share.strip().lstrip("\\/")
    if not share:
        return None
    host, _, _rest = share.partition("/")
    host, _, _rest = host.partition("\\")
    return host or None


def _check_host_reachable(host: str, *, port: int = 445, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _check_mount_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            return True
    except OSError:
        return False


def _tail_file(path: Path, *, line_count: int = 200) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _read_runtime_state(settings: NodeSettings) -> dict[str, Any]:
    if not settings.runtime_path.exists():
        return {}
    try:
        return json.loads(settings.runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_runtime_state(settings: NodeSettings, payload: dict[str, Any]) -> None:
    settings.runtime_path.parent.mkdir(parents=True, exist_ok=True)
    settings.runtime_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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
    ordered = sorted(values.items())
    path.write_text("\n".join(f"{key}={value}" for key, value in ordered) + "\n", encoding="utf-8")
