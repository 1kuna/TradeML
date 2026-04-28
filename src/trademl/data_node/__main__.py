"""CLI entry point for the TradeML data node."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import threading
from time import sleep
from typing import Callable

import yaml

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.bootstrap import Stage0UniverseBuilder
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.capabilities import (
    build_reference_jobs as build_reference_jobs_from_registry,
)
from trademl.data_node.capabilities import default_macro_series, load_audit_state
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.runtime import (
    build_connectors,
    build_connector,
    resolve_vendor_budgets,
)
from trademl.data_node.service import DataNodePaths, DataNodeService
from trademl.env import load_dotenv
from trademl.fleet.cluster import ClusterCoordinator

_STACK_DUMP_HANDLE = None


def _build_reference_jobs(
    *, connectors: dict[str, object], symbols: list[str]
) -> list[dict[str, object]]:
    """Build the default weekly reference collection plan from verified connector lanes."""
    return build_reference_jobs_from_registry(connectors=connectors, symbols=symbols)


def main() -> int:
    """Run one deterministic data-node cycle or the scheduled service loop."""
    parser = argparse.ArgumentParser(description="Run the TradeML data node.")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--passphrase", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    env_path = None
    if args.env_file:
        env_path = Path(args.env_file).expanduser()
    elif args.root:
        env_path = Path(args.root).expanduser() / ".env"
    elif Path(".env").exists():
        env_path = Path(".env")
    load_dotenv(env_path)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _apply_collection_runtime_env(config)

    workspace_root = (
        Path(args.root).expanduser()
        if args.root
        else Path(os.getenv("LOCAL_STATE", config["node"]["local_state"]))
        .expanduser()
        .parent
    )
    local_state_default = (
        workspace_root / "control"
        if args.root
        else Path(config["node"]["local_state"]).expanduser()
    )
    local_state = Path(os.getenv("LOCAL_STATE", str(local_state_default))).expanduser()
    _install_stack_dump_handler(local_state=local_state)
    vendor_limits = resolve_vendor_budgets(config)
    budgets = BudgetManager(
        vendor_limits,
        snapshot_path=local_state / "budget_state.json",
    )
    connectors = build_connectors(
        env_values=os.environ,
        vendor_limits=vendor_limits,
        budget_manager_factory=lambda _vendor: budgets,
    )
    connector = build_connector(
        vendor="alpaca",
        env_values=os.environ,
        vendor_limits=vendor_limits,
        budget_manager_factory=lambda _vendor: budgets,
    )
    if connector is None:
        raise SystemExit("ALPACA_API_KEY is required to start the data node")
    stage0_universe_builder = Stage0UniverseBuilder(connector=connector)
    worker_id = os.getenv(
        "EDGE_NODE_ID",
        config.get("node", {}).get(
            "worker_id", os.uname().nodename if hasattr(os, "uname") else "worker"
        ),
    )
    started_at = datetime.now(tz=UTC).isoformat()
    runtime_heartbeat = _runtime_heartbeat_writer(
        local_state=local_state,
        workspace_root=workspace_root,
        config_path=Path(args.config).expanduser(),
        env_path=env_path or (workspace_root / ".env"),
        worker_id=str(worker_id),
        started_at=started_at,
    )
    _start_runtime_watchdog(local_state=local_state)
    runtime_heartbeat({"running": True, "mode": "startup"})
    db = DataNodeDB(local_state / "node.sqlite")
    data_root = Path(
        args.data_root or os.getenv("NAS_MOUNT", config["node"]["nas_mount"])
    ).expanduser()
    audit_state = load_audit_state(
        data_root / "control" / "cluster" / "state" / "vendor_audit.json"
    )
    symbols = args.symbols or _load_stage_symbols(workspace_root)
    stage_years = _load_stage_years(workspace_root)
    service = DataNodeService(
        db=db,
        connectors=connectors,
        auditor=PartitionAuditor(
            db=db,
            calendar_store=ExchangeCalendarStore(
                root=data_root / "reference" / "calendars"
            ),
        ),
        curator=Curator(),
        paths=DataNodePaths(root=data_root),
        capability_audit_state=audit_state,
        worker_id=worker_id,
        stage_years=stage_years,
    )
    service.install_signal_handlers()
    reference_jobs = build_reference_jobs_from_registry(
        connectors=connectors, symbols=symbols, audit_state=audit_state
    )

    if args.once or args.date:
        if not args.date:
            raise SystemExit("--date is required with --once")
        service.run_cycle(
            trading_date=args.date,
            symbols=symbols,
            exchange="XNYS",
            audit_start=args.date,
            audit_end=args.date,
        )
        if "fred" in connectors:
            service.collect_macro_data(default_macro_series(), args.date, args.date)
        if reference_jobs:
            materialized = [
                service._materialize_job(job, args.date) for job in reference_jobs
            ]
            service.collect_reference_data(materialized)
            service.curate_dates(corp_actions=service.load_corp_actions_reference())
        if {"massive", "twelve_data", "tiingo"}.intersection(connectors):
            service.run_cross_vendor_price_checks(
                trading_date=args.date, sample_symbols=symbols[:5]
            )
        service.sync_partition_status()
        runtime_heartbeat({"running": False, "mode": "once", "stopped_at": datetime.now(tz=UTC).isoformat()})
        return 0

    coordinator = ClusterCoordinator(
        nas_root=data_root,
        workspace_root=workspace_root,
        config_path=Path(args.config).expanduser(),
        env_path=env_path or (workspace_root / ".env"),
        local_state=local_state,
        nas_share=os.getenv(
            "NAS_SHARE", config["node"].get("nas_share", "//nas/trademl")
        ),
        worker_id=worker_id,
        lease_ttl_seconds=int(config["node"].get("lease_ttl_seconds", 90)),
        heartbeat_interval_seconds=int(
            config["node"].get("heartbeat_interval_seconds", 30)
        ),
        universe_builder=stage0_universe_builder,
    )
    manifest = coordinator.ensure_cluster_ready(passphrase=args.passphrase)
    def cluster_heartbeat() -> object:
        runtime_heartbeat({"running": True, "mode": "cluster_startup"})
        return coordinator.heartbeat_worker()

    local_db_path = local_state / "node.sqlite"
    if _should_rebuild_local_state(local_db_path=local_db_path):
        coordinator.rebuild_local_state(local_db_path=local_db_path)
    db = DataNodeDB(local_state / "node.sqlite")
    service.db = db
    service.auditor.db = db
    active_symbols = symbols or manifest["stage"]["symbols"]
    service.default_symbols = list(active_symbols)
    service._ensure_planner_backlog_seeded_with_heartbeat(
        trading_date=(args.date or datetime.now(tz=UTC).date().isoformat()),
        heartbeat_fn=cluster_heartbeat,
    )
    try:
        service.run_cluster_forever(
            coordinator=coordinator,
            symbols=active_symbols,
            exchange=manifest["datasets"]["equities_eod"].get("exchange", "XNYS"),
            collection_time_et=os.getenv(
                "COLLECTION_TIME_ET", manifest["schedule"]["collection_time_et"]
            ),
            maintenance_hour_local=int(
                os.getenv(
                    "MAINTENANCE_HOUR_LOCAL", manifest["schedule"]["maintenance_hour_local"]
                )
            ),
            poll_seconds=args.poll_seconds,
            macro_series_ids=default_macro_series() if "fred" in connectors else [],
            reference_jobs=reference_jobs,
            price_check_symbols=(
                active_symbols[:5]
                if {"massive", "twelve_data", "tiingo"}.intersection(connectors)
                else []
            ),
            runtime_heartbeat_fn=runtime_heartbeat,
        )
    finally:
        runtime_heartbeat({"running": False, "stopped_at": datetime.now(tz=UTC).isoformat()})
    return 0


def _should_rebuild_local_state(*, local_db_path: Path) -> bool:
    """Return whether startup should rebuild the disposable local DB from NAS."""
    if os.getenv("TRADEML_FORCE_REBUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return not local_db_path.exists()


def _install_stack_dump_handler(*, local_state: Path) -> None:
    """Install an operator-triggered stack dump for stuck Pi workers."""
    global _STACK_DUMP_HANDLE
    log_dir = local_state / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _STACK_DUMP_HANDLE = (log_dir / "stack_dumps.log").open("a", encoding="utf-8")
    faulthandler.register(signal.SIGUSR1, file=_STACK_DUMP_HANDLE, all_threads=True)


def _load_stage_symbols(root: Path) -> list[str]:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return []
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return list(stage.get("symbols", []))


def _load_stage_years(root: Path) -> int:
    stage_path = root / "stage.yml"
    if not stage_path.exists():
        return 5
    with stage_path.open("r", encoding="utf-8") as handle:
        stage = yaml.safe_load(handle) or {}
    return int(stage.get("years", 5) or 5)


def _resolve_vendor_budgets(config: dict[str, object]) -> dict[str, dict[str, int]]:
    return resolve_vendor_budgets(config)


def _apply_collection_runtime_env(config: dict[str, object]) -> None:
    """Expose collection control knobs to runtime helpers."""
    collection = config.get("collection", {}) or {}
    if not isinstance(collection, dict):
        return
    watermark = collection.get("storage_watermark", {}) or {}
    if not isinstance(watermark, dict):
        return
    pause_threshold = watermark.get("pause_low_priority_percent")
    if pause_threshold is not None:
        os.environ.setdefault(
            "TRADEML_STORAGE_PAUSE_LOW_PRIORITY_PERCENT", str(pause_threshold)
        )


def _runtime_heartbeat_writer(
    *,
    local_state: Path,
    workspace_root: Path,
    config_path: Path,
    env_path: Path,
    worker_id: str,
    started_at: str,
) -> Callable[[dict[str, object]], None]:
    """Build an atomic node runtime heartbeat writer."""
    runtime_path = local_state / "node_runtime.json"
    base: dict[str, object] = {
        "pid": os.getpid(),
        "started_at": started_at,
        "worker_id": worker_id,
        "workspace_root": str(workspace_root),
        "config_path": str(config_path),
        "env_path": str(env_path),
        "log_path": str(local_state / "logs" / "node.log"),
        "command": list(sys.argv),
        "managed_by": "process",
    }

    def write(update: dict[str, object]) -> None:
        payload = {**base, **update, "heartbeat_at": datetime.now(tz=UTC).isoformat()}
        _write_runtime_json(runtime_path, payload)

    return write


def _start_runtime_watchdog(
    *,
    local_state: Path,
    stale_after_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> threading.Thread:
    """Start a daemon watchdog that records stale live runtime heartbeats."""
    threshold = int(
        stale_after_seconds
        if stale_after_seconds is not None
        else os.getenv("TRADEML_NODE_WATCHDOG_STALE_SECONDS", "300")
    )
    interval = int(
        poll_seconds
        if poll_seconds is not None
        else os.getenv("TRADEML_NODE_WATCHDOG_POLL_SECONDS", "60")
    )
    runtime_path = local_state / "node_runtime.json"
    alerts_root = local_state / "alerts"
    thread = threading.Thread(
        target=_runtime_watchdog_loop,
        kwargs={
            "runtime_path": runtime_path,
            "alerts_root": alerts_root,
            "stale_after_seconds": max(1, threshold),
            "poll_seconds": max(1, interval),
        },
        name="trademl-node-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _runtime_watchdog_loop(
    *,
    runtime_path: Path,
    alerts_root: Path,
    stale_after_seconds: int,
    poll_seconds: int,
) -> None:
    last_alert_key: str | None = None
    while True:
        alert = _runtime_watchdog_alert(
            runtime_path=runtime_path,
            now=datetime.now(tz=UTC),
            stale_after_seconds=stale_after_seconds,
        )
        if alert is not None:
            alert_key = str(alert.get("heartbeat_at"))
            if alert_key != last_alert_key:
                _write_watchdog_alert(alerts_root=alerts_root, alert=alert)
                last_alert_key = alert_key
        sleep(poll_seconds)


def _runtime_watchdog_alert(
    *, runtime_path: Path, now: datetime, stale_after_seconds: int
) -> dict[str, object] | None:
    """Return an alert payload when the node heartbeat is stale but PID is alive."""
    if not runtime_path.exists():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not runtime.get("running"):
        return None
    pid = runtime.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return None
    heartbeat_text = str(runtime.get("heartbeat_at") or "")
    try:
        heartbeat_at = datetime.fromisoformat(heartbeat_text)
    except ValueError:
        return None
    if heartbeat_at.tzinfo is None:
        heartbeat_at = heartbeat_at.replace(tzinfo=UTC)
    age_seconds = int((now - heartbeat_at).total_seconds())
    if age_seconds < int(stale_after_seconds):
        return None
    return {
        "alert_type": "node_runtime_heartbeat_stale",
        "created_at": now.isoformat(),
        "age_seconds": age_seconds,
        "stale_after_seconds": int(stale_after_seconds),
        "pid": pid,
        "mode": runtime.get("mode"),
        "worker_id": runtime.get("worker_id"),
        "heartbeat_at": heartbeat_text,
        "runtime_path": str(runtime_path),
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_watchdog_alert(*, alerts_root: Path, alert: dict[str, object]) -> None:
    alerts_root.mkdir(parents=True, exist_ok=True)
    created = str(alert["created_at"]).replace(":", "").replace("+", "_")
    json_path = alerts_root / f"{created}_node_watchdog.json"
    md_path = alerts_root / f"{created}_node_watchdog.md"
    latest_path = alerts_root / "node_watchdog_latest.json"
    json_path.write_text(json.dumps(alert, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(alert, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# TradeML Node Watchdog Alert",
                "",
                f"- Type: {alert['alert_type']}",
                f"- Worker: {alert.get('worker_id')}",
                f"- Mode: {alert.get('mode')}",
                f"- PID: {alert.get('pid')}",
                f"- Heartbeat age seconds: {alert.get('age_seconds')}",
                f"- Last heartbeat: {alert.get('heartbeat_at')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_runtime_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write node runtime JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
