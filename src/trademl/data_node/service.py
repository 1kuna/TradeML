"""Main data-node service loop."""

from __future__ import annotations

import logging
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from trademl.calendars.exchange import get_trading_days, is_trading_day
from trademl.connectors.base import BaseConnector, ConnectorError
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.auxiliary_runtime import AuxiliaryRuntime, ReferenceCollectionResult
from trademl.data_node.capabilities import forward_capabilities
from trademl.data_node.canonical_runtime import CanonicalRuntime
from trademl.data_node.curator import Curator, CuratorResult
from trademl.data_node.db import DataNodeDB
from trademl.data_node.planner import plan_coverage_tasks
from trademl.data_node.training_control import recommended_training_cutoff
from trademl.fleet.cluster import ClusterCoordinator, ShardSpec

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataNodePaths:
    """Filesystem paths for the node."""

    root: Path

    @property
    def raw_equities(self) -> Path:
        return self.root / "data" / "raw" / "equities_bars"

    @property
    def curated_equities(self) -> Path:
        return self.root / "data" / "curated" / "equities_ohlcv_adj"

    @property
    def qc_root(self) -> Path:
        return self.root / "data" / "qc"

    @property
    def reference_root(self) -> Path:
        return self.root / "data" / "reference"

class DataNodeService:
    """Collect raw bars, audit completeness, curate data, and sync QC state."""

    def __init__(
        self,
        *,
        db: DataNodeDB,
        connectors: dict[str, BaseConnector],
        auditor: PartitionAuditor,
        curator: Curator,
        paths: DataNodePaths,
        source_name: str = "alpaca",
        capability_audit_state: dict[str, object] | None = None,
        worker_id: str = "local-worker",
        stage_years: int = 5,
    ) -> None:
        self.db = db
        self.connectors = connectors
        self.auditor = auditor
        self.curator = curator
        self.paths = paths
        self.source_name = source_name
        self.capability_audit_state = capability_audit_state or {}
        self.worker_id = worker_id
        self.stage_years = stage_years
        self._stop_event = threading.Event()
        self.default_symbols: list[str] = []
        self._collection_history: set[str] = set()
        self._maintenance_history: set[str] = set()
        self._reference_history: set[str] = set()
        self._price_check_history: set[str] = set()
        self._planner_seed_history: set[str] = set()
        self._auxiliary_runtime = AuxiliaryRuntime(
            db=self.db,
            connectors=self.connectors,
            paths=self.paths,
            capability_audit_state=self.capability_audit_state,
            worker_id=self.worker_id,
            default_symbols_getter=lambda: self.default_symbols,
            stage_years_getter=lambda: self.stage_years,
            materialize_job_fn=self._materialize_job,
            week_key_fn=self._week_key,
            curate_dates_fn=self.curate_dates,
            collect_macro_data_fn=lambda series_ids, start_date, end_date: self.collect_macro_data(series_ids, start_date, end_date),
            collect_reference_data_fn=lambda jobs: self.collect_reference_data(jobs),
            run_price_checks_fn=lambda trading_date, sample_symbols: self.run_cross_vendor_price_checks(
                trading_date=trading_date,
                sample_symbols=sample_symbols,
            ),
        )
        self._canonical_runtime = CanonicalRuntime(
            db=self.db,
            connectors=self.connectors,
            paths=self.paths,
            source_name=self.source_name,
            capability_audit_state=self.capability_audit_state,
            worker_id=self.worker_id,
            default_symbols_getter=lambda: self.default_symbols,
            stop_requested=self._stop_event.is_set,
            write_raw_partition_fn=lambda frame, source_name: self._write_raw_partition(frame, source_name=source_name),
        )

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

    def _forward_connector_order(self) -> list[str]:
        ordered = [
            capability.vendor
            for capability in forward_capabilities(
                dataset="equities_eod",
                connectors=self.connectors,
                audit_state=self.capability_audit_state,
            )
        ]
        return ordered or [self.source_name]

    def _collect_forward_frame(self, *, trading_date: str, symbols: list[str]) -> tuple[str, pd.DataFrame]:
        last_error: Exception | None = None
        for vendor in self._forward_connector_order():
            connector = self.connectors.get(vendor)
            if connector is None:
                continue
            try:
                frame = connector.fetch("equities_eod", symbols, trading_date, trading_date)
            except ConnectorError as exc:
                last_error = exc
                LOGGER.warning("forward collection failed for vendor=%s trading_date=%s error=%s", vendor, trading_date, exc)
                continue
            if frame.empty:
                continue
            return vendor, frame
        if last_error is not None:
            LOGGER.warning("forward collection exhausted fallback order for trading_date=%s error=%s", trading_date, last_error)
        return self.source_name, pd.DataFrame()

    def _write_raw_partition(self, frame: pd.DataFrame, *, source_name: str) -> list[str]:
        changed_dates: list[str] = []
        for day, day_frame in frame.groupby("date"):
            day_value = pd.Timestamp(day).strftime("%Y-%m-%d")
            partition = self.paths.raw_equities / f"date={day_value}"
            partition.mkdir(parents=True, exist_ok=True)
            merged = self._canonical_runtime._merge_partition_frame(partition=partition, frame=day_frame)
            merged.to_parquet(partition / "data.parquet", index=False)
            changed_dates.append(day_value)
            expected_rows = len(self.default_symbols) if self.default_symbols else len(merged)
            row_count = len(merged)
            status = "GREEN" if row_count >= expected_rows else "AMBER"
            qc_code = "OK" if status == "GREEN" else "INCOMPLETE"
            self.db.update_partition_status(
                source=source_name,
                dataset="equities_eod",
                date=day_value,
                status=status,
                row_count=row_count,
                expected_rows=expected_rows,
                qc_code=qc_code,
            )
        return changed_dates

    def collect_forward(self, *, trading_date: str, symbols: list[str]) -> list[str]:
        """Fetch and persist the primary daily bars."""
        source_name, frame = self._collect_forward_frame(trading_date=trading_date, symbols=symbols)
        if frame.empty:
            return []
        return self._write_raw_partition(frame, source_name=source_name)

    def collect_forward_shard(self, *, trading_date: str, symbols: list[str], shard_id: str) -> list[str]:
        """Fetch and persist a shard-specific daily bar slice."""
        if not symbols:
            return []
        source_name, frame = self._collect_forward_frame(trading_date=trading_date, symbols=symbols)
        if frame.empty:
            return []
        return self._write_raw_shard_partition(frame, shard_id=shard_id, source_name=source_name)

    def process_backfill_queue(self) -> list[str]:
        """Compatibility wrapper that now drains the planner-native queue."""
        if len(self.default_symbols) > 1 and not self.db.has_pending_planner_tasks(task_families=("canonical_bars",)) and self.db.has_pending_backfill():
            self._seed_planner_tasks()
            if self.db.has_pending_planner_tasks(task_families=("canonical_bars",)):
                migrated = self.db.mark_legacy_datewide_backfill_migrated()
                if migrated:
                    LOGGER.info("migrated_legacy_backfill_rows count=%s", migrated)
                return self.process_planner_queue()
        if self.default_symbols and self.db.has_pending_datewide_backfill():
            self._seed_planner_tasks()
            if self.db.has_pending_planner_tasks(task_families=("canonical_bars",)):
                migrated = self.db.mark_legacy_datewide_backfill_migrated()
                if migrated:
                    LOGGER.info("migrated_legacy_datewide_backfill count=%s", migrated)
                return self.process_planner_queue()
        if self.db.has_pending_backfill():
            return self._process_legacy_backfill_queue()
        return self.process_planner_queue()

    def _process_legacy_backfill_queue(self) -> list[str]:
        """Drain legacy date/symbol backfill rows during the migration window."""
        lane_widths = self._canonical_runtime._backfill_lane_widths()
        if not lane_widths:
            return []
        changed_dates: list[str] = []
        futures: dict[object, str] = {}
        with ThreadPoolExecutor(max_workers=sum(lane_widths.values())) as executor:
            while True:
                scheduled = False
                for vendor, width in lane_widths.items():
                    active_for_vendor = sum(1 for active_vendor in futures.values() if active_vendor == vendor)
                    while active_for_vendor < width:
                        task = self._canonical_runtime._lease_next_task_for_vendor(vendor)
                        if task is None:
                            break
                        future = executor.submit(self._canonical_runtime._process_backfill_task_for_vendor, task, vendor)
                        futures[future] = vendor
                        active_for_vendor += 1
                        scheduled = True
                if not futures and not scheduled:
                    break
                if not futures:
                    continue
                done, _pending = wait(list(futures), return_when=FIRST_COMPLETED, timeout=0.1)
                for future in done:
                    changed_dates.extend(future.result())
                    futures.pop(future, None)
            for future in list(futures):
                changed_dates.extend(future.result())
        return sorted(set(changed_dates))

    def process_planner_queue(self, *, trading_date: str | None = None, exchange: str = "XNYS") -> list[str]:
        """Process planner-native canonical and auxiliary work."""
        self._seed_planner_tasks(trading_date=trading_date)
        changed_dates: list[str] = []
        futures: dict[object, str] = {}
        canonical_lane_widths = self._canonical_runtime._backfill_lane_widths()
        canonical_backlog_active = self.db.has_pending_planner_tasks(task_families=("canonical_bars",)) or self.db.has_pending_backfill()
        aux_lane_widths = {} if canonical_backlog_active else self._aux_lane_widths(task_kinds={"REFERENCE", "EVENT", "MACRO"})
        max_workers = max(1, sum(canonical_lane_widths.values()) + sum(aux_lane_widths.values()))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for vendor, width in canonical_lane_widths.items():
                for _ in range(max(1, width)):
                    futures[executor.submit(self._drain_canonical_lane, vendor, exchange)] = "canonical"
            for vendor, width in aux_lane_widths.items():
                for _ in range(max(1, width)):
                    futures[executor.submit(self._drain_auxiliary_lane, vendor)] = "auxiliary"
            for future, task_type in futures.items():
                try:
                    result = future.result()
                except Exception:
                    LOGGER.exception("planner_lane_failed lane=%s worker_id=%s", task_type, self.worker_id)
                    continue
                if task_type == "canonical":
                    changed_dates.extend(result)
        return sorted(set(changed_dates))

    def _seed_planner_tasks(self, *, trading_date: str | None = None) -> None:
        """Seed or refresh planner tasks from the current stage definition."""
        if not self.default_symbols:
            return
        as_of_date = trading_date or datetime.now(tz=UTC).date().isoformat()
        freeze_cutoff = recommended_training_cutoff(
            data_root=self.paths.root,
            expected_symbol_count=len(self.default_symbols),
            as_of=as_of_date,
        )
        planned = plan_coverage_tasks(
            data_root=self.paths.root,
            stage_symbols=self.default_symbols,
            stage_years=self.stage_years,
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            current_date=as_of_date,
            freeze_report_date=freeze_cutoff.get("date"),
        )
        canonical_task_keys = {task.task_key for task in planned if task.task_family == "canonical_bars"}
        if canonical_task_keys:
            self.db.prune_planner_tasks(task_families=("canonical_bars",), valid_task_keys=canonical_task_keys)
        coverage_index = self._canonical_runtime._build_canonical_coverage_index() if any(task.task_family == "canonical_bars" for task in planned) else {}
        task_rows: list[dict[str, object]] = []
        progress_rows: list[dict[str, object]] = []
        canonical_task_map = {
            task.task_key: task
            for task in self.db.fetch_planner_tasks(task_family="canonical_bars")
        }
        regressed_task_keys: list[str] = []
        for task in planned:
            task_family = task.task_family
            if task_family == "auxiliary":
                task_family = self._planner_family_for_dataset(task.dataset)
            task_rows.append(
                {
                    "task_key": task.task_key,
                    "task_family": task_family,
                    "planner_group": task.planner_group,
                    "dataset": task.dataset,
                    "tier": task.tier,
                    "priority": task.priority,
                    "start_date": task.start_date,
                    "end_date": task.end_date,
                    "symbols": list(task.symbols),
                    "eligible_vendors": list(task.preferred_vendors),
                    "output_name": task.output_name,
                    "payload": task.payload,
                }
            )
        self.db.bulk_upsert_planner_tasks(task_rows)
        existing_progress_map = self.db.planner_task_progress_map()
        for task in planned:
            task_family = task.task_family
            if task_family == "auxiliary":
                task_family = self._planner_family_for_dataset(task.dataset)
            if task_family == "canonical_bars":
                existing_task = canonical_task_map.get(task.task_key)
                existing_progress = existing_progress_map.get(task.task_key)
                progress_payload = self._canonical_runtime._canonical_progress_for_scope(
                    symbols=list(task.symbols),
                    trading_days=list(task.payload.get("trading_days", [])),
                    coverage_index=coverage_index,
                )
                if (
                    existing_task is not None
                    and existing_task.status in {"SUCCESS", "PERMANENT_FAILED"}
                    and int(progress_payload["remaining_units"]) > 0
                ):
                    regressed_task_keys.append(task.task_key)
                if existing_progress is not None and existing_progress.completed_units == progress_payload["completed_units"] and existing_progress.remaining_units == progress_payload["remaining_units"]:
                    continue
                progress_rows.append(
                    {
                        "task_key": task.task_key,
                        "expected_units": progress_payload["expected_units"],
                        "completed_units": progress_payload["completed_units"],
                        "remaining_units": progress_payload["remaining_units"],
                        "completed_symbols": progress_payload["completed_symbols"],
                        "remaining_symbols": progress_payload["remaining_symbols"],
                        "state": {"scope_kind": task.scope_kind},
                    }
                )
            else:
                if task.task_key in existing_progress_map:
                    continue
                progress_rows.append(
                    {
                        "task_key": task.task_key,
                        "expected_units": 1,
                        "completed_units": 0,
                        "remaining_units": 1,
                        "state": {"scope_kind": task.scope_kind},
                    }
                )
        self.db.bulk_update_planner_task_progress(progress_rows)
        if regressed_task_keys:
            reopened = self.db.reopen_planner_tasks(sorted(set(regressed_task_keys)), reason="canonical coverage regressed")
            if reopened:
                LOGGER.warning("reopened_regressed_canonical_tasks count=%s", reopened)
        released = self._release_budget_blocked_canonical_tasks()
        if released:
            LOGGER.warning("released_budget_blocked_canonical_tasks count=%s", released)

    def _release_budget_blocked_canonical_tasks(self) -> int:
        """Clear poisoned task-level backoff when another canonical vendor can run now."""
        now_iso = datetime.now(tz=UTC).isoformat()
        releasable: list[str] = []
        page = 0
        limit = 512
        while True:
            candidates = self.db.fetch_planner_tasks(
                task_family="canonical_bars",
                statuses=("PARTIAL", "FAILED", "LEASED"),
                limit=limit,
                offset=page * limit,
            )
            if not candidates:
                break
            for task in candidates:
                if not task.next_eligible_at or task.next_eligible_at <= now_iso:
                    continue
                last_error = str(task.last_error or "")
                if "budget exhausted" not in last_error.lower():
                    continue
                failed_vendor = last_error.split(":", 1)[0].strip() if ":" in last_error else None
                if self._canonical_runtime._canonical_task_has_spendable_vendor(task, excluded_vendor=failed_vendor):
                    releasable.append(task.task_key)
            page += 1
        return self.db.clear_planner_task_backoff(
            sorted(set(releasable)),
            reason="released budget-blocked canonical task to alternate vendor",
        )

    def _drain_canonical_lane(self, vendor: str, exchange: str) -> list[str]:
        """Drain canonical planner tasks for a single vendor lane."""
        changed_dates: list[str] = []
        while not self._stop_event.is_set():
            batch = self._canonical_runtime._lease_canonical_batch(vendor)
            if not batch:
                break
            changed_dates.extend(self._canonical_runtime._process_canonical_planner_batch(batch=batch, vendor=vendor, exchange=exchange))
        return changed_dates

    def _drain_auxiliary_lane(self, vendor: str) -> list[str]:
        """Drain auxiliary planner tasks for a single vendor lane."""
        while not self._stop_event.is_set():
            task = self.db.lease_next_planner_task(
                lease_owner=self.worker_id,
                task_families=("security_master", "corp_actions", "events_filings", "macro", "supplemental_research"),
                vendor=vendor,
            )
            if task is None:
                break
            self._auxiliary_runtime._process_auxiliary_planner_task(task, vendor)
        return []

    def collect_reference_data(self, jobs: list[dict[str, object]]) -> ReferenceCollectionResult:
        """Collect reference datasets into parquet files."""
        return self._auxiliary_runtime.collect_reference_data(jobs)

    def collect_macro_data(self, series_ids: list[str], start_date: str, end_date: str) -> list[Path]:
        """Collect FRED macro series partitions."""
        return self._auxiliary_runtime.collect_macro_data(series_ids, start_date, end_date)

    def run_cross_vendor_price_checks(self, *, trading_date: str, sample_symbols: list[str]) -> Path:
        """Compare the primary source against backup vendors for a sample of symbols."""
        return self._auxiliary_runtime.run_cross_vendor_price_checks(
            source_name=self.source_name,
            trading_date=trading_date,
            sample_symbols=sample_symbols,
        )

    def curate_dates(
        self,
        corp_actions: pd.DataFrame | None = None,
        *,
        changed_dates: list[str] | None = None,
    ) -> CuratorResult:
        """Rebuild curated partitions from the current raw dataset."""
        actions = corp_actions if corp_actions is not None else self.load_corp_actions_reference()
        if changed_dates:
            raw_files = [
                self.paths.raw_equities / f"date={pd.Timestamp(day).strftime('%Y-%m-%d')}" / "data.parquet"
                for day in changed_dates
            ]
            raw_files = [path for path in raw_files if path.exists()]
            frames: list[pd.DataFrame] = []
            logs: list[pd.DataFrame] = []
            for path in raw_files:
                day_key = path.parent.name.partition("=")[2]
                raw_frame = self._canonical_runtime._read_partition_frame(path)
                if raw_frame.empty:
                    continue
                result = self.curator.write_curated(
                    raw_bars=raw_frame,
                    corp_actions=actions,
                    output_root=self.paths.curated_equities,
                    changed_dates=[day_key],
                    adjustment_log_path=None,
                )
                if not result.frame.empty:
                    frames.append(result.frame)
                if not result.adjustment_log.empty:
                    logs.append(result.adjustment_log)
            combined_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            combined_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
            if not combined_log.empty:
                dedupe_columns = [column for column in ["symbol", "date", "event_type", "ratio", "source"] if column in combined_log.columns]
                combined_log = combined_log.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
            adjustment_log_path = self.paths.root / "data" / "curated" / "adjustment_log.parquet"
            adjustment_log_path.parent.mkdir(parents=True, exist_ok=True)
            combined_log.to_parquet(adjustment_log_path, index=False)
            return CuratorResult(frame=combined_frame, adjustment_log=combined_log)

        raw_files = sorted(self.paths.raw_equities.glob("date=*/data.parquet"))
        raw_frames = [frame for frame in (self._canonical_runtime._read_partition_frame(path) for path in raw_files) if not frame.empty]
        raw_frame = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        return self.curator.write_curated(
            raw_bars=raw_frame,
            corp_actions=actions,
            output_root=self.paths.curated_equities,
            changed_dates=changed_dates,
            adjustment_log_path=self.paths.root / "data" / "curated" / "adjustment_log.parquet",
        )

    def sync_partition_status(self) -> Path:
        """Mirror the local SQLite QC ledger to parquet."""
        self.paths.qc_root.mkdir(parents=True, exist_ok=True)
        output = self.paths.qc_root / "partition_status.parquet"
        rows = [dict(row) for row in self.db.fetch_partition_status()]
        pd.DataFrame(rows).to_parquet(output, index=False)
        return output

    def load_corp_actions_reference(self) -> pd.DataFrame:
        """Load any persisted corp-action reference parquet files into the curator schema."""
        frames: list[pd.DataFrame] = []
        for filename in ("corp_actions.parquet", "splits.parquet", "dividends.parquet"):
            path = self.paths.reference_root / filename
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                LOGGER.warning("Skipping unreadable corp-actions reference parquet %s: %s", path, exc)
                continue
            if frame.empty:
                continue
            if path.stem == "corp_actions":
                normalized = frame.copy()
                if "ex_date" in normalized.columns:
                    normalized["ex_date"] = pd.to_datetime(normalized["ex_date"]).dt.date
                frames.append(normalized)
                continue
            if path.stem == "splits":
                normalized = pd.DataFrame(
                    {
                        "symbol": frame["symbol"],
                        "event_type": "split",
                        "ex_date": pd.to_datetime(frame.get("execution_date", frame.get("ex_date", frame.get("exDate")))).dt.date,
                        "ratio": pd.to_numeric(frame.get("ratio"), errors="coerce"),
                        "amount": pd.NA,
                        "source": frame.get("source", pd.Series(["splits"] * len(frame))),
                    }
                )
                if normalized["ratio"].isna().all() and {"split_from", "split_to"}.issubset(frame.columns):
                    normalized["ratio"] = pd.to_numeric(frame["split_from"], errors="coerce") / pd.to_numeric(frame["split_to"], errors="coerce")
                frames.append(normalized.dropna(subset=["ex_date", "ratio"]))
                continue
            if path.stem == "dividends":
                frames.append(
                    pd.DataFrame(
                        {
                            "symbol": frame["symbol"],
                            "event_type": "dividend",
                            "ex_date": pd.to_datetime(frame.get("ex_dividend_date", frame.get("ex_date", frame.get("exDate")))).dt.date,
                            "ratio": pd.NA,
                            "amount": pd.to_numeric(frame.get("cash_amount", frame.get("amount")), errors="coerce"),
                            "source": frame.get("source", pd.Series(["dividends"] * len(frame))),
                        }
                    ).dropna(subset=["ex_date", "amount"])
                )
        if not frames:
            return pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount", "source"])
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["symbol", "ex_date", "event_type"]).reset_index(drop=True)

    def run_cycle(
        self,
        *,
        trading_date: str,
        symbols: list[str],
        exchange: str,
        audit_start: str,
        audit_end: str,
        corp_actions: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """Run a single deterministic collection cycle."""
        self.default_symbols = symbols
        forward_dates = self.collect_forward(trading_date=trading_date, symbols=symbols)
        audit_result = self.auditor.audit_range(
            exchange=exchange,
            source=self.source_name,
            dataset="equities_eod",
            start_date=audit_start,
            end_date=audit_end,
            expected_rows=len(symbols),
        )
        backfill_dates = self.process_backfill_queue()
        changed_dates = sorted(set(forward_dates + backfill_dates))
        curated = self.curate_dates(
            corp_actions=corp_actions,
            changed_dates=None if corp_actions is not None and not corp_actions.empty else changed_dates,
        )
        qc_path = self.sync_partition_status()
        return {
            "forward_dates": forward_dates,
            "backfill_dates": backfill_dates,
            "audit_result": audit_result,
            "curated_rows": len(curated.frame),
            "qc_path": qc_path,
        }

    def run_cluster_forever(
        self,
        *,
        coordinator: ClusterCoordinator,
        symbols: list[str],
        exchange: str,
        collection_time_et: str = "16:30",
        maintenance_hour_local: int = 2,
        poll_seconds: float = 60.0,
        audit_lookback_days: int = 7,
        macro_series_ids: list[str] | None = None,
        reference_jobs: list[dict[str, object]] | None = None,
        price_check_symbols: list[str] | None = None,
        now_fn=lambda: datetime.now(tz=UTC),
        sleep_fn=sleep,
    ) -> None:
        """Run the clustered scheduled data-node loop until a stop signal is received."""
        self.default_symbols = symbols
        owned_shards: list[ShardSpec] = []

        def before_iteration(*, current: datetime, current_et: datetime, current_local: datetime, trading_date: str) -> None:
            nonlocal owned_shards
            coordinator.heartbeat_worker()
            owned_shards = coordinator.sync_shard_leases()

        def collect_iteration(*, trading_date: str, current_et: datetime, current_local: datetime) -> None:
            for shard in owned_shards:
                self._collect_cluster_shard(trading_date=trading_date, shard=shard)
            if coordinator.acquire_singleton("audit_curate", trading_date):
                audit_start = (pd.Timestamp(trading_date) - pd.Timedelta(days=audit_lookback_days)).strftime("%Y-%m-%d")
                self.auditor.audit_range(
                    exchange=exchange,
                    source=self.source_name,
                    dataset="equities_eod",
                    start_date=audit_start,
                    end_date=trading_date,
                    expected_rows=len(symbols),
                )
                self.curate_dates(corp_actions=self.load_corp_actions_reference(), changed_dates=[trading_date])
                self.sync_partition_status()
                coordinator.mark_singleton_success("audit_curate", trading_date, {"symbol_count": len(symbols)})
            self._auxiliary_runtime._run_cluster_auxiliary_tasks(
                coordinator=coordinator,
                trading_date=trading_date,
                current_et=current_et,
                macro_series_ids=macro_series_ids,
                reference_jobs=reference_jobs,
                price_check_symbols=price_check_symbols,
                source_name=self.source_name,
            )

        def backfill_iteration(*, local_day: str, current_local: datetime, pending_backfill: bool) -> None:
            if self._acquire_backfill_singleton(coordinator=coordinator, bucket_key=local_day, pending_backfill=pending_backfill):
                changed_dates = self.process_backfill_queue()
                if changed_dates:
                    self.curate_dates(
                        corp_actions=self.load_corp_actions_reference(),
                        changed_dates=changed_dates,
                    )
                self.sync_partition_status()
                coordinator.mark_singleton_success("backfill", local_day, {"changed_dates": len(changed_dates)})
                self._maintenance_history.add(local_day)

        def after_backlog_clear(*, trading_date: str, current_et: datetime, current_local: datetime) -> None:
            self._auxiliary_runtime._run_cluster_auxiliary_tasks(
                coordinator=coordinator,
                trading_date=trading_date,
                current_et=current_et,
                macro_series_ids=macro_series_ids,
                reference_jobs=reference_jobs,
                price_check_symbols=price_check_symbols,
                source_name=self.source_name,
            )

        self._run_scheduler_loop(
            exchange=exchange,
            collection_time_et=collection_time_et,
            maintenance_hour_local=maintenance_hour_local,
            poll_seconds=poll_seconds,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            collect_fn=collect_iteration,
            backfill_fn=backfill_iteration,
            before_iteration_fn=before_iteration,
            backfill_when_pending=True,
            after_backlog_clear_fn=after_backlog_clear,
        )

    def run_forever(
        self,
        *,
        symbols: list[str],
        exchange: str,
        collection_time_et: str = "16:30",
        maintenance_hour_local: int = 2,
        poll_seconds: float = 60.0,
        audit_lookback_days: int = 7,
        corp_actions: pd.DataFrame | None = None,
        macro_series_ids: list[str] | None = None,
        reference_jobs: list[dict[str, object]] | None = None,
        price_check_symbols: list[str] | None = None,
        now_fn=lambda: datetime.now(tz=UTC),
        sleep_fn=sleep,
    ) -> None:
        """Run the scheduled data-node loop until a stop signal is received."""
        def collect_iteration(*, trading_date: str, current_et: datetime, current_local: datetime) -> None:
            audit_start = (pd.Timestamp(trading_date) - pd.Timedelta(days=audit_lookback_days)).strftime("%Y-%m-%d")
            self.run_cycle(
                trading_date=trading_date,
                symbols=symbols,
                exchange=exchange,
                audit_start=audit_start,
                audit_end=trading_date,
                corp_actions=corp_actions,
            )
            if macro_series_ids and "fred" in self.connectors:
                self.collect_macro_data(macro_series_ids, trading_date, trading_date)
            if reference_jobs and self._should_run_reference(current_et):
                materialized_jobs = [self._materialize_job(job, trading_date) for job in reference_jobs]
                self.collect_reference_data(materialized_jobs)
                if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                    corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                    if corp_actions_path.exists():
                        self.curate_dates(corp_actions=pd.read_parquet(corp_actions_path))
                self._reference_history.add(self._week_key(current_et))
            if price_check_symbols and self._should_run_price_checks(current_et):
                self.run_cross_vendor_price_checks(
                    trading_date=trading_date,
                    sample_symbols=price_check_symbols,
                )
                self._price_check_history.add(self._week_key(current_et))
            self.sync_partition_status()

        def backfill_iteration(*, local_day: str, current_local: datetime, pending_backfill: bool) -> None:
            changed_dates = self.process_backfill_queue()
            if changed_dates:
                self.curate_dates(corp_actions=corp_actions, changed_dates=changed_dates)
            self.sync_partition_status()
            self._maintenance_history.add(local_day)

        self._run_scheduler_loop(
            exchange=exchange,
            collection_time_et=collection_time_et,
            maintenance_hour_local=maintenance_hour_local,
            poll_seconds=poll_seconds,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            collect_fn=collect_iteration,
            backfill_fn=backfill_iteration,
            backfill_when_pending=False,
        )

    def _run_scheduler_loop(
        self,
        *,
        exchange: str,
        collection_time_et: str,
        maintenance_hour_local: int,
        poll_seconds: float,
        now_fn: Callable[[], datetime],
        sleep_fn: Callable[[float], None],
        collect_fn: Callable[..., None],
        backfill_fn: Callable[..., None],
        before_iteration_fn: Callable[..., None] | None = None,
        backfill_when_pending: bool,
        after_backlog_clear_fn: Callable[..., None] | None = None,
    ) -> None:
        """Run the shared scheduled service loop for local and clustered modes."""
        market_tz = ZoneInfo("America/New_York")
        collection_hour, collection_minute = (int(part) for part in collection_time_et.split(":", 1))
        requeued_failures = self.db.requeue_retryable_failures()
        if requeued_failures:
            LOGGER.info("requeued_retryable_failures count=%s", requeued_failures)

        while not self._stop_event.is_set():
            current = now_fn()
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            current_et = current.astimezone(market_tz)
            current_local = current.astimezone()
            trading_date = current_et.date().isoformat()

            if before_iteration_fn is not None:
                before_iteration_fn(
                    current=current,
                    current_et=current_et,
                    current_local=current_local,
                    trading_date=trading_date,
                )

            self._ensure_planner_backlog_seeded(trading_date=trading_date)

            if self._should_run_collection(
                trading_date=trading_date,
                current_et=current_et,
                collection_hour=collection_hour,
                collection_minute=collection_minute,
                exchange=exchange,
            ):
                collect_fn(
                    trading_date=trading_date,
                    current_et=current_et,
                    current_local=current_local,
                )
                self._collection_history.add(trading_date)

            local_day = current_local.date().isoformat()
            pending_backfill = self.db.has_pending_planner_tasks() or self.db.has_pending_backfill()
            should_run_backfill = self._should_run_maintenance(
                local_day=local_day,
                current_local=current_local,
                maintenance_hour_local=maintenance_hour_local,
            ) or (backfill_when_pending and pending_backfill)
            if should_run_backfill:
                backfill_fn(
                    local_day=local_day,
                    current_local=current_local,
                    pending_backfill=pending_backfill,
                )

            if (
                after_backlog_clear_fn is not None
                and not self.db.has_pending_planner_tasks(task_families=("canonical_bars",))
                and not self.db.has_pending_backfill()
            ):
                anchor_date = self._latest_raw_date()
                if anchor_date:
                    after_backlog_clear_fn(
                        trading_date=anchor_date,
                        current_et=current_et,
                        current_local=current_local,
                    )

            if not self._stop_event.is_set():
                sleep_fn(poll_seconds)

    def _should_run_collection(
        self,
        *,
        trading_date: str,
        current_et: datetime,
        collection_hour: int,
        collection_minute: int,
        exchange: str,
    ) -> bool:
        if trading_date in self._collection_history:
            return False
        if not is_trading_day(exchange, trading_date):
            return False
        if current_et.hour < collection_hour:
            return False
        if current_et.hour == collection_hour and current_et.minute < collection_minute:
            return False
        return True

    def _ensure_planner_backlog_seeded(self, *, trading_date: str) -> None:
        """Refresh planner tasks once per service instance and trading date."""
        if not self.default_symbols:
            return
        if trading_date in self._planner_seed_history:
            return
        self._seed_planner_tasks(trading_date=trading_date)
        self._planner_seed_history.add(trading_date)

    def _should_run_maintenance(self, *, local_day: str, current_local: datetime, maintenance_hour_local: int) -> bool:
        return local_day not in self._maintenance_history and current_local.hour >= maintenance_hour_local

    @staticmethod
    def _materialize_job(job: dict[str, object], trading_date: str) -> dict[str, object]:
        materialized = dict(job)
        materialized.setdefault("start_date", trading_date)
        materialized.setdefault("end_date", trading_date)
        symbols = list(materialized.get("symbols", []))
        max_symbols = int(materialized.get("max_symbols_per_run", 0) or 0)
        if max_symbols > 0 and len(symbols) > max_symbols:
            materialized["symbols"] = DataNodeService._rotate_symbols(
                symbols,
                limit=max_symbols,
                trading_date=trading_date,
                rotation_key=str(materialized.get("rotation_key", f"{materialized.get('source', '')}:{materialized.get('dataset', '')}")),
            )
        return materialized

    def _should_run_reference(self, current_et: datetime) -> bool:
        return self._week_key(current_et) not in self._reference_history

    def _should_run_price_checks(self, current_et: datetime) -> bool:
        return self._week_key(current_et) not in self._price_check_history

    def _acquire_backfill_singleton(self, *, coordinator: ClusterCoordinator, bucket_key: str, pending_backfill: bool) -> bool:
        """Acquire backfill execution rights, reopening same-day buckets when new backlog was seeded later."""
        if coordinator.acquire_singleton("backfill", bucket_key):
            return True
        if not pending_backfill:
            return False
        if not self._backfill_needs_rerun(coordinator=coordinator, bucket_key=bucket_key):
            renew = getattr(coordinator, "acquire_or_renew_lease", None)
            if callable(renew):
                return bool(renew(f"singleton::backfill::{bucket_key}"))
            return False
        renew = getattr(coordinator, "acquire_or_renew_lease", None)
        if callable(renew):
            return bool(renew(f"singleton::backfill::{bucket_key}"))
        return False

    def _backfill_needs_rerun(self, *, coordinator: ClusterCoordinator, bucket_key: str) -> bool:
        """Return whether pending backlog was updated after the last successful backfill bucket."""
        read_last_success = getattr(coordinator, "read_last_success", None)
        if not callable(read_last_success):
            return False
        state = read_last_success().get("backfill", {})
        if str(state.get("bucket")) != bucket_key:
            return False
        last_success = state.get("updated_at")
        latest_planner_update = self.db.latest_planner_update(statuses=("PENDING", "PARTIAL", "FAILED"))
        latest_queue_update = self.db.latest_queue_update(statuses=("PENDING", "FAILED"))
        latest_update = max(str(latest_planner_update or ""), str(latest_queue_update or ""))
        if not last_success or not latest_update:
            return False
        return latest_update > str(last_success)

    @staticmethod
    def _week_key(current_et: datetime) -> str:
        iso = current_et.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _latest_raw_date(self) -> str | None:
        latest: str | None = None
        for path in self.paths.raw_equities.glob("date=*"):
            _, _, value = path.name.partition("=")
            if value and (latest is None or value > latest):
                latest = value
        return latest

    @staticmethod
    def _planner_family_for_dataset(dataset: str) -> str:
        """Map auxiliary datasets into planner task families."""
        if dataset in {"assets", "listings", "reference_tickers", "stocks", "delistings", "symbol_changes", "company_tickers"}:
            return "security_master"
        if dataset in {"corp_actions", "reference_dividends", "reference_splits", "dividends", "splits"}:
            return "corp_actions"
        if dataset in {"filing_index", "companyfacts", "earnings_calendar"}:
            return "events_filings"
        if dataset in {"macros_treasury", "vintagedates"}:
            return "macro"
        return "supplemental_research"

    @staticmethod
    def _rotate_symbols(symbols: list[str], *, limit: int, trading_date: str, rotation_key: str) -> list[str]:
        ordered = list(dict.fromkeys(symbols))
        if len(ordered) <= limit:
            return ordered
        iso = pd.Timestamp(trading_date).isocalendar()
        bucket = iso.year * 53 + iso.week
        offset_seed = sum(ord(char) for char in rotation_key)
        start = ((bucket + offset_seed) * limit) % len(ordered)
        window = ordered[start : start + limit]
        if len(window) < limit:
            window.extend(ordered[: limit - len(window)])
        return window

    def _write_raw_shard_partition(self, frame: pd.DataFrame, *, shard_id: str, source_name: str) -> list[str]:
        return self._canonical_runtime._write_raw_shard_partition(frame, shard_id=shard_id, source_name=source_name)

    def _merge_raw_shards_for_date(self, day_value: str) -> Path:
        return self._canonical_runtime._merge_raw_shards_for_date(day_value)

    def _collect_cluster_shard(self, *, trading_date: str, shard: ShardSpec) -> None:
        partition = self.paths.raw_equities / f"date={trading_date}" / "shards" / f"{shard.shard_id}.parquet"
        if partition.exists():
            return
        self.collect_forward_shard(trading_date=trading_date, symbols=shard.symbols, shard_id=shard.shard_id)

    def _aux_lane_widths(self, *, task_kinds: set[str]) -> dict[str, int]:
        return self._auxiliary_runtime._aux_lane_widths(task_kinds=task_kinds)
