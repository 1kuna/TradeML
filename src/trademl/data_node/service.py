"""Main data-node service loop."""

from __future__ import annotations

from collections import defaultdict
import contextlib
import logging
import os
import signal
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.dataset as ds

from trademl.calendars.exchange import get_trading_days, is_trading_day
from trademl.connectors.base import BaseConnector, ConnectorError, PermanentConnectorError, TemporaryConnectorError
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.capabilities import auxiliary_capabilities, backfill_capabilities, default_macro_series, forward_capabilities
from trademl.data_node.curator import Curator, CuratorResult
from trademl.data_node.db import DataNodeDB, PlannerTask
from trademl.data_node.planner import (
    canonical_task_key,
    choose_vendor_for_canonical_task,
    plan_auxiliary_tasks,
    plan_coverage_tasks,
)
from trademl.fleet.cluster import ClusterCoordinator, ShardSpec
from trademl.reference.security_master import rebuild_derived_references

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


@dataclass(slots=True)
class ReferenceCollectionResult:
    """Outcome of a reference-data collection pass."""

    outputs: list[Path]
    failures: list[str]
    deferred: list[str]


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
            merged = self._merge_partition_frame(partition=partition, frame=day_frame)
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
        lane_widths = self._backfill_lane_widths()
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
                        task = self._lease_next_task_for_vendor(vendor)
                        if task is None:
                            break
                        future = executor.submit(self._process_backfill_task_for_vendor, task, vendor)
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
        canonical_lane_widths = self._backfill_lane_widths()
        aux_lane_widths = self._aux_lane_widths(task_kinds={"REFERENCE", "EVENT", "MACRO"})
        max_workers = max(1, sum(canonical_lane_widths.values()) + sum(aux_lane_widths.values()))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for vendor, width in canonical_lane_widths.items():
                for _ in range(max(1, width)):
                    futures[executor.submit(self._drain_canonical_lane, vendor, exchange)] = "canonical"
            for vendor, width in aux_lane_widths.items():
                for _ in range(max(1, width)):
                    futures[executor.submit(self._drain_auxiliary_lane, vendor)] = "auxiliary"
            for future, task_type in futures.items():
                result = future.result()
                if task_type == "canonical":
                    changed_dates.extend(result)
        return sorted(set(changed_dates))

    def _seed_planner_tasks(self, *, trading_date: str | None = None) -> None:
        """Seed or refresh planner tasks from the current stage definition."""
        if not self.default_symbols:
            return
        as_of_date = trading_date or datetime.now(tz=UTC).date().isoformat()
        planned = plan_coverage_tasks(
            data_root=self.paths.root,
            stage_symbols=self.default_symbols,
            stage_years=self.stage_years,
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            current_date=as_of_date,
        )
        coverage_index = self._build_canonical_coverage_index() if any(task.task_family == "canonical_bars" for task in planned) else {}
        task_rows: list[dict[str, object]] = []
        progress_rows: list[dict[str, object]] = []
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
                existing_progress = existing_progress_map.get(task.task_key)
                progress_payload = self._canonical_progress_for_scope(
                    symbols=list(task.symbols),
                    trading_days=list(task.payload.get("trading_days", [])),
                    coverage_index=coverage_index,
                )
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

    def _drain_canonical_lane(self, vendor: str, exchange: str) -> list[str]:
        """Drain canonical planner tasks for a single vendor lane."""
        changed_dates: list[str] = []
        while not self._stop_event.is_set():
            batch = self._lease_canonical_batch(vendor)
            if not batch:
                break
            changed_dates.extend(self._process_canonical_planner_batch(batch=batch, vendor=vendor, exchange=exchange))
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
            self._process_auxiliary_planner_task(task, vendor)
        return []

    def _lease_canonical_batch(self, vendor: str) -> list[PlannerTask]:
        """Lease a vendor-compatible batch of canonical planner tasks."""
        capability = self._canonical_capability(vendor)
        if capability is None:
            return []
        batch_limit = self._capability_batch_size(capability)
        now_iso = datetime.now(tz=UTC).isoformat()
        base_task: PlannerTask | None = None
        candidates: list[PlannerTask] = []
        scan_limit = max(256, batch_limit * 8)
        for page in range(16):
            page_candidates = self.db.fetch_planner_tasks(
                task_family="canonical_bars",
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                limit=scan_limit,
                offset=page * scan_limit,
            )
            if not page_candidates:
                break
            candidates.extend(page_candidates)
            for candidate in page_candidates:
                if not self._planner_task_vendor_eligible(candidate, vendor=vendor, now_iso=now_iso):
                    continue
                leased = self.db.lease_planner_task_by_key(task_key=candidate.task_key, lease_owner=self.worker_id)
                if leased is not None:
                    base_task = leased
                    break
            if base_task is not None:
                break
        if base_task is None:
            return []
        batch = [base_task]
        if capability.batching_mode != "multi_symbol":
            return batch
        symbol_count = sum(len(task.symbols) for task in batch)
        for candidate in candidates:
            if symbol_count >= batch_limit:
                break
            if candidate.task_key == base_task.task_key:
                continue
            if candidate.dataset != base_task.dataset or candidate.start_date != base_task.start_date or candidate.end_date != base_task.end_date:
                continue
            if not self._planner_task_vendor_eligible(candidate, vendor=vendor, now_iso=now_iso):
                continue
            leased = self.db.lease_planner_task_by_key(task_key=candidate.task_key, lease_owner=self.worker_id)
            if leased is None:
                continue
            batch.append(leased)
            symbol_count += len(leased.symbols)
        return batch

    def collect_reference_data(self, jobs: list[dict[str, object]]) -> ReferenceCollectionResult:
        """Collect reference datasets into parquet files."""
        outputs: list[Path] = []
        self.paths.reference_root.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        deferred: dict[str, int] = {}
        expanded_jobs = self._expand_reference_jobs(jobs)
        lane_widths = self._aux_lane_widths(task_kinds={"REFERENCE", "EVENT"})
        vendor_jobs: dict[str, list[dict[str, object]]] = {}
        for job in expanded_jobs:
            vendor_jobs.setdefault(str(job["source"]), []).append(job)

        with ThreadPoolExecutor(max_workers=max(1, sum(lane_widths.get(vendor, 1) for vendor in vendor_jobs))) as executor:
            futures: dict[object, str] = {}
            positions = {vendor: 0 for vendor in vendor_jobs}
            while True:
                scheduled = False
                for vendor, jobs_for_vendor in vendor_jobs.items():
                    width = max(1, lane_widths.get(vendor, 1))
                    active_for_vendor = sum(1 for active_vendor in futures.values() if active_vendor == vendor)
                    while active_for_vendor < width and positions[vendor] < len(jobs_for_vendor):
                        job = jobs_for_vendor[positions[vendor]]
                        positions[vendor] += 1
                        future = executor.submit(self._run_reference_job, job)
                        futures[future] = vendor
                        active_for_vendor += 1
                        scheduled = True
                if not futures and not scheduled:
                    break
                if not futures:
                    continue
                done, _pending = wait(list(futures), return_when=FIRST_COMPLETED, timeout=0.1)
                for future in done:
                    vendor = futures.pop(future)
                    result = future.result()
                    for output in result.get("outputs", []):
                        outputs.append(Path(output))
                    for failure in result.get("failures", []):
                        failures.append(failure)
                    if result.get("deferred"):
                        deferred[vendor] = deferred.get(vendor, 0) + int(result["deferred"])
        outputs.extend(rebuild_derived_references(self.paths.reference_root))
        deferred_messages = [f"{vendor}: deferred {count} jobs after budget exhaustion" for vendor, count in sorted(deferred.items())]
        return ReferenceCollectionResult(outputs=outputs, failures=failures, deferred=deferred_messages)

    def collect_macro_data(self, series_ids: list[str], start_date: str, end_date: str) -> list[Path]:
        """Collect FRED macro series partitions."""
        connector = self.connectors["fred"]
        frame = connector.fetch("macros_treasury", series_ids, start_date, end_date)
        outputs: list[Path] = []
        if frame.empty or "series_id" not in frame.columns:
            return outputs
        for series_id, series_frame in frame.groupby("series_id"):
            partition = self.paths.root / "data" / "raw" / "macros_fred" / f"series={series_id}"
            partition.mkdir(parents=True, exist_ok=True)
            output = partition / "data.parquet"
            series_frame.to_parquet(output, index=False)
            outputs.append(output)
        vintages = connector.fetch("vintagedates", series_ids, start_date, end_date)
        if not vintages.empty and {"series_id", "vintage_date"}.issubset(vintages.columns):
            self.paths.reference_root.mkdir(parents=True, exist_ok=True)
            vintage_output = self.paths.reference_root / "fred_vintagedates.parquet"
            if vintage_output.exists():
                existing = pd.read_parquet(vintage_output)
                vintages = pd.concat([existing, vintages], ignore_index=True)
            vintages.sort_values(["series_id", "vintage_date"]).drop_duplicates(["series_id", "vintage_date"]).to_parquet(vintage_output, index=False)
            outputs.append(vintage_output)
        return outputs

    def run_cross_vendor_price_checks(self, *, trading_date: str, sample_symbols: list[str]) -> Path:
        """Compare the primary source against backup vendors for a sample of symbols."""
        comparisons: list[pd.DataFrame] = []
        primary = self.connectors[self.source_name].fetch("equities_eod", sample_symbols, trading_date, trading_date)
        for vendor in ["massive", "finnhub", "twelve_data"]:
            if vendor not in self.connectors:
                continue
            try:
                backup = self.connectors[vendor].fetch("equities_eod", sample_symbols, trading_date, trading_date)
            except Exception:
                continue
            merged = primary[["symbol", "close"]].merge(
                backup[["symbol", "close"]],
                on="symbol",
                how="inner",
                suffixes=("_primary", f"_{vendor}"),
            )
            if not merged.empty:
                merged["vendor"] = vendor
                merged["date"] = trading_date
                comparisons.append(merged)
        self.paths.qc_root.mkdir(parents=True, exist_ok=True)
        output = self.paths.qc_root / f"price_checks_{trading_date}.parquet"
        if comparisons:
            pd.concat(comparisons, ignore_index=True).to_parquet(output, index=False)
        else:
            pd.DataFrame().to_parquet(output, index=False)
        return output

    def curate_dates(
        self,
        corp_actions: pd.DataFrame | None = None,
        *,
        changed_dates: list[str] | None = None,
    ) -> CuratorResult:
        """Rebuild curated partitions from the current raw dataset."""
        raw_files = sorted(self.paths.raw_equities.glob("date=*/data.parquet"))
        raw_frame = pd.concat((pd.read_parquet(path) for path in raw_files), ignore_index=True) if raw_files else pd.DataFrame()
        result = self.curator.write_curated(
            raw_bars=raw_frame,
            corp_actions=corp_actions if corp_actions is not None else self.load_corp_actions_reference(),
            output_root=self.paths.curated_equities,
            changed_dates=changed_dates,
            adjustment_log_path=self.paths.root / "data" / "curated" / "adjustment_log.parquet",
        )
        return result

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
        for path in sorted(self.paths.reference_root.glob("*.parquet")):
            frame = pd.read_parquet(path)
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
        market_tz = ZoneInfo("America/New_York")
        collection_hour, collection_minute = (int(part) for part in collection_time_et.split(":", 1))
        self.default_symbols = symbols
        requeued_failures = self.db.requeue_retryable_failures()
        if requeued_failures:
            LOGGER.info("requeued_retryable_failures count=%s", requeued_failures)

        while not self._stop_event.is_set():
            coordinator.heartbeat_worker()
            current = now_fn()
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            current_et = current.astimezone(market_tz)
            current_local = current.astimezone()
            trading_date = current_et.date().isoformat()
            owned_shards = coordinator.sync_shard_leases()
            self._ensure_planner_backlog_seeded(trading_date=trading_date)

            if self._should_run_collection(
                trading_date=trading_date,
                current_et=current_et,
                collection_hour=collection_hour,
                collection_minute=collection_minute,
                exchange=exchange,
            ):
                for shard in owned_shards:
                    self._collect_cluster_shard(trading_date=trading_date, shard=shard)
                self._collection_history.add(trading_date)

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
                    self.curate_dates(corp_actions=self.load_corp_actions_reference())
                    self.sync_partition_status()
                    coordinator.mark_singleton_success("audit_curate", trading_date, {"symbol_count": len(symbols)})
                self._run_cluster_auxiliary_tasks(
                    coordinator=coordinator,
                    trading_date=trading_date,
                    current_et=current_et,
                    macro_series_ids=macro_series_ids,
                    reference_jobs=reference_jobs,
                    price_check_symbols=price_check_symbols,
                )

            local_day = current_local.date().isoformat()
            should_run_maintenance = self._should_run_maintenance(
                local_day=local_day,
                current_local=current_local,
                maintenance_hour_local=maintenance_hour_local,
            )
            should_drain_backlog = self.db.has_pending_planner_tasks() or self.db.has_pending_backfill()
            if should_run_maintenance or should_drain_backlog:
                if self._acquire_backfill_singleton(coordinator=coordinator, bucket_key=local_day, pending_backfill=should_drain_backlog):
                    changed_dates = self.process_backfill_queue()
                    if changed_dates:
                        self.curate_dates(corp_actions=self.load_corp_actions_reference())
                    self.sync_partition_status()
                    coordinator.mark_singleton_success("backfill", local_day, {"changed_dates": len(changed_dates)})
                    self._maintenance_history.add(local_day)

            # Once EOD bars are fully present, opportunistically fill the slower
            # auxiliary datasets instead of waiting for the next scheduled lane.
            if not self.db.has_pending_planner_tasks(task_families=("canonical_bars",)) and not self.db.has_pending_backfill():
                anchor_date = self._latest_raw_date()
                if anchor_date:
                    self._run_cluster_auxiliary_tasks(
                        coordinator=coordinator,
                        trading_date=anchor_date,
                        current_et=current_et,
                        macro_series_ids=macro_series_ids,
                        reference_jobs=reference_jobs,
                        price_check_symbols=price_check_symbols,
                    )

            if not self._stop_event.is_set():
                sleep_fn(poll_seconds)

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
            trading_date = current_et.date()
            self._ensure_planner_backlog_seeded(trading_date=trading_date.isoformat())

            if self._should_run_collection(
                trading_date=trading_date.isoformat(),
                current_et=current_et,
                collection_hour=collection_hour,
                collection_minute=collection_minute,
                exchange=exchange,
            ):
                audit_start = (trading_date - pd.Timedelta(days=audit_lookback_days)).strftime("%Y-%m-%d")
                self.run_cycle(
                    trading_date=trading_date.isoformat(),
                    symbols=symbols,
                    exchange=exchange,
                    audit_start=audit_start,
                    audit_end=trading_date.isoformat(),
                    corp_actions=corp_actions,
                )
                if macro_series_ids and "fred" in self.connectors:
                    self.collect_macro_data(macro_series_ids, trading_date.isoformat(), trading_date.isoformat())
                if reference_jobs and self._should_run_reference(current_et):
                    materialized_jobs = [self._materialize_job(job, trading_date.isoformat()) for job in reference_jobs]
                    self.collect_reference_data(materialized_jobs)
                    if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                        corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                        if corp_actions_path.exists():
                            self.curate_dates(corp_actions=pd.read_parquet(corp_actions_path))
                    self._reference_history.add(self._week_key(current_et))
                if price_check_symbols and self._should_run_price_checks(current_et):
                    self.run_cross_vendor_price_checks(
                        trading_date=trading_date.isoformat(),
                        sample_symbols=price_check_symbols,
                    )
                    self._price_check_history.add(self._week_key(current_et))
                self.sync_partition_status()
                self._collection_history.add(trading_date.isoformat())

            local_day = current_local.date().isoformat()
            if self._should_run_maintenance(local_day=local_day, current_local=current_local, maintenance_hour_local=maintenance_hour_local):
                changed_dates = self.process_backfill_queue()
                if changed_dates:
                    self.curate_dates(corp_actions=corp_actions)
                self.sync_partition_status()
                self._maintenance_history.add(local_day)

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
        """Seed planner tasks when the local planner ledger is empty."""
        if not self.default_symbols:
            return
        if self.db.fetch_planner_tasks(limit=1):
            return
        self._seed_planner_tasks(trading_date=trading_date)

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

    def _process_canonical_planner_task(self, task: PlannerTask, exchange: str) -> list[str]:
        """Compatibility wrapper for a single planner task."""
        vendor = next((name for name in task.eligible_vendors if self._canonical_capability(name) is not None), None)
        if vendor is None:
            self.db.mark_planner_task_failed(task.task_key, error="no eligible canonical vendors", backoff_minutes=5)
            return []
        return self._process_canonical_planner_batch(batch=[task], vendor=vendor, exchange=exchange)

    def _process_canonical_planner_batch(self, *, batch: list[PlannerTask], vendor: str, exchange: str) -> list[str]:
        """Process a vendor-compatible batch of atomic canonical tasks."""
        if not batch:
            return []
        connector = self.connectors[vendor]
        dataset = batch[0].dataset
        start_date = batch[0].start_date
        end_date = batch[0].end_date
        leased_tasks: list[PlannerTask] = []
        symbols: list[str] = []
        for task in batch:
            attempt = self.db.lease_vendor_attempt(
                task_key=task.task_key,
                task_family=task.task_family,
                planner_group=task.planner_group,
                vendor=vendor,
                lease_owner=self.worker_id,
                payload={"symbols": list(task.symbols), "start_date": task.start_date, "end_date": task.end_date},
            )
            if attempt is None:
                self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: attempt unavailable", backoff_minutes=5)
                continue
            leased_tasks.append(task)
            symbols.extend(list(task.symbols))
        if not leased_tasks:
            return []

        try:
            frame = connector.fetch(dataset, symbols, start_date, end_date)
        except PermanentConnectorError as exc:
            for task in leased_tasks:
                self.db.mark_vendor_attempt_failed(task_key=task.task_key, vendor=vendor, error=str(exc), backoff_minutes=0, permanent=True)
                if self._canonical_task_has_remaining_vendors(task, failed_vendor=vendor):
                    self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: {exc}", backoff_minutes=1)
                else:
                    self.db.mark_planner_task_failed(task.task_key, error=f"{vendor}: {exc}", backoff_minutes=15, permanent=True)
            return []
        except ConnectorError as exc:
            message = str(exc)
            backoff = 30 if "budget exhausted" in message else 15
            for task in leased_tasks:
                self.db.mark_vendor_attempt_failed(task_key=task.task_key, vendor=vendor, error=message, backoff_minutes=backoff)
                self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: {message}", backoff_minutes=backoff)
            return []

        changed_dates: list[str] = []
        if not frame.empty:
            changed_dates = self._write_raw_partition(frame, source_name=vendor)
        progress_map = self._canonical_batch_progress(tasks=leased_tasks, exchange=exchange)
        normalized_symbols = (
            frame.get("symbol", pd.Series(dtype="string")).dropna().astype("string").str.upper().tolist()
            if not frame.empty
            else []
        )
        for task in leased_tasks:
            task_symbols = {symbol.upper() for symbol in task.symbols}
            task_rows = int(sum(1 for symbol in normalized_symbols if symbol in task_symbols))
            if task_rows > 0:
                self.db.mark_vendor_attempt_success(task_key=task.task_key, vendor=vendor, rows_returned=task_rows)
            else:
                self.db.mark_vendor_attempt_failed(
                    task_key=task.task_key,
                    vendor=vendor,
                    error="empty planner canonical result",
                    backoff_minutes=0,
                    permanent=True,
                )
                if self._canonical_task_has_remaining_vendors(task, failed_vendor=vendor):
                    self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: empty planner canonical result", backoff_minutes=1)
                    continue
                self.db.mark_planner_task_failed(
                    task.task_key,
                    error=f"{vendor}: empty planner canonical result",
                    backoff_minutes=15,
                    permanent=True,
                )
                continue
            progress_payload = progress_map[task.task_key]
            self.db.update_planner_task_progress(
                task_key=task.task_key,
                expected_units=progress_payload["expected_units"],
                completed_units=progress_payload["completed_units"],
                remaining_units=progress_payload["remaining_units"],
                completed_symbols=progress_payload["completed_symbols"],
                remaining_symbols=progress_payload["remaining_symbols"],
                state={"trading_days": progress_payload["trading_days"], "scope_kind": task.payload.get("scope_kind", "symbol_range")},
            )
            if progress_payload["remaining_units"] == 0:
                self.db.mark_planner_task_success(task.task_key)
            else:
                self.db.mark_planner_task_partial(
                    task.task_key,
                    error=f"remaining_units={progress_payload['remaining_units']}",
                    backoff_minutes=5,
                )
        return changed_dates

    def _canonical_task_progress(self, *, task: PlannerTask, exchange: str) -> dict[str, object]:
        """Compute symbol-date coverage for a canonical planner task from raw partitions."""
        trading_days = list(task.payload.get("trading_days", []))
        if not trading_days:
            trading_days = [day.isoformat() for day in get_trading_days(exchange, pd.Timestamp(task.start_date).date(), pd.Timestamp(task.end_date).date())]
        return self._canonical_progress_for_scope(symbols=list(task.symbols), trading_days=trading_days)

    def _canonical_progress_for_scope(
        self,
        *,
        symbols: list[str],
        trading_days: list[str],
        coverage_index: dict[str, set[str]] | None = None,
    ) -> dict[str, object]:
        """Compute canonical symbol-date coverage for a given task scope."""
        symbol_set = {symbol.upper() for symbol in symbols}
        present_pairs: set[tuple[str, str]] = set()
        counts_by_symbol = {symbol.upper(): 0 for symbol in symbols}
        if coverage_index is None:
            coverage_index = self._build_canonical_coverage_index(trading_days=trading_days)
        for symbol in symbol_set:
            present_days = coverage_index.get(symbol, set())
            for day in trading_days:
                if day not in present_days:
                    continue
                pair = (day, symbol)
                if pair in present_pairs:
                    continue
                present_pairs.add(pair)
                counts_by_symbol[symbol] = counts_by_symbol.get(symbol, 0) + 1
        completed_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count >= len(trading_days))
        remaining_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count < len(trading_days))
        expected_units = len(symbol_set) * len(trading_days)
        completed_units = len(present_pairs)
        remaining_units = max(0, expected_units - completed_units)
        return {
            "trading_days": trading_days,
            "completed_symbols": completed_symbols,
            "remaining_symbols": remaining_symbols,
            "expected_units": expected_units,
            "completed_units": completed_units,
            "remaining_units": remaining_units,
        }

    def _canonical_batch_progress(self, *, tasks: list[PlannerTask], exchange: str) -> dict[str, dict[str, object]]:
        """Compute canonical progress for a homogeneous task batch."""
        trading_days = list(tasks[0].payload.get("trading_days", []))
        if not trading_days:
            trading_days = [
                day.isoformat()
                for day in get_trading_days(exchange, pd.Timestamp(tasks[0].start_date).date(), pd.Timestamp(tasks[0].end_date).date())
            ]
        coverage_index = self._build_canonical_coverage_index(trading_days=trading_days)
        return {
            task.task_key: self._canonical_progress_for_scope(
                symbols=list(task.symbols),
                trading_days=trading_days,
                coverage_index=coverage_index,
            )
            for task in tasks
        }

    def _canonical_task_has_remaining_vendors(self, task: PlannerTask, *, failed_vendor: str) -> bool:
        """Return whether another eligible vendor could still complete this canonical task."""
        attempts = {attempt.vendor: attempt for attempt in self.db.vendor_attempts_for_task(task.task_key)}
        for candidate in task.eligible_vendors:
            if candidate == failed_vendor:
                continue
            if self._canonical_capability(candidate) is None:
                continue
            attempt = attempts.get(candidate)
            if attempt is None:
                return True
            if attempt.status in {"SUCCESS", "PERMANENT_FAILED"}:
                continue
            return True
        return False

    def _build_canonical_coverage_index(self, *, trading_days: list[str] | None = None) -> dict[str, set[str]]:
        """Return the existing canonical coverage index keyed by symbol."""
        coverage: dict[str, set[str]] = defaultdict(set)
        if not self.paths.raw_equities.exists():
            return coverage
        try:
            dataset = ds.dataset(self.paths.raw_equities, format="parquet", partitioning="hive")
            filter_expression = None
            if trading_days:
                filter_expression = ds.field("date").isin(list(trading_days))
            table = dataset.to_table(columns=["symbol", "date"], filter=filter_expression)
            if table.num_rows == 0:
                return coverage
            frame = table.to_pandas()
            if frame.empty:
                return coverage
            normalized = frame.dropna(subset=["symbol", "date"]).copy()
            normalized["symbol"] = normalized["symbol"].astype("string").str.upper()
            normalized["date"] = normalized["date"].astype("string")
            for row in normalized.drop_duplicates(subset=["symbol", "date"]).itertuples(index=False):
                coverage[str(row.symbol)].add(str(row.date))
            return coverage
        except Exception:  # pragma: no cover - fallback for older pyarrow/dataset quirks
            days = trading_days or [path.name.partition("=")[2] for path in sorted(self.paths.raw_equities.glob("date=*")) if "=" in path.name]
            for day in days:
                path = self.paths.raw_equities / f"date={day}" / "data.parquet"
                if not path.exists():
                    continue
                try:
                    frame = pd.read_parquet(path, columns=["symbol"])
                except Exception as exc:  # pragma: no cover - exercised via service tests
                    LOGGER.warning("skipping_unreadable_raw_partition path=%s error=%s", path, exc)
                    continue
                if frame.empty:
                    continue
                for symbol in frame["symbol"].dropna().astype("string").str.upper().tolist():
                    coverage[symbol].add(day)
        return coverage

    def _process_auxiliary_planner_task(self, task: PlannerTask, vendor: str) -> list[str]:
        """Run a single planner-native auxiliary task."""
        leased = self.db.lease_vendor_attempt(
            task_key=task.task_key,
            task_family=task.task_family,
            planner_group=task.planner_group,
            vendor=vendor,
            lease_owner=self.worker_id,
            payload={"symbols": list(task.symbols), "dataset": task.dataset},
        )
        if leased is None:
            self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: attempt unavailable", backoff_minutes=5)
            return []
        if task.task_family == "macro":
            try:
                self.collect_macro_data(list(task.symbols), task.start_date, task.end_date)
            except ConnectorError as exc:
                self.db.mark_vendor_attempt_failed(task_key=task.task_key, vendor=vendor, error=str(exc), backoff_minutes=15)
                self.db.mark_planner_task_failed(task.task_key, error=str(exc), backoff_minutes=15)
                return []
            self.db.mark_vendor_attempt_success(task_key=task.task_key, vendor=vendor, rows_returned=len(task.symbols))
            self.db.update_planner_task_progress(
                task_key=task.task_key,
                expected_units=1,
                completed_units=1,
                remaining_units=0,
                state={"dataset": task.dataset},
            )
            self.db.mark_planner_task_success(task.task_key)
            return []

        job = {
            "source": vendor,
            "dataset": task.dataset,
            "symbols": list(task.symbols),
            "start_date": task.start_date,
            "end_date": task.end_date,
            "output_name": task.output_name or task.dataset,
            "planner_group": task.planner_group,
            "explode_symbols": False,
            "tier": task.tier,
        }
        result = self._run_reference_job(job)
        if result["failures"]:
            self.db.mark_planner_task_failed(task.task_key, error=" | ".join(result["failures"]), backoff_minutes=15)
        elif result["deferred"]:
            self.db.mark_planner_task_partial(task.task_key, error="deferred", backoff_minutes=30)
        elif result["outputs"]:
            self.db.update_planner_task_progress(
                task_key=task.task_key,
                expected_units=1,
                completed_units=1,
                remaining_units=0,
                state={"outputs": result["outputs"]},
            )
            self.db.mark_planner_task_success(task.task_key)
        else:
            self.db.mark_planner_task_partial(task.task_key, error="empty result", backoff_minutes=15)
        return []

    def _run_cluster_auxiliary_tasks(
        self,
        *,
        coordinator: ClusterCoordinator,
        trading_date: str,
        current_et: datetime,
        macro_series_ids: list[str] | None,
        reference_jobs: list[dict[str, object]] | None,
        price_check_symbols: list[str] | None,
    ) -> None:
        planner_managed_aux = self.db.has_pending_planner_tasks(
            task_families=("security_master", "corp_actions", "events_filings", "macro", "supplemental_research")
        )
        planned_macro_series, planned_reference_jobs = self._planned_auxiliary_work(trading_date=trading_date)
        effective_macro_series = planned_macro_series or list(macro_series_ids or [])
        effective_reference_jobs = planned_reference_jobs or list(reference_jobs or [])
        if not planner_managed_aux and effective_macro_series and "fred" in self.connectors and coordinator.acquire_singleton("macro", trading_date):
            try:
                self.collect_macro_data(effective_macro_series, trading_date, trading_date)
            except ConnectorError:
                LOGGER.exception("macro collection failed for trading_date=%s", trading_date)
            else:
                coordinator.mark_singleton_success("macro", trading_date, {"series_count": len(effective_macro_series)})

        week_key = self._week_key(current_et)
        reference_bucket = trading_date if self._core_auxiliary_incomplete() else week_key
        if not planner_managed_aux and effective_reference_jobs and coordinator.acquire_singleton("reference", reference_bucket):
            materialized_jobs = [self._materialize_job(job, trading_date) for job in effective_reference_jobs]
            result = self.collect_reference_data(materialized_jobs)
            if isinstance(result, list):
                result = ReferenceCollectionResult(outputs=result, failures=[], deferred=[])
            if result.failures or result.deferred:
                LOGGER.warning(
                    "reference collection partial for bucket=%s trading_date=%s failures=%s deferred=%s",
                    reference_bucket,
                    trading_date,
                    " | ".join(result.failures) if result.failures else "-",
                    " | ".join(result.deferred) if result.deferred else "-",
                )
            if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                if corp_actions_path.exists():
                    self.curate_dates(corp_actions=pd.read_parquet(corp_actions_path))
            coordinator.mark_singleton_success(
                "reference",
                reference_bucket,
                {
                    "job_count": len(materialized_jobs),
                    "output_count": len(result.outputs),
                    "failure_count": len(result.failures),
                    "deferred_count": len(result.deferred),
                },
            )

        if price_check_symbols and coordinator.acquire_singleton("price_checks", week_key):
            try:
                self.run_cross_vendor_price_checks(trading_date=trading_date, sample_symbols=price_check_symbols)
            except ConnectorError:
                LOGGER.exception("price check collection failed for bucket=%s trading_date=%s", week_key, trading_date)
            else:
                coordinator.mark_singleton_success("price_checks", week_key, {"symbol_count": len(price_check_symbols)})

    def _planned_auxiliary_work(self, *, trading_date: str) -> tuple[list[str], list[dict[str, object]]]:
        if not self.default_symbols:
            return [], []
        planned = plan_auxiliary_tasks(
            data_root=self.paths.root,
            stage_symbols=self.default_symbols,
            stage_years=self.stage_years,
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            current_date=trading_date,
        )
        macro_series: list[str] = []
        reference_jobs: list[dict[str, object]] = []
        for task in planned:
            if task.dataset in {"macros_treasury", "vintagedates"}:
                macro_series.extend(list(task.symbols))
                continue
            if task.task_family != "auxiliary" or not task.preferred_vendors:
                continue
            reference_jobs.append(
                {
                    "source": task.preferred_vendors[0],
                    "dataset": task.dataset,
                    "symbols": list(task.symbols),
                    "output_name": task.output_name or task.dataset,
                    "planner_group": task.planner_group,
                    "explode_symbols": False,
                    "tier": task.tier,
                }
            )
        return sorted(set(macro_series)), reference_jobs

    def _core_auxiliary_incomplete(self) -> bool:
        required_reference = {"listing_history.parquet", "delistings.parquet", "corp_actions.parquet", "sec_filings.parquet", "fred_vintagedates.parquet"}
        available_reference = {path.name for path in self.paths.reference_root.glob("*.parquet")} if self.paths.reference_root.exists() else set()
        if not required_reference.issubset(available_reference):
            return True
        available_macro = {path.name.partition("=")[2] for path in (self.paths.root / "data" / "raw" / "macros_fred").glob("series=*")}
        return not set(default_macro_series()).issubset(available_macro)

    @staticmethod
    def _expand_reference_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
        expanded: list[dict[str, object]] = []
        for job in jobs:
            symbols = list(job.get("symbols", []))
            if not job.get("explode_symbols", True) or len(symbols) <= 1:
                expanded.append(dict(job))
                continue
            for symbol in symbols:
                item = dict(job)
                item["symbols"] = [symbol]
                expanded.append(item)
        return expanded

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
        changed_dates: list[str] = []
        for day, day_frame in frame.groupby("date"):
            day_value = pd.Timestamp(day).strftime("%Y-%m-%d")
            partition = self.paths.raw_equities / f"date={day_value}"
            shard_root = partition / "shards"
            shard_root.mkdir(parents=True, exist_ok=True)
            shard_path = shard_root / f"{shard_id}.parquet"
            tmp_path = shard_root / f"{shard_id}.{uuid.uuid4().hex}.tmp"
            day_frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, shard_path)
            self._merge_raw_shards_for_date(day_value)
            changed_dates.append(day_value)
        return changed_dates

    def _merge_raw_shards_for_date(self, day_value: str) -> Path:
        partition = self.paths.raw_equities / f"date={day_value}"
        shard_root = partition / "shards"
        lock_path = partition / ".merge.lock"
        self._acquire_file_lock(lock_path)
        try:
            frames = [pd.read_parquet(path) for path in sorted(shard_root.glob("*.parquet"))]
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            tmp_path = partition / f"data.{uuid.uuid4().hex}.tmp"
            merged.to_parquet(tmp_path, index=False)
            output = partition / "data.parquet"
            os.replace(tmp_path, output)
            return output
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    @staticmethod
    def _acquire_file_lock(path: Path, *, stale_after_seconds: int = 15) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                if path.exists() and (datetime.now(tz=UTC).timestamp() - path.stat().st_mtime) > stale_after_seconds:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    continue
                sleep(0.05)

    def _collect_cluster_shard(self, *, trading_date: str, shard: ShardSpec) -> None:
        partition = self.paths.raw_equities / f"date={trading_date}" / "shards" / f"{shard.shard_id}.parquet"
        if partition.exists():
            return
        self.collect_forward_shard(trading_date=trading_date, symbols=shard.symbols, shard_id=shard.shard_id)

    def _backfill_connector_order(self, *, dataset: str) -> list[str]:
        if dataset == "equities_eod":
            capabilities = backfill_capabilities(
                dataset=dataset,
                connectors=self.connectors,
                audit_state=self.capability_audit_state,
            )
            ordered = [capability.vendor for capability in capabilities]
            if ordered:
                return ordered
        preferred = [self.source_name]
        return [name for index, name in enumerate(preferred) if name in self.connectors and name not in preferred[:index]]

    def _backfill_lane_widths(self) -> dict[str, int]:
        widths: dict[str, int] = {}
        for capability in backfill_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            widths[capability.vendor] = max(widths.get(capability.vendor, 0), int(capability.lane_width or 1))
        return widths

    def _aux_lane_widths(self, *, task_kinds: set[str]) -> dict[str, int]:
        widths: dict[str, int] = {}
        for job in auxiliary_capabilities(
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            include_research=False,
        ):
            if job.task_kind not in task_kinds:
                continue
            widths[job.vendor] = max(widths.get(job.vendor, 0), int(job.lane_width or 1))
        return widths

    def _lease_next_task_for_vendor(self, vendor: str):
        for candidate in self.db.peek_next_tasks(limit=64):
            chosen = choose_vendor_for_canonical_task(
                task=candidate,
                connectors=self.connectors,
                audit_state=self.capability_audit_state,
                attempts=self.db.vendor_attempts_for_task(canonical_task_key(candidate)),
            )
            if chosen != vendor:
                continue
            leased = self.db.lease_task_by_id(candidate.id)
            if leased is not None:
                return leased
        return None

    def _process_backfill_task_for_vendor(self, task, vendor: str) -> list[str]:
        task_key = canonical_task_key(task)
        capability = next(
            (
                candidate
                for candidate in backfill_capabilities(
                    dataset=task.dataset,
                    connectors=self.connectors,
                    audit_state=self.capability_audit_state,
                )
                if candidate.vendor == vendor
            ),
            None,
        )
        if task.symbol is None:
            return self._process_parallel_datewide_backfill_task(task=task, coordinator_vendor=vendor)
        if task.symbol is None and capability is not None and capability.batching_mode == "single_symbol":
            self.db.mark_vendor_attempt_failed(
                task_key=task_key,
                vendor=vendor,
                error="unsupported canonical task shape for single-symbol vendor",
                backoff_minutes=60,
            )
            self.db.defer_task(task.id, reason=f"{vendor}: unsupported canonical task shape", backoff_minutes=5)
            return []
        symbols = [task.symbol] if task.symbol else self.default_symbols
        attempt = self.db.lease_vendor_attempt(
            task_key=task_key,
            task_family="canonical",
            planner_group="canonical_bars_backlog",
            vendor=vendor,
            lease_owner=self.worker_id,
            payload={
                "dataset": task.dataset,
                "symbol": task.symbol,
                "start_date": task.start_date,
                "end_date": task.end_date,
                "kind": task.kind,
            },
        )
        if attempt is None:
            self.db.mark_task_failed(task.id, f"{vendor}: attempt unavailable", backoff_minutes=1)
            return []
        connector = self.connectors[vendor]
        try:
            frame = connector.fetch(task.dataset, symbols, task.start_date, task.end_date)
        except PermanentConnectorError as exc:
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=str(exc), backoff_minutes=0, permanent=True)
            self.db.mark_task_failed(task.id, f"{vendor}: {exc}", backoff_minutes=1)
            return []
        except ConnectorError as exc:
            message = str(exc)
            if "budget exhausted" in message:
                self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=message, backoff_minutes=30)
                self.db.defer_task(task.id, reason=f"{vendor}: {message}", backoff_minutes=5)
                return []
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=message, backoff_minutes=15)
            self.db.mark_task_failed(task.id, f"{vendor}: {message}", backoff_minutes=1)
            return []
        if frame.empty:
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error="empty backfill result", backoff_minutes=30)
            self.db.mark_task_failed(task.id, f"{vendor}: empty backfill result", backoff_minutes=1)
            return []
        changed = self._write_raw_partition(frame, source_name=vendor)
        self.db.mark_vendor_attempt_success(task_key=task_key, vendor=vendor, rows_returned=len(frame))
        self.db.mark_task_done(task.id)
        return changed

    def _process_parallel_datewide_backfill_task(self, *, task, coordinator_vendor: str) -> list[str]:
        """Fill a date-wide canonical partition by saturating all eligible vendors in parallel."""
        existing = self._read_existing_partition(date=task.start_date)
        existing_symbols = {
            str(symbol).upper()
            for symbol in existing.get("symbol", pd.Series(dtype="string")).dropna().astype("string").tolist()
        }
        remaining_symbols = [symbol for symbol in self.default_symbols if symbol.upper() not in existing_symbols]
        if not remaining_symbols:
            self.db.mark_task_done(task.id)
            return []

        capabilities = self._datewide_backfill_capabilities(dataset=task.dataset, preferred_vendor=coordinator_vendor)
        if not capabilities:
            self.db.defer_task(task.id, reason="no eligible date-wide backfill vendors", backoff_minutes=5)
            return []

        frames: list[pd.DataFrame] = []
        remaining: set[str] = {symbol.upper() for symbol in remaining_symbols}
        reserved: set[str] = set()
        attempted_by_symbol: dict[str, set[str]] = defaultdict(set)
        unavailable_vendors: set[str] = set()
        state_lock = threading.Lock()

        def claim_batch(vendor_name: str, batch_size: int) -> list[str]:
            with state_lock:
                available = [
                    symbol
                    for symbol in remaining_symbols
                    if symbol.upper() in remaining
                    and symbol.upper() not in reserved
                    and vendor_name not in attempted_by_symbol[symbol.upper()]
                ]
                batch = available[:batch_size]
                for symbol in batch:
                    symbol_key = symbol.upper()
                    reserved.add(symbol_key)
                    attempted_by_symbol[symbol_key].add(vendor_name)
                return batch

        def release_batch(symbols: list[str], *, completed: list[str] | None = None) -> None:
            completed_keys = {symbol.upper() for symbol in (completed or [])}
            with state_lock:
                for symbol in symbols:
                    symbol_key = symbol.upper()
                    reserved.discard(symbol_key)
                    if symbol_key in completed_keys:
                        remaining.discard(symbol_key)

        def worker(capability):
            connector = self.connectors[capability.vendor]
            batch_size = self._capability_batch_size(capability)
            vendor_frames: list[pd.DataFrame] = []
            while not self._stop_event.is_set():
                if capability.vendor in unavailable_vendors:
                    return vendor_frames
                batch = claim_batch(capability.vendor, batch_size)
                if not batch:
                    with state_lock:
                        unresolved_for_vendor = any(
                            symbol.upper() in remaining and capability.vendor not in attempted_by_symbol[symbol.upper()]
                            for symbol in remaining_symbols
                        )
                        if not unresolved_for_vendor and not reserved:
                            return vendor_frames
                    sleep(0.01)
                    continue
                try:
                    frame = connector.fetch(task.dataset, batch, task.start_date, task.end_date)
                except PermanentConnectorError:
                    unavailable_vendors.add(capability.vendor)
                    release_batch(batch, completed=[])
                    return vendor_frames
                except ConnectorError as exc:
                    if "budget exhausted" in str(exc):
                        unavailable_vendors.add(capability.vendor)
                    release_batch(batch, completed=[])
                    return vendor_frames
                if frame.empty:
                    release_batch(batch, completed=[])
                    continue
                covered = (
                    frame.get("symbol", pd.Series(dtype="string"))
                    .dropna()
                    .astype("string")
                    .str.upper()
                    .unique()
                    .tolist()
                )
                release_batch(batch, completed=covered)
                vendor_frames.append(frame)
            return vendor_frames

        futures: dict[object, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, sum(max(1, int(cap.lane_width or 1)) for cap in capabilities))) as executor:
            for capability in capabilities:
                for _ in range(max(1, int(capability.lane_width or 1))):
                    futures[executor.submit(worker, capability)] = capability.vendor
            for future in futures:
                for frame in future.result():
                    frames.append(frame)

        changed_dates: list[str] = []
        if frames:
            changed_dates = self._write_raw_partition(pd.concat(frames, ignore_index=True), source_name=self.source_name)

        merged = self._read_existing_partition(date=task.start_date)
        merged_symbols = {
            str(symbol).upper()
            for symbol in merged.get("symbol", pd.Series(dtype="string")).dropna().astype("string").tolist()
        }
        expected_symbols = {symbol.upper() for symbol in self.default_symbols}
        unresolved_count = len(expected_symbols.difference(merged_symbols))
        if unresolved_count == 0:
            self.db.mark_task_done(task.id)
        else:
            self.db.defer_task(task.id, reason=f"remaining_symbols={unresolved_count}", backoff_minutes=5)
        return changed_dates

    def _datewide_backfill_capabilities(self, *, dataset: str, preferred_vendor: str) -> list:
        """Return eligible capabilities for a date-wide canonical task."""
        capabilities = backfill_capabilities(
            dataset=dataset,
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        )
        ordered = sorted(
            capabilities,
            key=lambda capability: (
                0 if capability.vendor == preferred_vendor else 1,
                capability.priority,
                capability.vendor,
            ),
        )
        return ordered

    def _canonical_capability(self, vendor: str):
        """Return the canonical bar capability for a specific vendor."""
        for capability in backfill_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            if capability.vendor == vendor:
                return capability
        return None

    def _planner_task_vendor_eligible(self, task: PlannerTask, *, vendor: str, now_iso: str) -> bool:
        """Return whether a planner task can be attempted by the given vendor now."""
        capability = self._canonical_capability(vendor)
        if capability is None:
            return False
        if vendor not in task.eligible_vendors:
            return False
        if task.status == "LEASED" and task.lease_expires_at and task.lease_expires_at > now_iso and task.lease_owner != self.worker_id:
            return False
        if task.next_eligible_at and task.next_eligible_at > now_iso:
            return False
        if capability.batching_mode == "single_symbol" and len(task.symbols) != 1:
            return False
        for attempt in self.db.vendor_attempts_for_task(task.task_key):
            if attempt.vendor != vendor:
                continue
            if attempt.status in {"SUCCESS", "PERMANENT_FAILED"}:
                return False
            if attempt.status == "LEASED" and attempt.lease_expires_at and attempt.lease_expires_at > now_iso and attempt.lease_owner != self.worker_id:
                return False
            if attempt.status == "FAILED" and attempt.next_eligible_at and attempt.next_eligible_at > now_iso:
                return False
        return True

    @staticmethod
    def _capability_batch_size(capability) -> int:
        """Return the batch size to use for a capability inside date-wide backfill dispatch."""
        if capability.batching_mode == "multi_symbol":
            return 100
        return 1

    def _read_existing_partition(self, *, date: str) -> pd.DataFrame:
        """Read the existing canonical raw partition for a date if present."""
        path = self.paths.raw_equities / f"date={date}" / "data.parquet"
        if not path.exists():
            return pd.DataFrame()
        return self._read_partition_frame(path)

    def _merge_partition_frame(self, *, partition: Path, frame: pd.DataFrame) -> pd.DataFrame:
        """Merge new rows into an existing raw partition without losing already collected symbols."""
        existing_path = partition / "data.parquet"
        existing = self._read_partition_frame(existing_path, empty_columns=frame.columns) if existing_path.exists() else pd.DataFrame(columns=frame.columns)
        combined = pd.concat([existing, frame], ignore_index=True)
        if combined.empty:
            return combined
        if "vendor_ts" in combined.columns:
            combined["vendor_ts"] = pd.to_datetime(combined["vendor_ts"], errors="coerce", utc=True)
        if "ingested_at" in combined.columns:
            combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], errors="coerce", utc=True)
        priority = self._canonical_vendor_priority()
        combined["_vendor_priority"] = (
            combined.get("source_name", pd.Series(dtype="string"))
            .astype("string")
            .map(priority)
            .fillna(9999)
            .astype(int)
        )
        sort_columns = ["date", "symbol", "_vendor_priority"]
        ascending = [True, True, True]
        if "vendor_ts" in combined.columns:
            sort_columns.append("vendor_ts")
            ascending.append(False)
        if "ingested_at" in combined.columns:
            sort_columns.append("ingested_at")
            ascending.append(False)
        combined = combined.sort_values(sort_columns, ascending=ascending)
        combined = combined.drop_duplicates(subset=["date", "symbol"], keep="first")
        return combined.drop(columns=["_vendor_priority"], errors="ignore").reset_index(drop=True)

    def _read_partition_frame(self, path: Path, *, empty_columns: object | None = None) -> pd.DataFrame:
        """Read a parquet partition, quarantining unreadable files instead of crashing the worker."""
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            quarantine = path.with_name(f"{path.stem}.corrupt.{uuid.uuid4().hex}{path.suffix}")
            LOGGER.warning("quarantining_unreadable_partition path=%s quarantine=%s error=%s", path, quarantine, exc)
            with contextlib.suppress(OSError):
                os.replace(path, quarantine)
            if empty_columns is None:
                return pd.DataFrame()
            return pd.DataFrame(columns=list(empty_columns))

    def _canonical_vendor_priority(self) -> dict[str, int]:
        """Return the canonical vendor preference ordering for equities backfill/forward rows."""
        priorities: dict[str, int] = {}
        for capability in forward_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ) + backfill_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            priorities[capability.vendor] = min(priorities.get(capability.vendor, capability.priority), capability.priority)
        priorities.setdefault(self.source_name, 0)
        return priorities

    def _run_reference_job(self, job: dict[str, object]) -> dict[str, object]:
        vendor = str(job["source"])
        connector = self.connectors[vendor]
        budget_manager = getattr(connector, "budget_manager", None)
        if budget_manager is not None and not budget_manager.can_spend(vendor, task_kind="OTHER"):
            return {"outputs": [], "failures": [], "deferred": 1}
        task_key = self._aux_task_key(job)
        leased = self.db.lease_vendor_attempt(
            task_key=task_key,
            task_family="auxiliary",
            planner_group=str(job.get("planner_group", "reference_events_backlog")),
            vendor=vendor,
            lease_owner=self.worker_id,
            payload=job,
        )
        if leased is None:
            return {"outputs": [], "failures": [], "deferred": 0}
        try:
            frame = connector.fetch(
                str(job["dataset"]),
                list(job.get("symbols", [])),
                str(job["start_date"]),
                str(job["end_date"]),
            )
        except PermanentConnectorError as exc:
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=str(exc), backoff_minutes=0, permanent=True)
            return {"outputs": [], "failures": [f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{exc}"], "deferred": 0}
        except ConnectorError as exc:
            message = str(exc)
            if "budget exhausted" in message:
                self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=message, backoff_minutes=30)
                return {"outputs": [], "failures": [], "deferred": 1}
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=message, backoff_minutes=15)
            return {"outputs": [], "failures": [f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{message}"], "deferred": 0}
        output = self.paths.reference_root / f"{job['output_name']}.parquet"
        self._append_reference_frame(output, frame)
        self.db.mark_vendor_attempt_success(task_key=task_key, vendor=vendor, rows_returned=len(frame))
        return {"outputs": [str(output)], "failures": [], "deferred": 0}

    def _append_reference_frame(self, output: Path, frame: pd.DataFrame) -> None:
        lock_path = output.with_suffix(".lock")
        self._acquire_file_lock(lock_path)
        try:
            if output.exists():
                existing = pd.read_parquet(output)
                frame = pd.concat([existing, frame], ignore_index=True) if not frame.empty else existing
            if not frame.empty:
                frame = frame.drop_duplicates().reset_index(drop=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output)
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    @staticmethod
    def _aux_task_key(job: dict[str, object]) -> str:
        symbols = ",".join(list(job.get("symbols", []))) or "__GLOBAL__"
        return "::".join(
            [
                "aux",
                str(job.get("source", "")),
                str(job.get("dataset", "")),
                symbols,
                str(job.get("start_date", "")),
                str(job.get("end_date", "")),
                str(job.get("output_name", "")),
            ]
        )
