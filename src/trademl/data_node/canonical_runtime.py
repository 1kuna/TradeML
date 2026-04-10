"""Canonical bar and backfill runtime helpers."""

from __future__ import annotations

from collections import defaultdict
import contextlib
import hashlib
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable

import pandas as pd

from trademl.calendars.exchange import get_trading_days
from trademl.connectors.base import BaseConnector, ConnectorError, PermanentConnectorError
from trademl.data_node.capabilities import backfill_capabilities, forward_capabilities
from trademl.data_node.db import DataNodeDB, PlannerTask
from trademl.data_node.planner import canonical_task_key, choose_vendor_for_canonical_task
from trademl.data_node.provider_contracts import dataset_contract

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CanonicalFetchResult:
    """Normalized outcome for canonical connector fetches."""

    frame: pd.DataFrame
    error: str | None = None
    permanent: bool = False
    backoff_minutes: int = 0

    @property
    def ok(self) -> bool:
        """Return whether the fetch produced a usable frame."""
        return self.error is None

    @property
    def budget_exhausted(self) -> bool:
        """Return whether the fetch failed due to vendor budget exhaustion."""
        return self.error is not None and "budget exhausted" in self.error


class CanonicalRuntime:
    """Encapsulate canonical bar execution, progress, and raw partition I/O."""

    def __init__(
        self,
        *,
        db: DataNodeDB,
        connectors: dict[str, BaseConnector],
        paths: Any,
        source_name: str,
        capability_audit_state: dict[str, object],
        worker_id: str,
        default_symbols_getter: Callable[[], list[str]],
        stop_requested: Callable[[], bool],
        write_raw_partition_fn: Callable[[pd.DataFrame, str], list[str]],
    ) -> None:
        self.db = db
        self.connectors = connectors
        self.paths = paths
        self.source_name = source_name
        self.capability_audit_state = capability_audit_state
        self.worker_id = worker_id
        self._default_symbols_getter = default_symbols_getter
        self._stop_requested = stop_requested
        self._write_raw_partition_fn = write_raw_partition_fn
        self._logged_unreadable_paths: set[tuple[str, str]] = set()
        self._log_dedupe_lock = threading.Lock()
        self._vendor_symbol_floor_cache: dict[tuple[str, str], str | None] = {}
        self._tiingo_supported_ticker_cache: dict[str, tuple[str | None, str | None]] | None = None

    def _lease_canonical_batch(self, vendor: str) -> list[PlannerTask]:
        """Lease a vendor-compatible batch of canonical planner tasks."""
        capability = self._canonical_capability(vendor)
        if capability is None:
            return []
        batch_limit = self._capability_batch_size(capability)
        now_iso = datetime.now(tz=UTC).isoformat()
        progress_map = self.db.planner_task_progress_map()
        base_task: PlannerTask | None = None
        candidates: list[PlannerTask] = []
        scan_limit = max(256, batch_limit * 8)
        for page in range(16):
            page_candidates = self.db.fetch_planner_tasks(
                task_families=("canonical_bars", "canonical_repair"),
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                limit=scan_limit,
                offset=page * scan_limit,
            )
            if not page_candidates:
                break
            page_candidates.sort(
                key=lambda task: (
                    self._canonical_backlog_rank(task),
                    0 if task.status == "PARTIAL" else 1 if task.status == "FAILED" else 2 if task.status == "PENDING" else 3,
                    progress_map.get(task.task_key).remaining_units if progress_map.get(task.task_key) is not None else 10**9,
                    task.priority,
                    task.created_at,
                    task.task_key,
                )
            )
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

    @staticmethod
    def _canonical_backlog_rank(task: PlannerTask) -> int:
        """Rank canonical backlog classes so repair work preempts rolling drift."""
        backlog_class = str(task.payload.get("backlog_class", "")).strip().lower() if isinstance(task.payload, dict) else ""
        if backlog_class == "phase1_pinned" or task.planner_group == "phase1_pinned_canonical":
            return 0
        if backlog_class == "repair" or task.planner_group == "canonical_repair":
            return 1
        if backlog_class == "rolling" or task.planner_group == "rolling_canonical":
            return 2
        return 3

    def _process_canonical_planner_task(self, task: PlannerTask, exchange: str) -> list[str]:
        """Compatibility wrapper for a single planner task."""
        vendor = next(
            (
                name
                for name in task.eligible_vendors
                if self._canonical_capability(name) is not None and self._vendor_can_serve_canonical_task(vendor=name, task=task)
            ),
            None,
        )
        if vendor is None:
            self.db.mark_planner_task_failed(task.task_key, error="no eligible canonical vendors", backoff_minutes=5)
            return []
        return self._process_canonical_planner_batch(batch=[task], vendor=vendor, exchange=exchange)

    def _fetch_canonical_frame(
        self,
        *,
        vendor: str,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
        empty_error: str | None = None,
        empty_permanent: bool = False,
        empty_backoff_minutes: int = 15,
    ) -> CanonicalFetchResult:
        """Fetch canonical data and normalize vendor failure semantics."""
        connector = self.connectors[vendor]
        try:
            frame = connector.fetch(dataset, symbols, start_date, end_date)
        except PermanentConnectorError as exc:
            return CanonicalFetchResult(frame=pd.DataFrame(), error=str(exc), permanent=True, backoff_minutes=0)
        except ConnectorError as exc:
            message = str(exc)
            return CanonicalFetchResult(
                frame=pd.DataFrame(),
                error=message,
                permanent=False,
                backoff_minutes=30 if "budget exhausted" in message else 15,
            )
        if frame.empty and empty_error is not None:
            return CanonicalFetchResult(
                frame=frame,
                error=empty_error,
                permanent=empty_permanent,
                backoff_minutes=empty_backoff_minutes,
            )
        return CanonicalFetchResult(frame=frame)

    def _mark_planner_canonical_batch_failure(
        self,
        *,
        tasks: list[PlannerTask],
        vendor: str,
        fetch_result: CanonicalFetchResult,
    ) -> None:
        """Apply a shared planner-task failure transition for a canonical batch."""
        assert fetch_result.error is not None
        for task in tasks:
            self.db.mark_vendor_attempt_failed(
                task_key=task.task_key,
                vendor=vendor,
                error=fetch_result.error,
                backoff_minutes=fetch_result.backoff_minutes,
                permanent=fetch_result.permanent,
            )
            if fetch_result.permanent and not self._canonical_task_has_remaining_vendors(task, failed_vendor=vendor):
                self.db.mark_planner_task_failed(
                    task.task_key,
                    error=f"{vendor}: {fetch_result.error}",
                    backoff_minutes=15,
                    permanent=True,
                )
                continue
            self.db.mark_planner_task_partial(
                task.task_key,
                error=f"{vendor}: {fetch_result.error}",
                backoff_minutes=self._canonical_next_task_backoff_minutes(
                    task,
                    excluded_vendor=vendor,
                    default_minutes=fetch_result.backoff_minutes or 1,
                ),
            )

    def _process_canonical_planner_batch(self, *, batch: list[PlannerTask], vendor: str, exchange: str) -> list[str]:
        """Process a vendor-compatible batch of atomic canonical tasks."""
        if not batch:
            return []
        dataset = batch[0].dataset
        start_date = batch[0].start_date
        end_date = batch[0].end_date
        leased_tasks: list[PlannerTask] = []
        symbols: list[str] = []
        for task in batch:
            allow_success_retry = self._planner_task_vendor_success_reusable(task=task, vendor=vendor)
            attempt = self.db.lease_vendor_attempt(
                task_key=task.task_key,
                task_family=task.task_family,
                planner_group=task.planner_group,
                vendor=vendor,
                lease_owner=self.worker_id,
                payload={"symbols": list(task.symbols), "start_date": task.start_date, "end_date": task.end_date},
                allow_success_retry=allow_success_retry,
            )
            if attempt is None:
                self.db.mark_planner_task_partial(task.task_key, error=f"{vendor}: attempt unavailable", backoff_minutes=5)
                continue
            leased_tasks.append(task)
            symbols.extend(list(task.symbols))
        if not leased_tasks:
            return []

        fetch_result = self._fetch_canonical_frame(
            vendor=vendor,
            dataset=dataset,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            empty_error=None,
        )
        if not fetch_result.ok:
            self._mark_planner_canonical_batch_failure(tasks=leased_tasks, vendor=vendor, fetch_result=fetch_result)
            return []
        frame = fetch_result.frame

        changed_dates: list[str] = []
        if not frame.empty:
            changed_dates = self._write_raw_partition_fn(frame, vendor)
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
                empty_error = self._canonical_empty_result_error(vendor=vendor, dataset=task.dataset)
                self.db.mark_vendor_attempt_failed(
                    task_key=task.task_key,
                    vendor=vendor,
                    error=empty_error,
                    backoff_minutes=0,
                    permanent=True,
                )
                if self._canonical_task_has_remaining_vendors(task, failed_vendor=vendor):
                    self.db.mark_planner_task_partial(
                        task.task_key,
                        error=f"{vendor}: {empty_error}",
                        backoff_minutes=self._canonical_next_task_backoff_minutes(task, excluded_vendor=vendor),
                    )
                    continue
                self.db.mark_planner_task_failed(
                    task.task_key,
                    error=f"{vendor}: {empty_error}",
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
                    backoff_minutes=self._canonical_next_task_backoff_minutes(task, excluded_vendor=vendor, default_minutes=5),
                )
        return changed_dates

    def _canonical_task_progress(self, *, task: PlannerTask, exchange: str) -> dict[str, object]:
        """Compute symbol-date coverage for a canonical planner task from raw partitions."""
        trading_days = list(task.payload.get("trading_days", []))
        if not trading_days:
            trading_days = [day.isoformat() for day in get_trading_days(exchange, pd.Timestamp(task.start_date).date(), pd.Timestamp(task.end_date).date())]
        return self._canonical_progress_for_scope(dataset=task.dataset, symbols=list(task.symbols), trading_days=trading_days)

    def _canonical_progress_for_scope(
        self,
        *,
        dataset: str,
        symbols: list[str],
        trading_days: list[str],
    ) -> dict[str, object]:
        """Compute canonical symbol-date coverage for a given task scope from the durable ledger."""
        return self.db.fetch_canonical_progress(
            dataset=dataset,
            symbols=list(symbols),
            trading_days=list(trading_days),
        )

    def _canonical_batch_progress(self, *, tasks: list[PlannerTask], exchange: str) -> dict[str, dict[str, object]]:
        """Compute canonical progress for a homogeneous task batch from the durable ledger."""
        progress: dict[str, dict[str, object]] = {}
        for task in tasks:
            trading_days = list(task.payload.get("trading_days", []))
            if not trading_days:
                trading_days = [
                    day.isoformat()
                    for day in get_trading_days(exchange, pd.Timestamp(task.start_date).date(), pd.Timestamp(task.end_date).date())
                ]
            progress[task.task_key] = self._canonical_progress_for_scope(
                dataset=task.dataset,
                symbols=list(task.symbols),
                trading_days=trading_days,
            )
        return progress

    def _canonical_task_has_remaining_vendors(self, task: PlannerTask, *, failed_vendor: str) -> bool:
        """Return whether another eligible vendor could still complete this canonical task."""
        attempts = {attempt.vendor: attempt for attempt in self.db.vendor_attempts_for_task(task.task_key)}
        for candidate in task.eligible_vendors:
            if candidate == failed_vendor:
                continue
            if self._canonical_capability(candidate) is None:
                continue
            if not self._vendor_can_serve_canonical_task(vendor=candidate, task=task):
                continue
            attempt = attempts.get(candidate)
            if attempt is None:
                return True
            if attempt.status == "PERMANENT_FAILED":
                continue
            if attempt.status == "SUCCESS" and not self._planner_task_vendor_success_reusable(task=task, vendor=candidate):
                continue
            return True
        return False

    def _canonical_task_has_spendable_vendor(self, task: PlannerTask, *, excluded_vendor: str | None = None) -> bool:
        """Return whether another eligible vendor can plausibly run this task right now."""
        now_iso = datetime.now(tz=UTC).isoformat()
        attempts = {attempt.vendor: attempt for attempt in self.db.vendor_attempts_for_task(task.task_key)}
        for candidate in task.eligible_vendors:
            if candidate == excluded_vendor:
                continue
            if self._canonical_capability(candidate) is None:
                continue
            if not self._vendor_can_serve_canonical_task(vendor=candidate, task=task):
                continue
            if not self._vendor_has_local_budget(vendor=candidate, dataset=task.dataset, symbol_count=len(task.symbols)):
                continue
            attempt = attempts.get(candidate)
            if attempt is None:
                return True
            if attempt.status == "PERMANENT_FAILED":
                continue
            if attempt.status == "SUCCESS" and not self._planner_task_vendor_success_reusable(task=task, vendor=candidate):
                continue
            if attempt.status == "LEASED" and attempt.lease_expires_at and attempt.lease_expires_at > now_iso and attempt.lease_owner != self.worker_id:
                continue
            if attempt.status == "FAILED" and attempt.next_eligible_at and attempt.next_eligible_at > now_iso:
                continue
            return True
        return False

    def _canonical_next_task_backoff_minutes(
        self,
        task: PlannerTask,
        *,
        excluded_vendor: str | None = None,
        default_minutes: int = 1,
    ) -> int:
        """Return the soonest task-level backoff implied by remaining vendor eligibility."""
        now = datetime.now(tz=UTC)
        attempts = {attempt.vendor: attempt for attempt in self.db.vendor_attempts_for_task(task.task_key)}
        earliest_ready: datetime | None = None
        for candidate in task.eligible_vendors:
            if candidate == excluded_vendor:
                continue
            if self._canonical_capability(candidate) is None:
                continue
            if not self._vendor_can_serve_canonical_task(vendor=candidate, task=task):
                continue
            if not self._vendor_has_local_budget(vendor=candidate, dataset=task.dataset, symbol_count=len(task.symbols)):
                continue
            attempt = attempts.get(candidate)
            if attempt is None:
                earliest_ready = now
                break
            if attempt.status == "PERMANENT_FAILED":
                continue
            if attempt.status == "SUCCESS" and not self._planner_task_vendor_success_reusable(task=task, vendor=candidate):
                continue
            if attempt.status == "LEASED":
                if attempt.lease_expires_at:
                    expiry = datetime.fromisoformat(str(attempt.lease_expires_at))
                    earliest_ready = expiry if earliest_ready is None else min(earliest_ready, expiry)
                continue
            if attempt.status == "FAILED" and attempt.next_eligible_at:
                retry_at = datetime.fromisoformat(str(attempt.next_eligible_at))
                earliest_ready = retry_at if earliest_ready is None else min(earliest_ready, retry_at)
                continue
            earliest_ready = now if earliest_ready is None else min(earliest_ready, now)
        if earliest_ready is None:
            return max(0, int(default_minutes))
        delay_seconds = max(0.0, (earliest_ready - now).total_seconds())
        return int((delay_seconds + 59) // 60)

    def _vendor_can_serve_canonical_task(self, *, vendor: str, task: PlannerTask) -> bool:
        """Return whether attempt history implies a vendor can plausibly serve this task."""
        contract = dataset_contract(vendor, task.dataset)
        if contract is not None and task.priority <= 5 and not contract.critical_path_allowed:
            return False
        if contract is not None and contract.max_history_years is not None:
            current = pd.Timestamp(datetime.now(tz=UTC).date())
            earliest_supported = (current - pd.DateOffset(years=int(contract.max_history_years))).normalize()
            if pd.Timestamp(task.start_date) < earliest_supported:
                return False
        if len(task.symbols) != 1:
            return True
        if vendor == "tiingo" and not self._tiingo_supported_ticker_allows(task.symbols[0], start_date=str(task.start_date), end_date=str(task.end_date)):
            return False
        floor = self._vendor_symbol_floor(vendor=vendor, symbol=task.symbols[0])
        if not floor:
            return True
        return str(task.end_date) >= floor

    def _tiingo_supported_ticker_allows(self, symbol: str, *, start_date: str, end_date: str) -> bool:
        """Return whether Tiingo metadata says a symbol can cover the requested window."""
        metadata = self._tiingo_supported_tickers().get(str(symbol).upper())
        if metadata is None:
            return True
        start_floor, end_ceiling = metadata
        if start_floor and pd.Timestamp(end_date) < pd.Timestamp(start_floor):
            return False
        if end_ceiling and pd.Timestamp(start_date) > pd.Timestamp(end_ceiling):
            return False
        return True

    def _tiingo_supported_tickers(self) -> dict[str, tuple[str | None, str | None]]:
        """Load Tiingo supported ticker metadata keyed by symbol."""
        if self._tiingo_supported_ticker_cache is not None:
            return self._tiingo_supported_ticker_cache
        path = self.paths.reference_root / "tiingo_supported_tickers.parquet"
        if not path.exists():
            self._tiingo_supported_ticker_cache = {}
            return self._tiingo_supported_ticker_cache
        try:
            frame = pd.read_parquet(path, columns=["symbol", "start_date", "end_date"])
        except Exception as exc:  # pragma: no cover
            self._log_unreadable_partition_once(event="skipping_unreadable_reference_partition", path=path, error=exc)
            self._tiingo_supported_ticker_cache = {}
            return self._tiingo_supported_ticker_cache
        metadata: dict[str, tuple[str | None, str | None]] = {}
        if not frame.empty:
            normalized = frame.copy()
            normalized["symbol"] = normalized["symbol"].astype("string").str.upper()
            normalized["start_date"] = pd.to_datetime(normalized.get("start_date"), errors="coerce").dt.strftime("%Y-%m-%d")
            normalized["end_date"] = pd.to_datetime(normalized.get("end_date"), errors="coerce").dt.strftime("%Y-%m-%d")
            for row in normalized.dropna(subset=["symbol"]).itertuples(index=False):
                metadata[str(row.symbol)] = (
                    str(row.start_date) if pd.notna(row.start_date) else None,
                    str(row.end_date) if pd.notna(row.end_date) else None,
                )
        self._tiingo_supported_ticker_cache = metadata
        return self._tiingo_supported_ticker_cache

    def _vendor_symbol_floor(self, *, vendor: str, symbol: str) -> str | None:
        """Infer the earliest non-empty date range a vendor has shown for a symbol."""
        key = (vendor, str(symbol).upper())
        if key in self._vendor_symbol_floor_cache:
            return self._vendor_symbol_floor_cache[key]
        attempts = self.db.vendor_attempts_for_symbol(vendor=vendor, symbol=symbol, task_family="canonical_bars")
        success_starts: list[str] = []
        empty_fail_ends: list[str] = []
        for attempt in attempts:
            payload = attempt.payload
            start_date = str(payload.get("start_date") or "")
            end_date = str(payload.get("end_date") or "")
            if attempt.status == "SUCCESS" and start_date:
                success_starts.append(start_date)
            if attempt.status == "PERMANENT_FAILED" and attempt.last_error in {
                "empty planner canonical result",
                "valid empty planner canonical result",
            } and end_date:
                empty_fail_ends.append(end_date)
        boundary: str | None = None
        if success_starts and empty_fail_ends:
            earliest_success = min(success_starts)
            if any(failed_end < earliest_success for failed_end in empty_fail_ends):
                boundary = earliest_success
        if boundary is not None:
            self._vendor_symbol_floor_cache[key] = boundary
        return boundary

    def _build_canonical_coverage_index(self, *, trading_days: list[str] | None = None) -> dict[str, set[str]]:
        """Return the existing canonical coverage index keyed by symbol."""
        coverage: dict[str, set[str]] = defaultdict(set)
        if not self.paths.raw_equities.exists():
            return coverage
        days = trading_days or [path.name.partition("=")[2] for path in sorted(self.paths.raw_equities.glob("date=*")) if "=" in path.name]
        for day in days:
            path = self.paths.raw_equities / f"date={day}" / "data.parquet"
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path, columns=["symbol"])
            except Exception as exc:  # pragma: no cover
                self._log_unreadable_partition_once(event="skipping_unreadable_raw_partition", path=path, error=exc)
                continue
            if frame.empty or "symbol" not in frame.columns:
                continue
            symbols = frame["symbol"].dropna().astype("string").str.upper().drop_duplicates().tolist()
            for symbol in symbols:
                coverage[symbol].add(day)
        return coverage

    def _write_raw_shard_partition(self, frame: pd.DataFrame, *, shard_id: str, source_name: str) -> list[str]:
        """Write immutable shard rows, then compact into the canonical raw partition."""
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
        """Compact immutable shard files into the canonical raw partition and ledger."""
        partition = self.paths.raw_equities / f"date={day_value}"
        shard_root = partition / "shards"
        lock_path = partition / ".merge.lock"
        self._acquire_file_lock(lock_path)
        try:
            frames = [pd.read_parquet(path) for path in sorted(shard_root.glob("*.parquet"))]
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            merged = self._merge_partition_frame(partition=partition, frame=merged) if not merged.empty else pd.DataFrame()
            tmp_path = partition / f"data.{uuid.uuid4().hex}.tmp"
            merged.to_parquet(tmp_path, index=False)
            output = partition / "data.parquet"
            os.replace(tmp_path, output)
            symbols = (
                merged.get("symbol", pd.Series(dtype="string")).dropna().astype("string").str.upper().drop_duplicates().tolist()
                if not merged.empty
                else []
            )
            source_names = {}
            if not merged.empty and "symbol" in merged.columns and "source_name" in merged.columns:
                latest_sources = (
                    merged.assign(symbol=merged["symbol"].astype("string").str.upper())
                    .dropna(subset=["symbol"])
                    .drop_duplicates(subset=["symbol"], keep="first")
                )
                source_names = {
                    str(row["symbol"]).upper(): str(row["source_name"])
                    for row in latest_sources[["symbol", "source_name"]].to_dict("records")
                }
            current_manifest = self.db.get_raw_partition_manifest(dataset="equities_eod", trading_date=day_value)
            partition_revision = int(current_manifest.partition_revision + 1) if current_manifest is not None else 1
            content_hash = self._partition_content_hash(symbols=symbols, row_count=len(merged))
            self.db.replace_canonical_units_for_date(
                dataset="equities_eod",
                trading_date=day_value,
                symbols=symbols,
                partition_revision=partition_revision,
                source_names=source_names,
            )
            self.db.upsert_raw_partition_manifest(
                dataset="equities_eod",
                trading_date=day_value,
                partition_revision=partition_revision,
                symbol_count=len(symbols),
                row_count=len(merged),
                symbols=symbols,
                content_hash=content_hash,
                status="HEALTHY",
            )
            return output
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    @staticmethod
    def _partition_content_hash(*, symbols: list[str], row_count: int) -> str:
        """Return a stable manifest hash for a compacted partition."""
        payload = f"{row_count}|{'|'.join(sorted({str(symbol).upper() for symbol in symbols}))}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _acquire_file_lock(path: Path, *, stale_after_seconds: int = 15) -> None:
        """Acquire a simple filesystem lock file with stale-lock recovery."""
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

    def _collect_cluster_shard(self, *, trading_date: str, shard) -> None:
        """Collect a cluster shard if it is not already present."""
        partition = self.paths.raw_equities / f"date={trading_date}" / "shards" / f"{shard.shard_id}.parquet"
        if partition.exists():
            return
        # the service shell supplies the actual shard fetch path
        raise NotImplementedError

    def _backfill_lane_widths(self) -> dict[str, int]:
        """Return the lane widths to use for canonical backfill vendors."""
        widths: dict[str, int] = {}
        order_keys: dict[str, tuple[int, int, str]] = {}
        for capability in backfill_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            lane_state = self._vendor_lane_state(capability.vendor, dataset=capability.dataset)
            if lane_state in {"COOLDOWN", "DISABLED"}:
                continue
            lane_width = int(capability.lane_width or 1)
            if lane_state == "DEGRADED":
                lane_width = 1
            widths[capability.vendor] = max(widths.get(capability.vendor, 0), lane_width)
            order_keys[capability.vendor] = min(
                order_keys.get(capability.vendor, self._canonical_lane_order_key(capability.vendor, capability.priority)),
                self._canonical_lane_order_key(capability.vendor, capability.priority),
            )
        return {vendor: widths[vendor] for vendor in sorted(widths, key=lambda vendor: order_keys[vendor])}

    @staticmethod
    def _canonical_lane_order_key(vendor: str, capability_priority: int) -> tuple[int, int, str]:
        """Return the scheduler ordering for canonical vendor lanes."""
        preferred = {
            "alpaca": 0,
            "tiingo": 1,
            "twelve_data": 2,
            "massive": 3,
            "finnhub": 4,
        }
        return (preferred.get(vendor, 99), int(capability_priority), str(vendor))

    def _lease_next_task_for_vendor(self, vendor: str):
        """Lease the next legacy backfill task best served by a given vendor."""
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
        symbols = [task.symbol] if task.symbol else self._default_symbols_getter()
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
        fetch_result = self._fetch_canonical_frame(
            vendor=vendor,
            dataset=task.dataset,
            symbols=symbols,
            start_date=task.start_date,
            end_date=task.end_date,
            empty_error="empty backfill result",
            empty_permanent=False,
            empty_backoff_minutes=30,
        )
        if not fetch_result.ok:
            assert fetch_result.error is not None
            self.db.mark_vendor_attempt_failed(
                task_key=task_key,
                vendor=vendor,
                error=fetch_result.error,
                backoff_minutes=fetch_result.backoff_minutes,
                permanent=fetch_result.permanent,
            )
            if fetch_result.budget_exhausted:
                self.db.defer_task(task.id, reason=f"{vendor}: {fetch_result.error}", backoff_minutes=5)
                return []
            self.db.mark_task_failed(task.id, f"{vendor}: {fetch_result.error}", backoff_minutes=1)
            return []
        changed = self._write_raw_partition_fn(fetch_result.frame, vendor)
        self.db.mark_vendor_attempt_success(task_key=task_key, vendor=vendor, rows_returned=len(fetch_result.frame))
        self.db.mark_task_done(task.id)
        return changed

    def _process_parallel_datewide_backfill_task(self, *, task, coordinator_vendor: str) -> list[str]:
        """Fill a date-wide canonical partition by saturating all eligible vendors in parallel."""
        existing = self._read_existing_partition(date=task.start_date)
        existing_symbols = {
            str(symbol).upper()
            for symbol in existing.get("symbol", pd.Series(dtype="string")).dropna().astype("string").tolist()
        }
        default_symbols = self._default_symbols_getter()
        remaining_symbols = [symbol for symbol in default_symbols if symbol.upper() not in existing_symbols]
        if not remaining_symbols:
            self.db.mark_task_done(task.id)
            return []

        capabilities = self._datewide_backfill_capabilities(dataset=task.dataset, preferred_vendor=coordinator_vendor)
        if not capabilities:
            self.db.defer_task(task.id, reason="no eligible date-wide backfill vendors", backoff_minutes=5)
            return []

        from concurrent.futures import ThreadPoolExecutor

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
            batch_size = self._capability_batch_size(capability)
            vendor_frames: list[pd.DataFrame] = []
            while not self._stop_requested():
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
                fetch_result = self._fetch_canonical_frame(
                    vendor=capability.vendor,
                    dataset=task.dataset,
                    symbols=batch,
                    start_date=task.start_date,
                    end_date=task.end_date,
                    empty_error=None,
                )
                if not fetch_result.ok:
                    if fetch_result.permanent or fetch_result.budget_exhausted:
                        unavailable_vendors.add(capability.vendor)
                    release_batch(batch, completed=[])
                    return vendor_frames
                frame = fetch_result.frame
                if frame.empty:
                    release_batch(batch, completed=[])
                    continue
                covered = frame.get("symbol", pd.Series(dtype="string")).dropna().astype("string").str.upper().unique().tolist()
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
            changed_dates = self._write_raw_partition_fn(pd.concat(frames, ignore_index=True), self.source_name)

        merged = self._read_existing_partition(date=task.start_date)
        merged_symbols = {
            str(symbol).upper()
            for symbol in merged.get("symbol", pd.Series(dtype="string")).dropna().astype("string").tolist()
        }
        expected_symbols = {symbol.upper() for symbol in default_symbols}
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
        if self._vendor_lane_state(vendor, dataset=task.dataset) in {"COOLDOWN", "DISABLED"}:
            return False
        if vendor not in task.eligible_vendors:
            return False
        if task.status == "LEASED" and task.lease_expires_at and task.lease_expires_at > now_iso and task.lease_owner != self.worker_id:
            return False
        if task.next_eligible_at and task.next_eligible_at > now_iso:
            return False
        if capability.batching_mode == "single_symbol" and len(task.symbols) != 1:
            return False
        if not self._vendor_has_local_budget(vendor=vendor, dataset=task.dataset, symbol_count=len(task.symbols)):
            return False
        for attempt in self.db.vendor_attempts_for_task(task.task_key):
            if attempt.vendor != vendor:
                continue
            if attempt.status == "PERMANENT_FAILED":
                return False
            if attempt.status == "SUCCESS" and not self._planner_task_vendor_success_reusable(task=task, vendor=vendor):
                return False
            if attempt.status == "LEASED" and attempt.lease_expires_at and attempt.lease_expires_at > now_iso and attempt.lease_owner != self.worker_id:
                return False
            if attempt.status == "FAILED" and attempt.next_eligible_at and attempt.next_eligible_at > now_iso:
                return False
        return True

    def _planner_task_vendor_success_reusable(self, *, task: PlannerTask, vendor: str) -> bool:
        """Return whether an incomplete planner task may retry a vendor that previously succeeded."""
        if task.status not in {"PARTIAL", "FAILED", "LEASED"}:
            return False
        if vendor not in task.eligible_vendors:
            return False
        progress = self.db.fetch_planner_task_progress(task.task_key)
        return bool(progress is not None and progress.remaining_units > 0)

    @staticmethod
    def _capability_batch_size(capability) -> int:
        """Return the batch size to use for a capability inside date-wide backfill dispatch."""
        contract = dataset_contract(capability.vendor, capability.dataset)
        if contract is not None and int(contract.max_batch_symbols or 0) > 0:
            return int(contract.max_batch_symbols)
        if int(getattr(capability, "preferred_batch_size", 0) or 0) > 0:
            return int(capability.preferred_batch_size)
        if capability.batching_mode == "multi_symbol":
            return 100
        return 1

    def _vendor_has_local_budget(self, *, vendor: str, dataset: str, symbol_count: int) -> bool:
        """Return whether a canonical vendor can spend for the current request shape."""
        connector = self.connectors.get(vendor)
        budget_manager = getattr(connector, "budget_manager", None)
        if budget_manager is None:
            return True
        if self._vendor_lane_state(vendor, dataset=dataset) in {"COOLDOWN", "DISABLED"}:
            return False
        contract = dataset_contract(vendor, dataset)
        request_units = max(1, int(getattr(contract, "request_cost_units", 1) or 1))
        if contract is not None and str(contract.request_cost_basis) == "symbol":
            request_units *= max(1, int(symbol_count))
        return bool(budget_manager.can_spend(vendor, task_kind="FORWARD", units=request_units))

    def _vendor_temporarily_throttled(self, vendor: str) -> bool:
        """Return whether a vendor should be held out due to recent pure-throttle behavior."""
        return self._vendor_lane_state(vendor, dataset="equities_eod") in {"COOLDOWN", "DISABLED"}

    def _vendor_lane_state(self, vendor: str, *, dataset: str) -> str:
        """Return and refresh the scheduler lane-health state for a vendor."""
        now = datetime.now(tz=UTC)
        connector = self.connectors.get(vendor)
        budget_manager = getattr(connector, "budget_manager", None)
        if budget_manager is None:
            self.db.upsert_vendor_lane_health(vendor=vendor, dataset=dataset, state="HEALTHY")
            return "HEALTHY"
        snapshot = budget_manager.snapshot(now=now)
        vendor_payload = (snapshot.get("vendors") or {}).get(vendor, {})
        telemetry = vendor_payload.get("telemetry") or {}
        window_counts = telemetry.get("window_counts") or {}
        outbound = int(window_counts.get("outbound_requests", 0) or 0)
        remote_429s = int(window_counts.get("remote_rate_limits", 0) or 0)
        local_blocks = int(window_counts.get("local_budget_blocks", 0) or 0)
        empty_valid = int(window_counts.get("empty_successes", 0) or 0)
        permanent_failures = int(window_counts.get("permanent_failures", 0) or 0)
        state = "HEALTHY"
        cooldown_until: str | None = None
        if permanent_failures >= 5 and outbound == 0:
            state = "DISABLED"
        elif remote_429s >= 3 and remote_429s >= max(3, outbound):
            state = "COOLDOWN"
            cooldown_until = (now + pd.Timedelta(minutes=5)).isoformat()
        elif local_blocks >= 3 and outbound == 0:
            state = "DEGRADED"
        self.db.upsert_vendor_lane_health(
            vendor=vendor,
            dataset=dataset,
            state=state,
            cooldown_until=cooldown_until,
            recent_outbound_requests=outbound,
            recent_success_units=int(telemetry.get("totals", {}).get("logical_units", 0) or 0),
            recent_remote_429s=remote_429s,
            recent_local_budget_blocks=local_blocks,
            recent_empty_valid=empty_valid,
            recent_permanent_failures=permanent_failures,
        )
        lane_health = self.db.get_vendor_lane_health(vendor=vendor, dataset=dataset)
        if lane_health is None:
            return state
        if lane_health.state == "COOLDOWN" and lane_health.cooldown_until and lane_health.cooldown_until > now.isoformat():
            return "COOLDOWN"
        return str(lane_health.state)

    @staticmethod
    def _canonical_empty_result_error(*, vendor: str, dataset: str) -> str:
        """Return the docs-backed empty-result error label for a canonical lane."""
        contract = dataset_contract(vendor, dataset)
        if contract is not None and contract.empty_result_policy == "valid_empty":
            return "valid empty planner canonical result"
        return "empty planner canonical result"

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
        combined["_vendor_priority"] = combined.get("source_name", pd.Series(dtype="string")).astype("string").map(priority).fillna(9999).astype(int)
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
            self._log_unreadable_partition_once(
                event="quarantining_unreadable_partition",
                path=path,
                error=exc,
                quarantine=quarantine,
            )
            with contextlib.suppress(OSError):
                os.replace(path, quarantine)
            if empty_columns is None:
                return pd.DataFrame()
            return pd.DataFrame(columns=list(empty_columns))

    def _log_unreadable_partition_once(
        self,
        *,
        event: str,
        path: Path,
        error: Exception,
        quarantine: Path | None = None,
    ) -> None:
        """Warn once per unreadable path/event pair, then downgrade repeats to debug."""
        key = (event, str(path))
        with self._log_dedupe_lock:
            first_seen = key not in self._logged_unreadable_paths
            if first_seen:
                self._logged_unreadable_paths.add(key)
        if first_seen:
            if quarantine is None:
                LOGGER.warning("%s path=%s error=%s", event, path, error)
            else:
                LOGGER.warning("%s path=%s quarantine=%s error=%s", event, path, quarantine, error)
            return
        if quarantine is None:
            LOGGER.debug("%s_repeat path=%s error=%s", event, path, error)
        else:
            LOGGER.debug("%s_repeat path=%s quarantine=%s error=%s", event, path, quarantine, error)

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
