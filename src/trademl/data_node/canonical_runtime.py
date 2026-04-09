"""Canonical bar and backfill runtime helpers."""

from __future__ import annotations

from collections import defaultdict
import contextlib
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
            if not self._vendor_can_serve_canonical_task(vendor=candidate, task=task):
                continue
            attempt = attempts.get(candidate)
            if attempt is None:
                return True
            if attempt.status in {"SUCCESS", "PERMANENT_FAILED"}:
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
            attempt = attempts.get(candidate)
            if attempt is None:
                earliest_ready = now
                break
            if attempt.status in {"SUCCESS", "PERMANENT_FAILED"}:
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
        """Write shard-specific raw partitions and merge them into the date partition."""
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
        """Merge shard parquet files into the canonical raw partition for a date."""
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
        for capability in backfill_capabilities(
            dataset="equities_eod",
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            widths[capability.vendor] = max(widths.get(capability.vendor, 0), int(capability.lane_width or 1))
        return widths

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
        contract = dataset_contract(capability.vendor, capability.dataset)
        if contract is not None and int(contract.max_batch_symbols or 0) > 0:
            return int(contract.max_batch_symbols)
        if int(getattr(capability, "preferred_batch_size", 0) or 0) > 0:
            return int(capability.preferred_batch_size)
        if capability.batching_mode == "multi_symbol":
            return 100
        return 1

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
