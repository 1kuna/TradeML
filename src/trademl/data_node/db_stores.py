"""Focused stores behind the data-node SQLite facade."""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any, Callable

from trademl.data_node.db_models import (
    CanonicalUnit,
    PlannerTask,
    PlannerTaskProgress,
    RawPartitionManifest,
    VendorAttempt,
    VendorLaneHealth,
    VendorTaskClaimResult,
)


class _Store:
    """Base class for focused SQLite stores."""

    def __init__(
        self,
        connect: Callable[[], AbstractContextManager[sqlite3.Connection]],
        clock: Callable[[], datetime],
    ) -> None:
        self._connect = connect
        self._clock = clock

    def _now(self) -> datetime:
        return self._clock()


class RuntimeStore(_Store):
    """Runtime partition status and compatibility markers."""

    def update_partition_status(
        self,
        source: str,
        dataset: str,
        date: str,
        status: str,
        row_count: int | None,
        expected_rows: int | None = None,
        qc_code: str | None = None,
        note: str | None = None,
    ) -> None:
        """Upsert the partition QC ledger."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO partition_status (
                  source, dataset, date, status, row_count, expected_rows, qc_code, note, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, dataset, date)
                DO UPDATE SET
                  status = excluded.status,
                  row_count = excluded.row_count,
                  expected_rows = excluded.expected_rows,
                  qc_code = excluded.qc_code,
                  note = excluded.note,
                  updated_at = excluded.updated_at
                """,
                (
                    source,
                    dataset,
                    date,
                    status,
                    row_count,
                    expected_rows,
                    qc_code,
                    note,
                    self._now().isoformat(),
                ),
            )

    def fetch_partition_status(self) -> list[sqlite3.Row]:
        """Return the mirrored partition ledger for testing and sync."""
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM partition_status ORDER BY source, dataset, date"
            ).fetchall()

    def mark_legacy_datewide_backfill_migrated(self) -> int:
        """Retire legacy backlog rows after planner migration is active."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE backfill_queue
                SET status = 'DONE',
                    last_error = 'migrated_to_planner',
                    updated_at = ?
                WHERE status IN ('PENDING', 'FAILED', 'LEASED')
                """,
                (self._now().isoformat(),),
            )
        return int(cursor.rowcount)

    def record_scheduler_decision(
        self,
        *,
        vendor: str,
        decision: str,
        dataset: str | None = None,
        task_key: str | None = None,
        reason: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Persist one scheduler claim decision for observability rollups."""
        timestamp = created_at or self._now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scheduler_decisions (
                  vendor, dataset, decision, task_key, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(vendor),
                    str(dataset) if dataset is not None else None,
                    str(decision),
                    str(task_key) if task_key is not None else None,
                    str(reason) if reason is not None else None,
                    timestamp,
                ),
            )

    def summarize_scheduler_decisions(
        self, *, minutes: int = 15, limit: int | None = None
    ) -> dict[str, Any]:
        """Return recent scheduler decisions grouped by vendor/dataset/decision."""
        window_minutes = max(1, int(minutes))
        since = (self._now() - timedelta(minutes=window_minutes)).isoformat()
        params: list[object] = [since]
        query = """
            SELECT vendor,
                   COALESCE(dataset, '') AS dataset,
                   decision,
                   COUNT(*) AS count,
                   MAX(created_at) AS latest_at,
                   MAX(reason) AS latest_reason
            FROM scheduler_decisions
            WHERE created_at >= ?
            GROUP BY vendor, COALESCE(dataset, ''), decision
            ORDER BY count DESC, vendor ASC, dataset ASC, decision ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {
            "window_minutes": window_minutes,
            "checked_at": self._now().isoformat(),
            "rows": [dict(row) for row in rows],
        }

    def prune_scheduler_decisions(self, *, retention_hours: int = 24) -> int:
        """Delete old scheduler decision telemetry."""
        cutoff = (self._now() - timedelta(hours=max(1, int(retention_hours)))).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM scheduler_decisions WHERE created_at < ?",
                (cutoff,),
            )
        return int(cursor.rowcount)

    def record_archive_write_telemetry(
        self,
        *,
        output_name: str,
        status: str,
        partition_date: str | None = None,
        rows_in: int = 0,
        rows_written: int = 0,
        duplicates_dropped: int = 0,
        coerced_columns: list[str] | tuple[str, ...] | None = None,
        schema_mismatch: bool = False,
        error: str | None = None,
        duration_ms: float | None = None,
        created_at: str | None = None,
    ) -> None:
        """Persist one raw/archive parquet write telemetry event."""
        timestamp = created_at or self._now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO archive_write_telemetry (
                  output_name, partition_date, status, rows_in, rows_written,
                  duplicates_dropped, coerced_columns_json, schema_mismatch,
                  error, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(output_name),
                    str(partition_date) if partition_date is not None else None,
                    str(status),
                    int(rows_in),
                    int(rows_written),
                    int(duplicates_dropped),
                    json.dumps(sorted({str(value) for value in (coerced_columns or [])})),
                    1 if schema_mismatch else 0,
                    str(error) if error is not None else None,
                    float(duration_ms) if duration_ms is not None else None,
                    timestamp,
                ),
            )

    def summarize_archive_write_telemetry(
        self, *, minutes: int = 60, limit: int | None = None
    ) -> dict[str, Any]:
        """Return recent archive write health grouped by output dataset."""
        window_minutes = max(1, int(minutes))
        since = (self._now() - timedelta(minutes=window_minutes)).isoformat()
        params: list[object] = [since]
        query = """
            SELECT output_name,
                   COUNT(*) AS writes,
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
                   SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failures,
                   SUM(rows_in) AS rows_in,
                   SUM(rows_written) AS rows_written,
                   SUM(duplicates_dropped) AS duplicates_dropped,
                   SUM(schema_mismatch) AS schema_mismatches,
                   MAX(created_at) AS latest_at,
                   MAX(error) AS latest_error
            FROM archive_write_telemetry
            WHERE created_at >= ?
            GROUP BY output_name
            ORDER BY failures DESC, writes DESC, output_name ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {
            "window_minutes": window_minutes,
            "checked_at": self._now().isoformat(),
            "rows": [dict(row) for row in rows],
        }

    def prune_archive_write_telemetry(self, *, retention_hours: int = 24) -> int:
        """Delete old archive write telemetry."""
        cutoff = (self._now() - timedelta(hours=max(1, int(retention_hours)))).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM archive_write_telemetry WHERE created_at < ?",
                (cutoff,),
            )
        return int(cursor.rowcount)

    def record_ingestion_ledger(
        self,
        *,
        dataset: str,
        output_name: str,
        status: str,
        vendor: str | None = None,
        partition_date: str | None = None,
        task_key: str | None = None,
        attempt_id: str | None = None,
        rows_in: int = 0,
        rows_normalized: int = 0,
        rows_written: int = 0,
        duplicates_dropped: int = 0,
        schema_version: str | None = None,
        payload_hash: str | None = None,
        partition_path: str | None = None,
        feature_visibility: dict[str, Any] | None = None,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Persist one ingestion event from collection through archive visibility."""
        timestamp = created_at or self._now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_ledger (
                  vendor, dataset, output_name, partition_date, task_key, attempt_id,
                  status, rows_in, rows_normalized, rows_written, duplicates_dropped,
                  schema_version, payload_hash, partition_path, feature_visibility_json,
                  error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(vendor) if vendor is not None else None,
                    str(dataset),
                    str(output_name),
                    str(partition_date) if partition_date is not None else None,
                    str(task_key) if task_key is not None else None,
                    str(attempt_id) if attempt_id is not None else None,
                    str(status),
                    int(rows_in),
                    int(rows_normalized),
                    int(rows_written),
                    int(duplicates_dropped),
                    str(schema_version) if schema_version is not None else None,
                    str(payload_hash) if payload_hash is not None else None,
                    str(partition_path) if partition_path is not None else None,
                    json.dumps(feature_visibility or {}, sort_keys=True),
                    str(error) if error is not None else None,
                    timestamp,
                ),
            )

    def summarize_ingestion_ledger(
        self, *, minutes: int = 60, limit: int | None = None
    ) -> dict[str, Any]:
        """Return recent ingestion events grouped by dataset/output/status."""
        window_minutes = max(1, int(minutes))
        since = (self._now() - timedelta(minutes=window_minutes)).isoformat()
        params: list[object] = [since]
        query = """
            SELECT COALESCE(vendor, '') AS vendor,
                   dataset,
                   output_name,
                   status,
                   COUNT(*) AS events,
                   SUM(rows_in) AS rows_in,
                   SUM(rows_normalized) AS rows_normalized,
                   SUM(rows_written) AS rows_written,
                   SUM(duplicates_dropped) AS duplicates_dropped,
                   MAX(partition_date) AS latest_partition_date,
                   MAX(partition_path) AS latest_partition_path,
                   MAX(payload_hash) AS latest_payload_hash,
                   MAX(error) AS latest_error,
                   MAX(created_at) AS latest_at
            FROM ingestion_ledger
            WHERE created_at >= ?
            GROUP BY COALESCE(vendor, ''), dataset, output_name, status
            ORDER BY latest_at DESC, events DESC, dataset ASC, output_name ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {
            "window_minutes": window_minutes,
            "checked_at": self._now().isoformat(),
            "rows": [dict(row) for row in rows],
        }

    def prune_ingestion_ledger(self, *, retention_hours: int = 168) -> int:
        """Delete old ingestion ledger rows after shared rollups have aged out."""
        cutoff = (self._now() - timedelta(hours=max(1, int(retention_hours)))).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ingestion_ledger WHERE created_at < ?",
                (cutoff,),
            )
        return int(cursor.rowcount)




class CanonicalLedgerStore(_Store):
    """Canonical symbol-date and raw manifest ledger access."""

    def replace_canonical_units_for_date(
        self,
        *,
        dataset: str,
        trading_date: str,
        symbols: list[str] | tuple[str, ...],
        partition_revision: int,
        source_names: dict[str, str] | None = None,
        task_key: str | None = None,
    ) -> None:
        """Replace the durable symbol-date ledger for a single compacted partition."""
        timestamp = self._now().isoformat()
        normalized = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
        with self._connect() as connection:
            existing_rows = connection.execute(
                """
                SELECT symbol
                FROM canonical_units
                WHERE dataset = ? AND trading_date = ?
                """,
                (dataset, trading_date),
            ).fetchall()
            existing = {str(row["symbol"]).upper() for row in existing_rows}
            missing = sorted(existing.difference(normalized))
            if missing:
                placeholders = ",".join("?" for _ in missing)
                connection.execute(
                    f"""
                    UPDATE canonical_units
                    SET status = 'MISSING',
                        written_at = ?,
                        partition_revision = ?,
                        last_error = 'missing from compacted partition'
                    WHERE dataset = ?
                      AND trading_date = ?
                      AND symbol IN ({placeholders})
                    """,
                    [timestamp, int(partition_revision), dataset, trading_date, *missing],
                )
            payloads = [
                (
                    dataset,
                    symbol,
                    trading_date,
                    "WRITTEN",
                    str((source_names or {}).get(symbol) or ""),
                    task_key,
                    timestamp,
                    int(partition_revision),
                    None,
                )
                for symbol in normalized
            ]
            connection.executemany(
                """
                INSERT INTO canonical_units (
                  dataset, symbol, trading_date, status, source_name, task_key,
                  written_at, partition_revision, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, symbol, trading_date)
                DO UPDATE SET
                  status = excluded.status,
                  source_name = excluded.source_name,
                  task_key = COALESCE(excluded.task_key, canonical_units.task_key),
                  written_at = excluded.written_at,
                  partition_revision = excluded.partition_revision,
                  last_error = excluded.last_error
                """,
                payloads,
            )

    def mark_canonical_units_status(
        self,
        *,
        dataset: str,
        trading_date: str,
        symbols: list[str] | tuple[str, ...] | None,
        status: str,
        last_error: str | None = None,
    ) -> int:
        """Mark canonical units for a date/symbol scope with an explicit status."""
        timestamp = self._now().isoformat()
        with self._connect() as connection:
            params: list[object] = [status, last_error, timestamp, dataset, trading_date]
            query = """
                UPDATE canonical_units
                SET status = ?,
                    last_error = ?,
                    written_at = ?
                WHERE dataset = ?
                  AND trading_date = ?
            """
            if symbols:
                normalized = sorted({str(symbol).upper() for symbol in symbols})
                placeholders = ",".join("?" for _ in normalized)
                query += f" AND symbol IN ({placeholders})"
                params.extend(normalized)
            cursor = connection.execute(query, params)
        return int(cursor.rowcount)

    def fetch_canonical_progress(
        self,
        *,
        dataset: str,
        symbols: list[str] | tuple[str, ...],
        trading_days: list[str] | tuple[str, ...],
    ) -> dict[str, object]:
        """Return canonical progress for a symbol/date scope from the durable ledger."""
        normalized_symbols = sorted({str(symbol).upper() for symbol in symbols})
        normalized_days = sorted({str(day) for day in trading_days})
        counts_by_symbol = {symbol: 0 for symbol in normalized_symbols}
        if not normalized_symbols or not normalized_days:
            return {
                "trading_days": normalized_days,
                "completed_symbols": [],
                "remaining_symbols": normalized_symbols,
                "expected_units": len(normalized_symbols) * len(normalized_days),
                "completed_units": 0,
                "remaining_units": len(normalized_symbols) * len(normalized_days),
            }
        with self._connect() as connection:
            symbol_placeholders = ",".join("?" for _ in normalized_symbols)
            day_placeholders = ",".join("?" for _ in normalized_days)
            rows = connection.execute(
                f"""
                SELECT symbol, trading_date
                FROM canonical_units
                WHERE dataset = ?
                  AND status = 'WRITTEN'
                  AND symbol IN ({symbol_placeholders})
                  AND trading_date IN ({day_placeholders})
                """,
                [dataset, *normalized_symbols, *normalized_days],
            ).fetchall()
        completed_pairs: set[tuple[str, str]] = set()
        for row in rows:
            symbol = str(row["symbol"]).upper()
            trading_date = str(row["trading_date"])
            pair = (trading_date, symbol)
            if pair in completed_pairs:
                continue
            completed_pairs.add(pair)
            counts_by_symbol[symbol] = counts_by_symbol.get(symbol, 0) + 1
        completed_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count >= len(normalized_days))
        remaining_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count < len(normalized_days))
        expected_units = len(normalized_symbols) * len(normalized_days)
        completed_units = len(completed_pairs)
        return {
            "trading_days": normalized_days,
            "completed_symbols": completed_symbols,
            "remaining_symbols": remaining_symbols,
            "expected_units": expected_units,
            "completed_units": completed_units,
            "remaining_units": max(0, expected_units - completed_units),
        }

    def fetch_canonical_units_for_date(
        self,
        *,
        dataset: str,
        trading_date: str,
        statuses: tuple[str, ...] | None = None,
    ) -> list[CanonicalUnit]:
        """Return canonical units for a trading date."""
        query = "SELECT * FROM canonical_units WHERE dataset = ? AND trading_date = ?"
        params: list[object] = [dataset, trading_date]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY symbol"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [CanonicalUnit(**dict(row)) for row in rows]

    def upsert_raw_partition_manifest(
        self,
        *,
        dataset: str,
        trading_date: str,
        partition_revision: int,
        symbol_count: int,
        row_count: int,
        symbols: list[str] | tuple[str, ...],
        content_hash: str | None,
        status: str,
        last_compacted_at: str | None = None,
    ) -> None:
        """Upsert the durable manifest row for a compacted raw partition."""
        timestamp = last_compacted_at or self._now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_partition_manifest (
                  dataset, trading_date, partition_revision, symbol_count, row_count,
                  symbols_json, content_hash, last_compacted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, trading_date)
                DO UPDATE SET
                  partition_revision = excluded.partition_revision,
                  symbol_count = excluded.symbol_count,
                  row_count = excluded.row_count,
                  symbols_json = excluded.symbols_json,
                  content_hash = excluded.content_hash,
                  last_compacted_at = excluded.last_compacted_at,
                  status = excluded.status
                """,
                (
                    dataset,
                    trading_date,
                    int(partition_revision),
                    int(symbol_count),
                    int(row_count),
                    json.dumps(sorted({str(symbol).upper() for symbol in symbols}), sort_keys=True),
                    content_hash,
                    timestamp,
                    status,
                ),
            )

    def get_raw_partition_manifest(self, *, dataset: str, trading_date: str) -> RawPartitionManifest | None:
        """Return the raw partition manifest for a date when present."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM raw_partition_manifest
                WHERE dataset = ? AND trading_date = ?
                """,
                (dataset, trading_date),
            ).fetchone()
        return RawPartitionManifest(**dict(row)) if row is not None else None

    def fetch_raw_partition_manifests(
        self,
        *,
        dataset: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> list[RawPartitionManifest]:
        """Return raw partition manifests filtered by dataset/status."""
        query = "SELECT * FROM raw_partition_manifest"
        clauses: list[str] = []
        params: list[object] = []
        if dataset:
            clauses.append("dataset = ?")
            params.append(dataset)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY trading_date ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [RawPartitionManifest(**dict(row)) for row in rows]

    def fetch_recent_raw_partition_dates(
        self,
        *,
        dataset: str,
        statuses: tuple[str, ...],
        limit: int = 10,
    ) -> list[str]:
        """Return the newest raw partition dates matching one dataset/status filter."""
        placeholders = ",".join("?" for _ in statuses)
        params: list[object] = [dataset, *statuses, int(limit)]
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT trading_date
                FROM raw_partition_manifest
                WHERE dataset = ?
                  AND status IN ({placeholders})
                ORDER BY trading_date DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [str(row["trading_date"]) for row in rows]

    def mark_raw_partition_manifest_status(
        self,
        *,
        dataset: str,
        trading_date: str,
        status: str,
    ) -> None:
        """Update only the manifest health status for a date."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE raw_partition_manifest
                SET status = ?,
                    last_compacted_at = ?
                WHERE dataset = ? AND trading_date = ?
                """,
                (status, self._now().isoformat(), dataset, trading_date),
            )




class VendorAttemptStore(_Store):
    """Vendor attempts, leases, lane health, and throughput summaries."""

    def upsert_vendor_lane_health(
        self,
        *,
        vendor: str,
        dataset: str,
        state: str,
        cooldown_until: str | None = None,
        recent_outbound_requests: int = 0,
        recent_success_units: int = 0,
        recent_remote_429s: int = 0,
        recent_local_budget_blocks: int = 0,
        recent_empty_valid: int = 0,
        recent_permanent_failures: int = 0,
    ) -> None:
        """Persist the current scheduler health state for a vendor lane."""
        timestamp = self._now().isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT state, last_state_change FROM vendor_lane_health WHERE vendor = ? AND dataset = ?",
                (vendor, dataset),
            ).fetchone()
            last_state_change = timestamp
            if existing is not None and str(existing["state"]) == state:
                last_state_change = str(existing["last_state_change"])
            connection.execute(
                """
                INSERT INTO vendor_lane_health (
                  vendor, dataset, state, cooldown_until, last_state_change,
                  recent_outbound_requests, recent_success_units, recent_remote_429s,
                  recent_local_budget_blocks, recent_empty_valid, recent_permanent_failures, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vendor, dataset)
                DO UPDATE SET
                  state = excluded.state,
                  cooldown_until = excluded.cooldown_until,
                  last_state_change = excluded.last_state_change,
                  recent_outbound_requests = excluded.recent_outbound_requests,
                  recent_success_units = excluded.recent_success_units,
                  recent_remote_429s = excluded.recent_remote_429s,
                  recent_local_budget_blocks = excluded.recent_local_budget_blocks,
                  recent_empty_valid = excluded.recent_empty_valid,
                  recent_permanent_failures = excluded.recent_permanent_failures,
                  updated_at = excluded.updated_at
                """,
                (
                    vendor,
                    dataset,
                    state,
                    cooldown_until,
                    last_state_change,
                    int(recent_outbound_requests),
                    int(recent_success_units),
                    int(recent_remote_429s),
                    int(recent_local_budget_blocks),
                    int(recent_empty_valid),
                    int(recent_permanent_failures),
                    timestamp,
                ),
            )

    def get_vendor_lane_health(self, *, vendor: str, dataset: str) -> VendorLaneHealth | None:
        """Return current lane-health state for a vendor/dataset."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vendor_lane_health WHERE vendor = ? AND dataset = ?",
                (vendor, dataset),
            ).fetchone()
        return VendorLaneHealth(**dict(row)) if row is not None else None

    def get_active_vendor_lane_cooldown(
        self, *, vendor: str, dataset: str, now: datetime | None = None
    ) -> VendorLaneHealth | None:
        """Return active cooldown state for a vendor/dataset lane."""
        lane = self.get_vendor_lane_health(vendor=vendor, dataset=dataset)
        if lane is None or not lane.cooldown_until:
            return None
        reference_time = (now or self._now()).isoformat()
        if lane.cooldown_until <= reference_time:
            return None
        return lane

    def clear_expired_vendor_lane_cooldowns(
        self, *, now: datetime | None = None
    ) -> int:
        """Mark expired budget/rate cooldown lanes healthy for current-state views."""
        reference_time = (now or self._now()).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vendor_lane_health
                SET state = 'HEALTHY',
                    cooldown_until = NULL,
                    updated_at = ?
                WHERE state IN ('BUDGET_BLOCKED', 'COOLDOWN')
                  AND cooldown_until IS NOT NULL
                  AND cooldown_until <= ?
                """,
                (reference_time, reference_time),
            )
        return int(cursor.rowcount)

    def clear_stale_idle_vendor_lanes(
        self, *, older_than: datetime | None = None
    ) -> int:
        """Clear old idle-budget markers so they do not masquerade as current state."""
        cutoff = (older_than or (self._now() - timedelta(minutes=15))).isoformat()
        timestamp = self._now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vendor_lane_health
                SET state = 'HEALTHY',
                    cooldown_until = NULL,
                    updated_at = ?
                WHERE state = 'IDLE_BUDGET'
                  AND updated_at < ?
                """,
                (timestamp, cutoff),
            )
        return int(cursor.rowcount)

    def vendor_lane_health_map(self, *, dataset: str) -> dict[str, VendorLaneHealth]:
        """Return lane-health rows keyed by vendor."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vendor_lane_health WHERE dataset = ? ORDER BY vendor",
                (dataset,),
            ).fetchall()
        return {str(row["vendor"]): VendorLaneHealth(**dict(row)) for row in rows}

    def fetch_vendor_lane_health(
        self, *, states: tuple[str, ...] | None = None
    ) -> list[VendorLaneHealth]:
        """Return lane-health rows, optionally filtered by state."""
        query = "SELECT * FROM vendor_lane_health"
        params: list[object] = []
        if states:
            placeholders = ",".join("?" for _ in states)
            query += f" WHERE state IN ({placeholders})"
            params.extend(states)
        query += " ORDER BY updated_at DESC, vendor, dataset"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [VendorLaneHealth(**dict(row)) for row in rows]

    def lease_vendor_attempt(
        self,
        *,
        task_key: str,
        task_family: str,
        planner_group: str,
        vendor: str,
        lease_owner: str,
        payload: dict | None = None,
        allow_success_retry: bool = False,
        lease_ttl_seconds: int = 90,
        now: datetime | None = None,
    ) -> VendorAttempt | None:
        """Lease a vendor attempt if it is eligible and not already complete."""
        current = (now or self._now())
        lease_time = current.isoformat()
        expires_at = (current + timedelta(seconds=lease_ttl_seconds)).isoformat()
        payload_json = json.dumps(payload, sort_keys=True) if payload else None
        for _attempt in range(3):
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM vendor_attempts WHERE task_key = ? AND vendor = ?",
                    (task_key, vendor),
                ).fetchone()
                if row is not None:
                    status = str(row["status"])
                    lease_expiry = row["lease_expires_at"]
                    if status == "SUCCESS" and not allow_success_retry:
                        return None
                    if status == "LEASED" and lease_expiry and lease_expiry > lease_time and row["lease_owner"] != lease_owner:
                        return None
                    if row["next_eligible_at"] and row["next_eligible_at"] > lease_time:
                        return None
                    connection.execute(
                        """
                        UPDATE vendor_attempts
                        SET task_family = ?,
                            planner_group = ?,
                            lease_owner = ?,
                            status = 'LEASED',
                            leased_at = ?,
                            lease_expires_at = ?,
                            next_eligible_at = NULL,
                            last_error = NULL,
                            payload_json = COALESCE(?, payload_json),
                            updated_at = ?
                        WHERE task_key = ? AND vendor = ?
                        """,
                        (
                            task_family,
                            planner_group,
                            lease_owner,
                            lease_time,
                            expires_at,
                            payload_json,
                            lease_time,
                            task_key,
                            vendor,
                        ),
                    )
                else:
                    try:
                        connection.execute(
                            """
                            INSERT INTO vendor_attempts (
                              task_key, task_family, planner_group, vendor, lease_owner, status, attempts,
                              leased_at, lease_expires_at, payload_json, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'LEASED', 0, ?, ?, ?, ?)
                            """,
                            (
                                task_key,
                                task_family,
                                planner_group,
                                vendor,
                                lease_owner,
                                lease_time,
                                expires_at,
                                payload_json,
                                lease_time,
                            ),
                        )
                    except sqlite3.IntegrityError:
                        continue
                leased = connection.execute(
                    "SELECT * FROM vendor_attempts WHERE task_key = ? AND vendor = ?",
                    (task_key, vendor),
                ).fetchone()
            return VendorAttempt(**dict(leased)) if leased is not None else None
        return None

    def mark_vendor_attempt_success(
        self,
        *,
        task_key: str,
        vendor: str,
        rows_returned: int = 0,
    ) -> None:
        """Mark a vendor attempt successful."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vendor_attempts
                SET status = 'SUCCESS',
                    rows_returned = ?,
                    lease_owner = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE task_key = ? AND vendor = ?
                """,
                (rows_returned, self._now().isoformat(), task_key, vendor),
            )

    def mark_vendor_attempt_failed(
        self,
        *,
        task_key: str,
        vendor: str,
        error: str,
        backoff_minutes: int,
        permanent: bool = False,
    ) -> None:
        """Mark a vendor attempt failed with backoff or permanently."""
        status = "PERMANENT_FAILED" if permanent else "FAILED"
        next_eligible_at = None if permanent else (self._now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vendor_attempts
                SET status = ?,
                    attempts = attempts + 1,
                    last_error = ?,
                    next_eligible_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE task_key = ? AND vendor = ?
                """,
                (status, error, next_eligible_at, self._now().isoformat(), task_key, vendor),
            )

    def vendor_attempts_for_task(self, task_key: str) -> list[VendorAttempt]:
        """Return all vendor attempts for a deterministic task key."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vendor_attempts WHERE task_key = ? ORDER BY vendor",
                (task_key,),
            ).fetchall()
        return [VendorAttempt(**dict(row)) for row in rows]

    def vendor_attempts_for_symbol(
        self,
        *,
        vendor: str,
        symbol: str,
        task_family: str = "canonical_bars",
    ) -> list[VendorAttempt]:
        """Return vendor attempts whose payload scope includes the requested symbol."""
        pattern = f'%"{str(symbol).upper()}"%'
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM vendor_attempts
                WHERE vendor = ?
                  AND task_family = ?
                  AND payload_json LIKE ?
                ORDER BY updated_at DESC
                """,
                (vendor, task_family, pattern),
            ).fetchall()
        return [VendorAttempt(**dict(row)) for row in rows]

    def fetch_vendor_attempts(
        self,
        *,
        planner_group: str | None = None,
        updated_after: str | None = None,
        limit: int | None = None,
    ) -> list[VendorAttempt]:
        """Return all vendor attempts for dashboard/status views."""
        query = "SELECT * FROM vendor_attempts"
        clauses: list[str] = []
        params: list[object] = []
        if planner_group:
            clauses.append("planner_group = ?")
            params.append(planner_group)
        if updated_after:
            clauses.append("updated_at >= ?")
            params.append(updated_after)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, task_key, vendor"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [VendorAttempt(**dict(row)) for row in rows]

    def summarize_vendor_attempts(self) -> dict[str, object]:
        """Return aggregate vendor-attempt summary without loading the full table."""
        with self._connect() as connection:
            count_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM vendor_attempts
                GROUP BY status
                """
            ).fetchall()
            vendor_rows = connection.execute(
                """
                SELECT vendor,
                       COUNT(*) AS total,
                       SUM(CASE WHEN status = 'LEASED' THEN 1 ELSE 0 END) AS leased,
                       SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS success,
                       SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed,
                       SUM(CASE WHEN status = 'PERMANENT_FAILED' THEN 1 ELSE 0 END) AS permanent_failed,
                       MAX(updated_at) AS latest_update
                FROM vendor_attempts
                GROUP BY vendor
                ORDER BY vendor
                """
            ).fetchall()
            failure_rows = connection.execute(
                """
                SELECT vendor, task_key, status, last_error, updated_at
                FROM vendor_attempts
                WHERE status IN ('FAILED', 'PERMANENT_FAILED')
                  AND last_error IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 20
                """
            ).fetchall()
        return {
            "counts": {str(row["status"]): int(row["count"]) for row in count_rows},
            "by_vendor": [
                {
                    "vendor": str(row["vendor"]),
                    "total": int(row["total"] or 0),
                    "leased": int(row["leased"] or 0),
                    "success": int(row["success"] or 0),
                    "failed": int(row["failed"] or 0),
                    "permanent_failed": int(row["permanent_failed"] or 0),
                    "latest_update": row["latest_update"],
                }
                for row in vendor_rows
            ],
            "recent_failures": [
                {
                    "vendor": str(row["vendor"]),
                    "task_key": str(row["task_key"]),
                    "status": str(row["status"]),
                    "last_error": str(row["last_error"]),
                    "updated_at": str(row["updated_at"]),
                }
                for row in failure_rows
            ],
        }

    def summarize_lane_throughput(
        self, *, minutes: int = 15, limit: int | None = None
    ) -> list[dict[str, object]]:
        """Return recent row throughput by vendor and planner lane."""
        window_minutes = max(1, int(minutes))
        since = (self._now() - timedelta(minutes=window_minutes)).isoformat()
        params: list[object] = [since]
        query = """
            SELECT va.vendor AS vendor,
                   va.task_family AS task_family,
                   va.planner_group AS planner_group,
                   COALESCE(pt.dataset, '') AS dataset,
                   COUNT(*) AS attempts,
                   SUM(CASE WHEN va.status = 'SUCCESS' THEN 1 ELSE 0 END) AS successes,
                   SUM(CASE WHEN va.status = 'FAILED' THEN 1 ELSE 0 END) AS failures,
                   SUM(CASE WHEN va.status = 'LEASED' THEN 1 ELSE 0 END) AS leased,
                   SUM(COALESCE(va.rows_returned, 0)) AS rows_returned,
                   MAX(va.updated_at) AS latest_update
            FROM vendor_attempts va
            LEFT JOIN planner_tasks pt ON pt.task_key = va.task_key
            WHERE va.updated_at >= ?
            GROUP BY va.vendor, va.task_family, va.planner_group, COALESCE(pt.dataset, '')
            ORDER BY rows_returned DESC, attempts DESC, va.vendor ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            health_rows = connection.execute(
                """
                SELECT *
                FROM vendor_lane_health
                WHERE cooldown_until IS NOT NULL
                  AND cooldown_until > ?
                """,
                (self._now().isoformat(),),
            ).fetchall()
        health = {
            (str(row["vendor"]), str(row["dataset"])): VendorLaneHealth(**dict(row))
            for row in health_rows
        }
        summaries: list[dict[str, object]] = []
        for row in rows:
            dataset = str(row["dataset"] or "")
            lane = health.get((str(row["vendor"]), dataset))
            item = {
                "vendor": str(row["vendor"]),
                "task_family": str(row["task_family"]),
                "planner_group": str(row["planner_group"]),
                "dataset": dataset,
                "attempts": int(row["attempts"] or 0),
                "successes": int(row["successes"] or 0),
                "failures": int(row["failures"] or 0),
                "leased": int(row["leased"] or 0),
                "rows_returned": int(row["rows_returned"] or 0),
                "rows_per_minute": float(row["rows_returned"] or 0)
                / float(window_minutes),
                "latest_update": row["latest_update"],
            }
            if lane is not None:
                item.update(
                    {
                        "lane_state": lane.state,
                        "cooldown_until": lane.cooldown_until,
                        "blocked_reason": "budget_exhausted"
                        if lane.state == "BUDGET_BLOCKED"
                        else lane.state.lower(),
                        "recent_local_budget_blocks": lane.recent_local_budget_blocks,
                        "recent_remote_429s": lane.recent_remote_429s,
                    }
                )
            summaries.append(item)
        return summaries

    def release_vendor_attempt_leases_for_owner(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
        reason: str,
    ) -> int:
        """Release any in-flight vendor attempts owned by a restarting worker."""
        clauses = ["status = 'LEASED'", "lease_owner = ?"]
        params: list[object] = [lease_owner]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE vendor_attempts
                SET status = 'FAILED',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = COALESCE(last_error, ?),
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [reason, self._now().isoformat(), *params],
            )
        return int(cursor.rowcount)

    def release_expired_vendor_attempt_leases(
        self,
        *,
        now: datetime | None = None,
        task_families: tuple[str, ...] | None = None,
        reason: str,
    ) -> int:
        """Release vendor attempt leases whose TTL has expired."""
        reference_time = (now or self._now()).isoformat()
        clauses = [
            "status = 'LEASED'",
            "COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?",
        ]
        params: list[object] = [reference_time]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE vendor_attempts
                SET status = 'FAILED',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = COALESCE(last_error, ?),
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [reason, reference_time, *params],
            )
        return int(cursor.rowcount)




class PlannerTaskStore(_Store):
    """Planner task queue, progress, leases, and summaries."""

    def defer_planner_tasks_for_vendor_dataset(
        self,
        *,
        vendor: str,
        dataset: str,
        next_eligible_at: str,
        error: str,
        task_families: tuple[str, ...] | None = None,
    ) -> int:
        """Defer eligible planner tasks for one budget-blocked vendor/dataset lane."""
        clauses = [
            "dataset = ?",
            "eligible_vendors_json LIKE ?",
            "status IN ('PENDING', 'PARTIAL', 'FAILED')",
        ]
        where_params: list[object] = [dataset, f'%"{vendor}"%']
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            where_params.extend(task_families)
        params: list[object] = [
            next_eligible_at,
            error,
            self._now().isoformat(),
            *where_params,
        ]
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = 'PARTIAL',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE {" AND ".join(clauses)}
                """,
                params,
            )
        return int(cursor.rowcount)

    def mark_planner_tasks_permanent_for_vendor_dataset(
        self,
        *,
        vendor: str,
        dataset: str,
        error: str,
        task_families: tuple[str, ...] | None = None,
    ) -> int:
        """Permanently block all planner tasks for an unavailable vendor/dataset lane."""
        clauses = [
            "dataset = ?",
            "eligible_vendors_json LIKE ?",
            "status IN ('PENDING', 'PARTIAL', 'FAILED', 'LEASED')",
        ]
        params: list[object] = [dataset, f'%"{vendor}"%']
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = 'PERMANENT_FAILED',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE {" AND ".join(clauses)}
                """,
                (error, self._now().isoformat(), *params),
            )
        return int(cursor.rowcount)

    def has_pending_planner_tasks(
        self,
        *,
        task_families: tuple[str, ...] | None = None,
        datasets: tuple[str, ...] | None = None,
    ) -> bool:
        """Return whether any planner task is eligible to run."""
        query = """
            SELECT 1
            FROM planner_tasks
            WHERE status IN ('PENDING', 'PARTIAL', 'FAILED')
              AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
        """
        params: list[object] = [self._now().isoformat()]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            query += f" AND task_family IN ({placeholders})"
            params.extend(task_families)
        if datasets:
            placeholders = ",".join("?" for _ in datasets)
            query += f" AND dataset IN ({placeholders})"
            params.extend(datasets)
        query += " LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return row is not None

    def latest_planner_update(self, *, statuses: tuple[str, ...] | None = None, task_families: tuple[str, ...] | None = None) -> str | None:
        """Return the latest planner task update timestamp."""
        query = "SELECT MAX(updated_at) AS updated_at FROM planner_tasks"
        clauses: list[str] = []
        params: list[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return row["updated_at"]

    def mark_completed_planner_progress_success(
        self, *, task_families: tuple[str, ...]
    ) -> int:
        """Mark active planner tasks successful when persisted progress is complete."""
        if not task_families:
            return 0
        placeholders = ",".join("?" for _ in task_families)
        timestamp = self._now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = 'SUCCESS',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE task_family IN ({placeholders})
                  AND status IN ('PENDING', 'PARTIAL', 'FAILED', 'LEASED')
                  AND EXISTS (
                    SELECT 1
                    FROM planner_task_progress progress
                    WHERE progress.task_key = planner_tasks.task_key
                      AND progress.remaining_units <= 0
                  )
                """,
                [timestamp, *task_families],
            )
        return int(cursor.rowcount)

    def upsert_planner_task(
        self,
        *,
        task_key: str,
        task_family: str,
        planner_group: str,
        dataset: str,
        tier: str,
        priority: int,
        start_date: str,
        end_date: str,
        symbols: list[str] | tuple[str, ...],
        eligible_vendors: list[str] | tuple[str, ...],
        output_name: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """Insert or refresh a planner task without clobbering in-flight state."""
        timestamp = self._now().isoformat()
        symbols_json = json.dumps(list(symbols), sort_keys=True)
        vendors_json = json.dumps(list(eligible_vendors), sort_keys=True)
        payload_json = json.dumps(payload or {}, sort_keys=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT status, created_at FROM planner_tasks WHERE task_key = ?",
                (task_key,),
            ).fetchone()
            status = str(existing["status"]) if existing is not None else "PENDING"
            created_at = str(existing["created_at"]) if existing is not None else timestamp
            connection.execute(
                """
                INSERT INTO planner_tasks (
                  task_key, task_family, planner_group, dataset, tier, priority,
                  start_date, end_date, symbols_json, eligible_vendors_json, output_name,
                  payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  task_family = excluded.task_family,
                  planner_group = excluded.planner_group,
                  dataset = excluded.dataset,
                  tier = excluded.tier,
                  priority = excluded.priority,
                  start_date = excluded.start_date,
                  end_date = excluded.end_date,
                  symbols_json = excluded.symbols_json,
                  eligible_vendors_json = excluded.eligible_vendors_json,
                  output_name = excluded.output_name,
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                (
                    task_key,
                    task_family,
                    planner_group,
                    dataset,
                    tier,
                    priority,
                    start_date,
                    end_date,
                    symbols_json,
                    vendors_json,
                    output_name,
                    payload_json,
                    status,
                    created_at,
                    timestamp,
                ),
            )

    def bulk_upsert_planner_tasks(self, tasks: list[dict[str, object]]) -> None:
        """Insert or refresh planner tasks in a single transaction."""
        if not tasks:
            return
        timestamp = self._now().isoformat()
        with self._connect() as connection:
            existing_map: dict[str, tuple[str, str]] = {}
            task_keys = [str(task["task_key"]) for task in tasks]
            for index in range(0, len(task_keys), 500):
                chunk = task_keys[index : index + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT task_key, status, created_at FROM planner_tasks WHERE task_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    existing_map[str(row["task_key"])] = (str(row["status"]), str(row["created_at"]))
            prepared: list[tuple[object, ...]] = []
            for task in tasks:
                status, created_at = existing_map.get(str(task["task_key"]), ("PENDING", timestamp))
                prepared.append(
                    (
                        task["task_key"],
                        task["task_family"],
                        task["planner_group"],
                        task["dataset"],
                        task["tier"],
                        int(task["priority"]),
                        task["start_date"],
                        task["end_date"],
                        json.dumps(list(task["symbols"]), sort_keys=True),
                        json.dumps(list(task["eligible_vendors"]), sort_keys=True),
                        task.get("output_name"),
                        json.dumps(task.get("payload") or {}, sort_keys=True),
                        status,
                        created_at,
                        timestamp,
                    )
                )
            connection.executemany(
                """
                INSERT INTO planner_tasks (
                  task_key, task_family, planner_group, dataset, tier, priority,
                  start_date, end_date, symbols_json, eligible_vendors_json, output_name,
                  payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  task_family = excluded.task_family,
                  planner_group = excluded.planner_group,
                  dataset = excluded.dataset,
                  tier = excluded.tier,
                  priority = excluded.priority,
                  start_date = excluded.start_date,
                  end_date = excluded.end_date,
                  symbols_json = excluded.symbols_json,
                  eligible_vendors_json = excluded.eligible_vendors_json,
                  output_name = excluded.output_name,
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                WHERE planner_tasks.task_family IS NOT excluded.task_family
                   OR planner_tasks.planner_group IS NOT excluded.planner_group
                   OR planner_tasks.dataset IS NOT excluded.dataset
                   OR planner_tasks.tier IS NOT excluded.tier
                   OR planner_tasks.priority IS NOT excluded.priority
                   OR planner_tasks.start_date IS NOT excluded.start_date
                   OR planner_tasks.end_date IS NOT excluded.end_date
                   OR planner_tasks.symbols_json IS NOT excluded.symbols_json
                   OR planner_tasks.eligible_vendors_json IS NOT excluded.eligible_vendors_json
                   OR planner_tasks.output_name IS NOT excluded.output_name
                   OR planner_tasks.payload_json IS NOT excluded.payload_json
                """,
                prepared,
            )

    def prune_planner_tasks(self, *, task_families: tuple[str, ...], valid_task_keys: set[str]) -> int:
        """Delete planner tasks and associated state that are no longer part of the planned backlog."""
        if not task_families:
            return 0
        placeholders = ",".join("?" for _ in task_families)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT task_key FROM planner_tasks WHERE task_family IN ({placeholders})",
                list(task_families),
            ).fetchall()
            stale_keys = [str(row["task_key"]) for row in rows if str(row["task_key"]) not in valid_task_keys]
            if not stale_keys:
                return 0
            for index in range(0, len(stale_keys), 500):
                chunk = stale_keys[index : index + 500]
                chunk_placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM vendor_attempts WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
                connection.execute(
                    f"DELETE FROM planner_task_progress WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
                connection.execute(
                    f"DELETE FROM planner_tasks WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
        return len(stale_keys)

    def get_planner_task(self, task_key: str) -> PlannerTask | None:
        """Return a planner task by key."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM planner_tasks WHERE task_key = ?", (task_key,)).fetchone()
        return PlannerTask(**dict(row)) if row is not None else None

    def fetch_planner_tasks(
        self,
        *,
        task_family: str | None = None,
        task_families: tuple[str, ...] | None = None,
        planner_group: str | None = None,
        statuses: tuple[str, ...] | None = None,
        vendor: str | None = None,
        datasets: tuple[str, ...] | None = None,
        updated_after: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[PlannerTask]:
        """Return planner tasks filtered for status views."""
        query = "SELECT * FROM planner_tasks"
        clauses: list[str] = []
        params: list[object] = []
        if task_family:
            clauses.append("task_family = ?")
            params.append(task_family)
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        if planner_group:
            clauses.append("planner_group = ?")
            params.append(planner_group)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if vendor:
            clauses.append("eligible_vendors_json LIKE ?")
            params.append(f'%"{vendor}"%')
        if datasets:
            placeholders = ",".join("?" for _ in datasets)
            clauses.append(f"dataset IN ({placeholders})")
            params.extend(datasets)
        if updated_after:
            clauses.append("updated_at >= ?")
            params.append(updated_after)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority ASC, created_at ASC, task_key ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [PlannerTask(**dict(row)) for row in rows]

    def lease_next_planner_task(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
        vendor: str | None = None,
        datasets: tuple[str, ...] | None = None,
        now: datetime | None = None,
        lease_ttl_seconds: int = 300,
        limit: int = 256,
        scan_pages: int = 16,
    ) -> PlannerTask | None:
        """Lease the next eligible planner task, optionally filtered for a vendor."""
        lease_time = (now or self._now())
        lease_time_iso = lease_time.isoformat()
        for page in range(max(1, scan_pages)):
            candidates = self.fetch_planner_tasks(
                task_families=task_families,
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                vendor=vendor,
                datasets=datasets,
                limit=limit,
                offset=page * limit,
            )
            if not candidates:
                break
            for candidate in candidates:
                if task_families and candidate.task_family not in task_families:
                    continue
                if vendor and vendor not in candidate.eligible_vendors:
                    continue
                if candidate.status == "LEASED" and candidate.lease_expires_at and candidate.lease_expires_at > lease_time_iso:
                    continue
                if candidate.next_eligible_at and candidate.next_eligible_at > lease_time_iso:
                    continue
                expires_at = (lease_time + timedelta(seconds=lease_ttl_seconds)).isoformat()
                with self._connect() as connection:
                    updated = connection.execute(
                        """
                        UPDATE planner_tasks
                        SET status = 'LEASED',
                            lease_owner = ?,
                            leased_at = ?,
                            lease_expires_at = ?,
                            updated_at = ?
                        WHERE task_key = ?
                          AND (
                            status IN ('PENDING', 'PARTIAL', 'FAILED')
                            OR (status = 'LEASED' AND COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?)
                          )
                          AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
                        """,
                        (
                            lease_owner,
                            lease_time_iso,
                            expires_at,
                            lease_time_iso,
                            candidate.task_key,
                            lease_time_iso,
                            lease_time_iso,
                        ),
                    ).rowcount
                    if updated:
                        row = connection.execute(
                            "SELECT * FROM planner_tasks WHERE task_key = ?",
                            (candidate.task_key,),
                        ).fetchone()
                        return PlannerTask(**dict(row))
        return None

    def claim_next_vendor_task(
        self,
        *,
        vendor: str,
        lease_owner: str,
        task_families: tuple[str, ...],
        allowed_datasets: tuple[str, ...] | None = None,
        budget_decision_provider: Callable[[PlannerTask], Any] | None = None,
        now: datetime | None = None,
        lease_ttl_seconds: int = 300,
        limit: int = 256,
        scan_pages: int = 16,
    ) -> VendorTaskClaimResult:
        """Atomically claim a planner task and its vendor-attempt lease."""
        lease_time = now or self._now()
        lease_time_iso = lease_time.isoformat()
        expires_at = (lease_time + timedelta(seconds=lease_ttl_seconds)).isoformat()
        first_skip: VendorTaskClaimResult | None = None
        for page in range(max(1, scan_pages)):
            candidates = self.fetch_planner_tasks(
                task_families=task_families,
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                vendor=vendor,
                datasets=allowed_datasets,
                limit=limit,
                offset=page * limit,
            )
            if not candidates:
                break
            for candidate in candidates:
                if candidate.status == "LEASED" and candidate.lease_expires_at and candidate.lease_expires_at > lease_time_iso:
                    first_skip = first_skip or VendorTaskClaimResult(
                        task=None,
                        skip_reason="planner_task_leased",
                        dataset=candidate.dataset,
                    )
                    continue
                decision = (
                    budget_decision_provider(candidate)
                    if budget_decision_provider is not None
                    else None
                )
                stale_budget_backoff = False
                if candidate.next_eligible_at and candidate.next_eligible_at > lease_time_iso:
                    stale_budget_backoff = (
                        "budget" in str(candidate.last_error or "").lower()
                        and decision is not None
                        and bool(getattr(decision, "allowed", False))
                    )
                    if not stale_budget_backoff:
                        first_skip = first_skip or VendorTaskClaimResult(
                            task=None,
                            skip_reason="planner_task_backoff",
                            dataset=candidate.dataset,
                        )
                        continue
                if decision is not None and not bool(getattr(decision, "allowed", True)):
                    first_skip = first_skip or VendorTaskClaimResult(
                        task=None,
                        skip_reason=f"budget:{getattr(decision, 'blocked_dimension', 'unknown')}",
                        dataset=candidate.dataset,
                        budget_decision=decision,
                    )
                    continue
                with self._connect() as connection:
                    lane = connection.execute(
                        """
                        SELECT state, cooldown_until
                        FROM vendor_lane_health
                        WHERE vendor = ? AND dataset = ?
                        """,
                        (vendor, candidate.dataset),
                    ).fetchone()
                    if lane is not None and str(lane["state"]) == "ENTITLEMENT_BLOCKED":
                        first_skip = first_skip or VendorTaskClaimResult(
                            task=None,
                            skip_reason="entitlement_blocked",
                            dataset=candidate.dataset,
                        )
                        continue
                    active_lane_cooldown = (
                        lane is not None
                        and lane["cooldown_until"]
                        and str(lane["cooldown_until"]) > lease_time_iso
                    )
                    stale_budget_cooldown = (
                        active_lane_cooldown
                        and str(lane["state"]) == "BUDGET_BLOCKED"
                        and decision is not None
                        and bool(getattr(decision, "allowed", False))
                    )
                    if stale_budget_cooldown:
                        connection.execute(
                            """
                            UPDATE vendor_lane_health
                            SET state = 'HEALTHY',
                                cooldown_until = NULL,
                                updated_at = ?
                            WHERE vendor = ? AND dataset = ?
                            """,
                            (lease_time_iso, vendor, candidate.dataset),
                        )
                    elif active_lane_cooldown:
                        first_skip = first_skip or VendorTaskClaimResult(
                            task=None,
                            skip_reason=f"lane_cooldown:{lane['state']}",
                            dataset=candidate.dataset,
                        )
                        continue
                    if stale_budget_backoff:
                        connection.execute(
                            """
                            UPDATE planner_tasks
                            SET next_eligible_at = NULL,
                                updated_at = ?
                            WHERE task_key = ?
                            """,
                            (lease_time_iso, candidate.task_key),
                        )
                    attempt = connection.execute(
                        """
                        SELECT *
                        FROM vendor_attempts
                        WHERE task_key = ? AND vendor = ?
                        """,
                        (candidate.task_key, vendor),
                    ).fetchone()
                    if attempt is not None:
                        status = str(attempt["status"])
                        if status == "SUCCESS":
                            retry_empty_success = (
                                candidate.status in {"PARTIAL", "FAILED"}
                                and int(attempt["rows_returned"] or 0) == 0
                                and "empty result"
                                in str(candidate.last_error or "").lower()
                            )
                            if not retry_empty_success:
                                first_skip = first_skip or VendorTaskClaimResult(
                                    task=None,
                                    skip_reason="vendor_attempt_success",
                                    dataset=candidate.dataset,
                                )
                                continue
                        if status == "PERMANENT_FAILED":
                            first_skip = first_skip or VendorTaskClaimResult(
                                task=None,
                                skip_reason="vendor_attempt_permanent_failed",
                                dataset=candidate.dataset,
                            )
                            continue
                        if (
                            status == "LEASED"
                            and attempt["lease_expires_at"]
                            and str(attempt["lease_expires_at"]) > lease_time_iso
                            and str(attempt["lease_owner"]) != lease_owner
                        ):
                            first_skip = first_skip or VendorTaskClaimResult(
                                task=None,
                                skip_reason="vendor_attempt_leased",
                                dataset=candidate.dataset,
                            )
                            continue
                        if attempt["next_eligible_at"] and str(attempt["next_eligible_at"]) > lease_time_iso:
                            stale_attempt_budget_backoff = (
                                "budget" in str(attempt["last_error"] or "").lower()
                                and decision is not None
                                and bool(getattr(decision, "allowed", False))
                            )
                            if not stale_attempt_budget_backoff:
                                first_skip = first_skip or VendorTaskClaimResult(
                                    task=None,
                                    skip_reason="vendor_attempt_backoff",
                                    dataset=candidate.dataset,
                                )
                                continue
                    updated = connection.execute(
                        """
                        UPDATE planner_tasks
                        SET status = 'LEASED',
                            lease_owner = ?,
                            leased_at = ?,
                            lease_expires_at = ?,
                            updated_at = ?
                        WHERE task_key = ?
                          AND (
                            status IN ('PENDING', 'PARTIAL', 'FAILED')
                            OR (status = 'LEASED' AND COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?)
                          )
                          AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
                        """,
                        (
                            lease_owner,
                            lease_time_iso,
                            expires_at,
                            lease_time_iso,
                            candidate.task_key,
                            lease_time_iso,
                            lease_time_iso,
                        ),
                    ).rowcount
                    if not updated:
                        first_skip = first_skip or VendorTaskClaimResult(
                            task=None,
                            skip_reason="planner_claim_race",
                            dataset=candidate.dataset,
                        )
                        continue
                    payload_json = json.dumps(candidate.payload, sort_keys=True)
                    if attempt is None:
                        connection.execute(
                            """
                            INSERT INTO vendor_attempts (
                              task_key, task_family, planner_group, vendor, lease_owner,
                              status, attempts, leased_at, lease_expires_at, payload_json, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'LEASED', 0, ?, ?, ?, ?)
                            """,
                            (
                                candidate.task_key,
                                candidate.task_family,
                                candidate.planner_group,
                                vendor,
                                lease_owner,
                                lease_time_iso,
                                expires_at,
                                payload_json,
                                lease_time_iso,
                            ),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE vendor_attempts
                            SET task_family = ?,
                                planner_group = ?,
                                lease_owner = ?,
                                status = 'LEASED',
                                leased_at = ?,
                                lease_expires_at = ?,
                                next_eligible_at = NULL,
                                last_error = NULL,
                                payload_json = ?,
                                updated_at = ?
                            WHERE task_key = ? AND vendor = ?
                            """,
                            (
                                candidate.task_family,
                                candidate.planner_group,
                                lease_owner,
                                lease_time_iso,
                                expires_at,
                                payload_json,
                                lease_time_iso,
                                candidate.task_key,
                                vendor,
                            ),
                        )
                    row = connection.execute(
                        "SELECT * FROM planner_tasks WHERE task_key = ?",
                        (candidate.task_key,),
                    ).fetchone()
                    if row is not None:
                        return VendorTaskClaimResult(task=PlannerTask(**dict(row)))
        return first_skip or VendorTaskClaimResult(
            task=None, skip_reason="no_eligible_task"
        )

    def lease_planner_task_by_key(
        self,
        *,
        task_key: str,
        lease_owner: str,
        now: datetime | None = None,
        lease_ttl_seconds: int = 300,
    ) -> PlannerTask | None:
        """Lease a specific planner task if it is still eligible."""
        lease_time = (now or self._now())
        lease_time_iso = lease_time.isoformat()
        expires_at = (lease_time + timedelta(seconds=lease_ttl_seconds)).isoformat()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'LEASED',
                    lease_owner = ?,
                    leased_at = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE task_key = ?
                  AND (
                    status IN ('PENDING', 'PARTIAL', 'FAILED')
                    OR (status = 'LEASED' AND COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?)
                  )
                  AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
                """,
                (
                    lease_owner,
                    lease_time_iso,
                    expires_at,
                    lease_time_iso,
                    task_key,
                    lease_time_iso,
                    lease_time_iso,
                ),
            ).rowcount
            if not updated:
                return None
            row = connection.execute("SELECT * FROM planner_tasks WHERE task_key = ?", (task_key,)).fetchone()
        return PlannerTask(**dict(row)) if row is not None else None

    def mark_planner_task_success(self, task_key: str) -> None:
        """Mark a planner task successful."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'SUCCESS',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (self._now().isoformat(), task_key),
            )

    def mark_planner_task_partial(self, task_key: str, *, error: str | None = None, backoff_minutes: int = 1) -> None:
        """Return a planner task to partial state with optional backoff."""
        next_eligible = (self._now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'PARTIAL',
                    attempts = attempts + 1,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (next_eligible, error, self._now().isoformat(), task_key),
            )

    def mark_planner_task_failed(
        self,
        task_key: str,
        *,
        error: str,
        backoff_minutes: int,
        permanent: bool = False,
    ) -> None:
        """Mark a planner task failed with backoff or permanently."""
        status = "PERMANENT_FAILED" if permanent else "FAILED"
        next_eligible = None if permanent else (self._now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = ?,
                    attempts = attempts + 1,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (status, next_eligible, error, self._now().isoformat(), task_key),
            )

    def update_planner_task_progress(
        self,
        *,
        task_key: str,
        expected_units: int,
        completed_units: int,
        remaining_units: int,
        completed_symbols: list[str] | tuple[str, ...] | None = None,
        remaining_symbols: list[str] | tuple[str, ...] | None = None,
        state: dict | None = None,
    ) -> None:
        """Upsert planner task progress."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO planner_task_progress (
                  task_key, expected_units, completed_units, remaining_units,
                  completed_symbols_json, remaining_symbols_json, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  expected_units = excluded.expected_units,
                  completed_units = excluded.completed_units,
                  remaining_units = excluded.remaining_units,
                  completed_symbols_json = excluded.completed_symbols_json,
                  remaining_symbols_json = excluded.remaining_symbols_json,
                  state_json = excluded.state_json,
                  updated_at = excluded.updated_at
                """,
                (
                    task_key,
                    int(expected_units),
                    int(completed_units),
                    int(remaining_units),
                    json.dumps(list(completed_symbols or ()), sort_keys=True) if completed_symbols is not None else None,
                    json.dumps(list(remaining_symbols or ()), sort_keys=True) if remaining_symbols is not None else None,
                    json.dumps(state or {}, sort_keys=True) if state is not None else None,
                    self._now().isoformat(),
                ),
            )

    def bulk_update_planner_task_progress(self, rows: list[dict[str, object]]) -> None:
        """Upsert planner progress rows in a single transaction."""
        if not rows:
            return
        timestamp = self._now().isoformat()
        payloads = [
            (
                row["task_key"],
                int(row["expected_units"]),
                int(row["completed_units"]),
                int(row["remaining_units"]),
                json.dumps(list(row.get("completed_symbols") or ()), sort_keys=True) if row.get("completed_symbols") is not None else None,
                json.dumps(list(row.get("remaining_symbols") or ()), sort_keys=True) if row.get("remaining_symbols") is not None else None,
                json.dumps(row.get("state") or {}, sort_keys=True) if row.get("state") is not None else None,
                timestamp,
            )
            for row in rows
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO planner_task_progress (
                  task_key, expected_units, completed_units, remaining_units,
                  completed_symbols_json, remaining_symbols_json, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  expected_units = excluded.expected_units,
                  completed_units = excluded.completed_units,
                  remaining_units = excluded.remaining_units,
                  completed_symbols_json = excluded.completed_symbols_json,
                  remaining_symbols_json = excluded.remaining_symbols_json,
                  state_json = excluded.state_json,
                  updated_at = excluded.updated_at
                """,
                payloads,
            )

    def reopen_planner_tasks(self, task_keys: list[str] | tuple[str, ...], *, reason: str) -> int:
        """Reopen planner tasks whose coverage regressed and clear stale attempt state."""
        if not task_keys:
            return 0
        timestamp = self._now().isoformat()
        reopened = 0
        with self._connect() as connection:
            for index in range(0, len(task_keys), 500):
                chunk = [str(task_key) for task_key in task_keys[index : index + 500]]
                placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM vendor_attempts WHERE task_key IN ({placeholders})",
                    chunk,
                )
                cursor = connection.execute(
                    f"""
                    UPDATE planner_tasks
                    SET status = 'PENDING',
                        attempts = 0,
                        lease_owner = NULL,
                        leased_at = NULL,
                        lease_expires_at = NULL,
                        next_eligible_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE task_key IN ({placeholders})
                      AND status IN ('SUCCESS', 'PERMANENT_FAILED')
                    """,
                    [reason, timestamp, *chunk],
                )
                reopened += int(cursor.rowcount)
        return reopened

    def clear_planner_task_backoff(self, task_keys: list[str] | tuple[str, ...], *, reason: str) -> int:
        """Clear task-level planner backoff while preserving vendor attempt history."""
        if not task_keys:
            return 0
        timestamp = self._now().isoformat()
        updated = 0
        with self._connect() as connection:
            for index in range(0, len(task_keys), 500):
                chunk = [str(task_key) for task_key in task_keys[index : index + 500]]
                placeholders = ",".join("?" for _ in chunk)
                cursor = connection.execute(
                    f"""
                    UPDATE planner_tasks
                    SET status = CASE WHEN status = 'FAILED' THEN 'PARTIAL' ELSE status END,
                        lease_owner = NULL,
                        leased_at = NULL,
                        lease_expires_at = NULL,
                        next_eligible_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE task_key IN ({placeholders})
                      AND status IN ('PARTIAL', 'FAILED', 'LEASED')
                    """,
                    [reason, timestamp, *chunk],
                )
                updated += int(cursor.rowcount)
        return updated

    def clear_budget_backoff_for_vendor_dataset(
        self,
        *,
        vendor: str,
        dataset: str,
        task_families: tuple[str, ...],
        reason: str,
    ) -> int:
        """Clear stale budget-created planner backoff for one affordable vendor lane."""
        timestamp = self._now().isoformat()
        placeholders = ",".join("?" for _ in task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = CASE WHEN status = 'FAILED' THEN 'PARTIAL' ELSE status END,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE dataset = ?
                  AND eligible_vendors_json LIKE ?
                  AND task_family IN ({placeholders})
                  AND status IN ('PENDING', 'PARTIAL', 'FAILED', 'LEASED')
                  AND next_eligible_at IS NOT NULL
                  AND lower(COALESCE(last_error, '')) LIKE '%budget%'
                """,
                [
                    reason,
                    timestamp,
                    dataset,
                    f'%"{vendor}"%',
                    *task_families,
                ],
            )
        return int(cursor.rowcount)

    def release_planner_leases_for_owner(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
    ) -> int:
        """Release any in-flight planner leases owned by a restarting worker."""
        clauses = ["status = 'LEASED'", "lease_owner = ?"]
        params: list[object] = [lease_owner]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = CASE WHEN attempts > 0 THEN 'PARTIAL' ELSE 'PENDING' END,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [self._now().isoformat(), *params],
            )
        return int(cursor.rowcount)

    def release_expired_planner_leases(
        self,
        *,
        now: datetime | None = None,
        task_families: tuple[str, ...] | None = None,
    ) -> int:
        """Release planner leases whose TTL has expired."""
        reference_time = (now or self._now()).isoformat()
        clauses = [
            "status = 'LEASED'",
            "COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?",
        ]
        params: list[object] = [reference_time]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = CASE WHEN attempts > 0 THEN 'PARTIAL' ELSE 'PENDING' END,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = COALESCE(last_error, 'expired planner lease reclaimed'),
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [reference_time, *params],
            )
        return int(cursor.rowcount)

    def count_repairable_stale_success_canonical_tasks(self, *, only_future_blocked: bool) -> int:
        """Count incomplete canonical tasks blocked by stale budget failures despite prior success."""
        now_iso = self._now().isoformat()
        where = [
            "pt.task_family = 'canonical_bars'",
            "pt.status IN ('PARTIAL', 'FAILED', 'LEASED')",
            "pt.last_error LIKE '%budget exhausted%'",
            "prog.remaining_units > 0",
            "EXISTS (SELECT 1 FROM vendor_attempts va WHERE va.task_key = pt.task_key AND va.status = 'SUCCESS')",
        ]
        params: list[object] = []
        if only_future_blocked:
            where.append("pt.next_eligible_at IS NOT NULL")
            where.append("pt.next_eligible_at > ?")
            params.append(now_iso)
        else:
            where.append("(pt.next_eligible_at IS NULL OR pt.next_eligible_at <= ?)")
            params.append(now_iso)
        query = f"""
            SELECT COUNT(*)
            FROM planner_tasks pt
            JOIN planner_task_progress prog
              ON prog.task_key = pt.task_key
            WHERE {' AND '.join(where)}
        """
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row[0] if row is not None else 0)

    def fetch_planner_task_progress(self, task_key: str) -> PlannerTaskProgress | None:
        """Return progress for a planner task."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM planner_task_progress WHERE task_key = ?",
                (task_key,),
            ).fetchone()
        return PlannerTaskProgress(**dict(row)) if row is not None else None

    def planner_task_progress_map(self) -> dict[str, PlannerTaskProgress]:
        """Return planner progress keyed by task key."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM planner_task_progress").fetchall()
        return {str(row["task_key"]): PlannerTaskProgress(**dict(row)) for row in rows}

    def fetch_planner_task_status_map(
        self,
        *,
        task_families: tuple[str, ...] | None = None,
        task_keys: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, str]:
        """Return planner task status keyed by task key for a filtered scope."""
        clauses: list[str] = []
        params: list[object] = []
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        if task_keys:
            placeholders = ",".join("?" for _ in task_keys)
            clauses.append(f"task_key IN ({placeholders})")
            params.extend(str(task_key) for task_key in task_keys)
        query = "SELECT task_key, status FROM planner_tasks"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {str(row["task_key"]): str(row["status"]) for row in rows}

    def fetch_planner_task_progress_counts(
        self,
        task_keys: list[str] | tuple[str, ...],
    ) -> dict[str, tuple[int, int]]:
        """Return completed/remaining units keyed by task key for a targeted scope."""
        if not task_keys:
            return {}
        rows: list[sqlite3.Row] = []
        with self._connect() as connection:
            for index in range(0, len(task_keys), 500):
                chunk = [str(task_key) for task_key in task_keys[index : index + 500]]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT task_key, completed_units, remaining_units
                        FROM planner_task_progress
                        WHERE task_key IN ({placeholders})
                        """,
                        chunk,
                    ).fetchall()
                )
        return {
            str(row["task_key"]): (int(row["completed_units"]), int(row["remaining_units"]))
            for row in rows
        }

    def planner_summary(self) -> dict[str, object]:
        """Return aggregate planner counts and progress."""
        with self._connect() as connection:
            task_rows = connection.execute(
                """
                SELECT task_family, status, COUNT(*) AS count
                FROM planner_tasks
                GROUP BY task_family, status
                """
            ).fetchall()
            progress_rows = connection.execute(
                """
                SELECT planner_tasks.task_family AS task_family,
                       SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units,
                       SUM(
                         CASE
                           WHEN planner_tasks.status IN ('PENDING', 'PARTIAL', 'FAILED', 'LEASED')
                           THEN planner_task_progress.remaining_units
                           ELSE 0
                         END
                       ) AS remaining_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                GROUP BY planner_tasks.task_family
                """
            ).fetchall()
            backlog_rows = connection.execute(
                """
                SELECT CASE
                         WHEN json_extract(planner_tasks.payload_json, '$.backlog_class') IS NOT NULL
                           THEN json_extract(planner_tasks.payload_json, '$.backlog_class')
                         WHEN planner_tasks.planner_group = 'phase1_pinned_canonical' THEN 'phase1_pinned'
                         WHEN planner_tasks.planner_group = 'rolling_canonical' THEN 'rolling'
                         WHEN planner_tasks.planner_group = 'canonical_repair' THEN 'repair'
                         WHEN planner_tasks.task_family = 'canonical_bars' THEN 'canonical_bars'
                         WHEN planner_tasks.task_family = 'canonical_repair' THEN 'repair'
                         WHEN planner_tasks.task_family = 'security_master' THEN 'security_master'
                         WHEN planner_tasks.task_family = 'corp_actions' THEN 'reference_events'
                         WHEN planner_tasks.task_family = 'events_filings' THEN 'reference_events'
                         WHEN planner_tasks.task_family = 'macro' THEN 'macro'
                         WHEN planner_tasks.task_family = 'supplemental_research' THEN 'research_archive'
                         ELSE planner_tasks.planner_group
                       END AS backlog_class,
                       SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units,
                       SUM(
                         CASE
                           WHEN planner_tasks.status IN ('PENDING', 'PARTIAL', 'FAILED', 'LEASED')
                           THEN planner_task_progress.remaining_units
                           ELSE 0
                         END
                       ) AS remaining_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                GROUP BY backlog_class
                """
            ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for row in task_rows:
            family = str(row["task_family"])
            counts.setdefault(family, {})[str(row["status"])] = int(row["count"])
        progress: dict[str, dict[str, int]] = {}
        for row in progress_rows:
            progress[str(row["task_family"])] = {
                "expected_units": int(row["expected_units"] or 0),
                "completed_units": int(row["completed_units"] or 0),
                "remaining_units": int(row["remaining_units"] or 0),
            }
        backlog_progress: dict[str, dict[str, int]] = {}
        for row in backlog_rows:
            backlog_progress[str(row["backlog_class"])] = {
                "expected_units": int(row["expected_units"] or 0),
                "completed_units": int(row["completed_units"] or 0),
                "remaining_units": int(row["remaining_units"] or 0),
            }
        return {"counts": counts, "progress": progress, "backlog_progress": backlog_progress}

    def count_planner_tasks_by_family(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        updated_after: str | None = None,
    ) -> dict[str, int]:
        """Return planner task counts grouped by family for a filtered scope."""
        query = "SELECT task_family, COUNT(*) AS count FROM planner_tasks"
        clauses: list[str] = []
        params: list[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if updated_after:
            clauses.append("updated_at >= ?")
            params.append(updated_after)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY task_family"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {str(row["task_family"]): int(row["count"]) for row in rows}
