"""
Manifest Manager for tracking connector watermarks and backfill progress.

Provides a unified interface for local and S3-backed manifest storage with
atomic updates to prevent data corruption during concurrent access.

SSOT v2 Section 2.4: Backfill subsystem requires tracking:
- Connector bookmarks (last fetch date per vendor/dataset)
- Backfill marks (historical progress through gaps)
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger


@dataclass
class ConnectorBookmark:
    """Bookmark for a data connector tracking ingestion progress."""
    vendor: str
    dataset: str
    last_date: str  # ISO format date (YYYY-MM-DD)
    last_timestamp: str  # ISO format timestamp
    row_count: int
    status: str  # 'success', 'partial', 'failed'
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BackfillMark:
    """Mark tracking backfill progress for a dataset."""
    dataset: str
    symbol: Optional[str]
    window_start: str  # ISO date
    window_end: str  # ISO date
    current_position: str  # ISO date - where we are in the backfill
    rows_backfilled: int
    gaps_remaining: int
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    priority: int
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None


class ManifestManager:
    """
    Manages connector bookmarks and backfill marks with atomic local file updates.

    Uses write-to-temp-then-rename pattern for crash safety.
    """

    def _use_db_backend(self) -> bool:
        """Whether to use Postgres as the authoritative backend."""
        return os.getenv("MANIFESTS_BACKEND", "").lower() == "postgres"

    def _db_connect(self):
        import psycopg2

        return psycopg2.connect(
            host=os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "trademl"),
            password=os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "trademl"),
            dbname=os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "trademl"),
        )

    def _ensure_db_tables(self) -> None:
        """Create tables if they do not exist."""
        if not self._use_db_backend():
            return
        try:
            conn = self._db_connect()
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS connector_bookmarks (
                        vendor TEXT NOT NULL,
                        dataset TEXT NOT NULL,
                        last_date TEXT,
                        last_timestamp TEXT,
                        row_count INT,
                        status TEXT,
                        updated_at TEXT,
                        metadata JSONB,
                        PRIMARY KEY (vendor, dataset)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS backfill_marks (
                        dataset TEXT NOT NULL,
                        symbol TEXT,
                        window_start TEXT,
                        window_end TEXT,
                        current_position TEXT,
                        rows_backfilled INT,
                        gaps_remaining INT,
                        status TEXT,
                        priority INT,
                        created_at TEXT,
                        updated_at TEXT,
                        metadata JSONB,
                        PRIMARY KEY (dataset, symbol)
                    );
                    """
                )
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to ensure manifests tables: {e}")

    def __init__(
        self,
        manifests_dir: str = "data_layer/manifests",
        bookmarks_file: str = "bookmarks.json",
        backfill_marks_file: str = "backfill_marks.json",
    ):
        """
        Initialize manifest manager.

        Args:
            manifests_dir: Directory for manifest files
            bookmarks_file: Filename for connector bookmarks
            backfill_marks_file: Filename for backfill marks
        """
        self.manifests_dir = Path(manifests_dir)
        self.bookmarks_path = self.manifests_dir / bookmarks_file
        self.backfill_marks_path = self.manifests_dir / backfill_marks_file

        # Ensure directory exists
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        # Ensure DB tables if configured
        self._ensure_db_tables()

        # Initialize files if they don't exist
        self._init_files()

        logger.info(f"ManifestManager initialized: {manifests_dir}")

    def _init_files(self) -> None:
        """Initialize manifest files if they don't exist."""
        if not self.bookmarks_path.exists():
            self._atomic_write(self.bookmarks_path, {"bookmarks": {}, "version": 1})
            logger.info(f"Initialized {self.bookmarks_path}")

        if not self.backfill_marks_path.exists():
            self._atomic_write(self.backfill_marks_path, {"marks": {}, "version": 1})
            logger.info(f"Initialized {self.backfill_marks_path}")

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write data to file atomically using temp file + rename."""
        tmp_name = f".{path.stem}_{uuid.uuid4().hex[:8]}.tmp"
        tmp_path = path.parent / tmp_name

        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Atomic rename
            shutil.move(str(tmp_path), str(path))

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to write {path}: {e}") from e

    def _load_json(self, path: Path) -> dict:
        """Load JSON file with error handling."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {path}: {e}")
            return {}

    # ========== Connector Bookmarks ==========

    def get_bookmark(self, vendor: str, dataset: str) -> Optional[ConnectorBookmark]:
        """
        Get bookmark for a vendor/dataset combination.

        Args:
            vendor: Vendor name (e.g., 'alpaca', 'finnhub')
            dataset: Dataset name (e.g., 'equities_eod', 'options_chains')

        Returns:
            ConnectorBookmark if exists, None otherwise
        """
        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT vendor, dataset, last_date, last_timestamp, row_count, status, updated_at, metadata
                        FROM connector_bookmarks
                        WHERE vendor = %s AND dataset = %s
                        """,
                        (vendor, dataset),
                    )
                    row = cur.fetchone()
                conn.close()
                if row:
                    meta = row[7]
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = None
                    elif not isinstance(meta, dict):
                        meta = None
                    return ConnectorBookmark(
                        vendor=row[0],
                        dataset=row[1],
                        last_date=row[2],
                        last_timestamp=row[3],
                        row_count=int(row[4]),
                        status=row[5],
                        updated_at=row[6],
                        metadata=meta,
                    )
                return None
            except Exception as e:
                logger.warning(f"DB bookmark fetch failed, falling back to local: {e}")

        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})

        key = f"{vendor}:{dataset}"
        if key in bookmarks:
            return ConnectorBookmark(**bookmarks[key])
        return None

    def set_bookmark(
        self,
        vendor: str,
        dataset: str,
        last_date: str,
        row_count: int,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set or update bookmark for a vendor/dataset.

        Args:
            vendor: Vendor name
            dataset: Dataset name
            last_date: Last successfully fetched date (YYYY-MM-DD)
            row_count: Number of rows in last fetch
            status: Fetch status
            metadata: Optional additional metadata
        """
        try:
            now = datetime.now(datetime.UTC).isoformat()
        except Exception:
            from datetime import timezone
            now = datetime.now(timezone.utc).isoformat()

        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO connector_bookmarks
                            (vendor, dataset, last_date, last_timestamp, row_count, status, updated_at, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (vendor, dataset)
                        DO UPDATE SET
                            last_date = EXCLUDED.last_date,
                            last_timestamp = EXCLUDED.last_timestamp,
                            row_count = EXCLUDED.row_count,
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                        """,
                        (vendor, dataset, last_date, now, row_count, status, now, json.dumps(metadata) if metadata else None),
                    )
                conn.close()
                logger.debug(f"[DB] Bookmark updated: {vendor}:{dataset} -> {last_date} ({row_count} rows)")
                return
            except Exception as e:
                logger.warning(f"DB bookmark update failed, falling back to local: {e}")

        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})

        key = f"{vendor}:{dataset}"

        bookmarks[key] = asdict(ConnectorBookmark(
            vendor=vendor,
            dataset=dataset,
            last_date=last_date,
            last_timestamp=now,
            row_count=row_count,
            status=status,
            updated_at=now,
            metadata=metadata,
        ))

        data["bookmarks"] = bookmarks
        self._atomic_write(self.bookmarks_path, data)

        logger.debug(f"Bookmark updated: {key} -> {last_date} ({row_count} rows)")

    def get_all_bookmarks(self) -> Dict[str, ConnectorBookmark]:
        """Get all connector bookmarks."""
        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT vendor, dataset, last_date, last_timestamp, row_count, status, updated_at, metadata
                        FROM connector_bookmarks
                        """
                    )
                    rows = cur.fetchall()
                conn.close()
                out: Dict[str, ConnectorBookmark] = {}
                for r in rows:
                    meta = r[7]
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = None
                    elif not isinstance(meta, dict):
                        meta = None
                    key = f"{r[0]}:{r[1]}"
                    out[key] = ConnectorBookmark(
                        vendor=r[0],
                        dataset=r[1],
                        last_date=r[2],
                        last_timestamp=r[3],
                        row_count=int(r[4]),
                        status=r[5],
                        updated_at=r[6],
                        metadata=meta,
                    )
                return out
            except Exception as e:
                logger.warning(f"DB bookmark fetch failed, falling back to local: {e}")

        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})
        return {k: ConnectorBookmark(**v) for k, v in bookmarks.items()}

    def get_last_date(self, vendor: str, dataset: str) -> Optional[str]:
        """Get last successfully fetched date for vendor/dataset."""
        bookmark = self.get_bookmark(vendor, dataset)
        return bookmark.last_date if bookmark else None

    def delete_bookmark(self, vendor: str, dataset: str) -> bool:
        """Delete a bookmark."""
        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM connector_bookmarks WHERE vendor = %s AND dataset = %s",
                        (vendor, dataset),
                    )
                    deleted = cur.rowcount
                conn.close()
                if deleted:
                    logger.info(f"[DB] Bookmark deleted: {vendor}:{dataset}")
                    return True
            except Exception as e:
                logger.warning(f"DB bookmark delete failed, falling back to local: {e}")

        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})

        key = f"{vendor}:{dataset}"
        if key in bookmarks:
            del bookmarks[key]
            data["bookmarks"] = bookmarks
            self._atomic_write(self.bookmarks_path, data)
            logger.info(f"Bookmark deleted: {key}")
            return True
        return False

    # ========== Backfill Marks ==========

    def get_backfill_mark(
        self,
        dataset: str,
        symbol: Optional[str] = None
    ) -> Optional[BackfillMark]:
        """
        Get backfill progress mark for a dataset/symbol.

        Args:
            dataset: Dataset name
            symbol: Symbol (optional, for symbol-level backfills)

        Returns:
            BackfillMark if exists, None otherwise
        """
        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT dataset, symbol, window_start, window_end, current_position,
                               rows_backfilled, gaps_remaining, status, priority, created_at, updated_at, metadata
                        FROM backfill_marks
                        WHERE dataset = %s AND (symbol = %s OR (symbol IS NULL AND %s IS NULL))
                        """,
                        (dataset, symbol, symbol),
                    )
                    row = cur.fetchone()
                conn.close()
                if row:
                    meta = row[11]
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = None
                    elif not isinstance(meta, dict):
                        meta = None
                    return BackfillMark(
                        dataset=row[0],
                        symbol=row[1],
                        window_start=row[2],
                        window_end=row[3],
                        current_position=row[4],
                        rows_backfilled=int(row[5]),
                        gaps_remaining=int(row[6]),
                        status=row[7],
                        priority=int(row[8]),
                        created_at=row[9],
                        updated_at=row[10],
                        metadata=meta,
                    )
                return None
            except Exception as e:
                logger.warning(f"DB backfill mark fetch failed, falling back to local: {e}")

        data = self._load_json(self.backfill_marks_path)
        marks = data.get("marks", {})

        key = f"{dataset}:{symbol}" if symbol else dataset
        if key in marks:
            return BackfillMark(**marks[key])
        return None

    def set_backfill_mark(
        self,
        dataset: str,
        symbol: Optional[str],
        window_start: str,
        window_end: str,
        current_position: str,
        rows_backfilled: int,
        gaps_remaining: int,
        status: str = "in_progress",
        priority: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set or update backfill progress mark.

        Args:
            dataset: Dataset name
            symbol: Symbol (optional)
            window_start: Start of backfill window (YYYY-MM-DD)
            window_end: End of backfill window (YYYY-MM-DD)
            current_position: Current position in backfill (YYYY-MM-DD)
            rows_backfilled: Total rows backfilled so far
            gaps_remaining: Number of gaps still to fill
            status: Backfill status
            priority: Priority for queue ordering
            metadata: Optional additional metadata
        """
        data = self._load_json(self.backfill_marks_path)
        marks = data.get("marks", {})

        key = f"{dataset}:{symbol}" if symbol else dataset
        try:
            now = datetime.now(datetime.UTC).isoformat()
        except Exception:
            from datetime import timezone
            now = datetime.now(timezone.utc).isoformat()

        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO backfill_marks (
                            dataset, symbol, window_start, window_end, current_position,
                            rows_backfilled, gaps_remaining, status, priority, created_at, updated_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (dataset, symbol)
                        DO UPDATE SET
                            window_start = EXCLUDED.window_start,
                            window_end = EXCLUDED.window_end,
                            current_position = EXCLUDED.current_position,
                            rows_backfilled = EXCLUDED.rows_backfilled,
                            gaps_remaining = EXCLUDED.gaps_remaining,
                            status = EXCLUDED.status,
                            priority = EXCLUDED.priority,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            dataset, symbol, window_start, window_end, current_position,
                            rows_backfilled, gaps_remaining, status, priority, now, now,
                            json.dumps(metadata) if metadata else None,
                        ),
                    )
                conn.close()
                logger.debug(f"[DB] Backfill mark updated: {key} -> {current_position} ({status})")
                return
            except Exception as e:
                logger.warning(f"DB backfill mark update failed, falling back to local: {e}")

        # Preserve created_at if updating
        existing = marks.get(key, {})
        created_at = existing.get("created_at", now)

        marks[key] = asdict(BackfillMark(
            dataset=dataset,
            symbol=symbol,
            window_start=window_start,
            window_end=window_end,
            current_position=current_position,
            rows_backfilled=rows_backfilled,
            gaps_remaining=gaps_remaining,
            status=status,
            priority=priority,
            created_at=created_at,
            updated_at=now,
            metadata=metadata,
        ))

        data["marks"] = marks
        self._atomic_write(self.backfill_marks_path, data)

        logger.debug(f"Backfill mark updated: {key} -> {current_position} ({status})")

    def get_all_backfill_marks(self) -> Dict[str, BackfillMark]:
        """Get all backfill marks."""
        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT dataset, symbol, window_start, window_end, current_position,
                               rows_backfilled, gaps_remaining, status, priority, created_at, updated_at, metadata
                        FROM backfill_marks
                        """
                    )
                    rows = cur.fetchall()
                conn.close()
                out: Dict[str, BackfillMark] = {}
                for r in rows:
                    key = f"{r[0]}:{r[1]}" if r[1] is not None else r[0]
                    meta = r[11]
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = None
                    elif not isinstance(meta, dict):
                        meta = None
                    out[key] = BackfillMark(
                        dataset=r[0],
                        symbol=r[1],
                        window_start=r[2],
                        window_end=r[3],
                        current_position=r[4],
                        rows_backfilled=int(r[5]),
                        gaps_remaining=int(r[6]),
                        status=r[7],
                        priority=int(r[8]),
                        created_at=r[9],
                        updated_at=r[10],
                        metadata=meta,
                    )
                return out
            except Exception as e:
                logger.warning(f"DB backfill fetch failed, falling back to local: {e}")

        data = self._load_json(self.backfill_marks_path)
        marks = data.get("marks", {})
        return {k: BackfillMark(**v) for k, v in marks.items()}

    def get_pending_backfills(self) -> List[BackfillMark]:
        """Get all pending/in_progress backfill marks sorted by priority."""
        all_marks = self.get_all_backfill_marks()
        pending = [
            m for m in all_marks.values()
            if m.status in ("pending", "in_progress")
        ]
        return sorted(pending, key=lambda x: (x.priority, x.created_at))

    def complete_backfill(self, dataset: str, symbol: Optional[str] = None) -> None:
        """Mark a backfill as completed."""
        mark = self.get_backfill_mark(dataset, symbol)
        if mark:
            self.set_backfill_mark(
                dataset=dataset,
                symbol=symbol,
                window_start=mark.window_start,
                window_end=mark.window_end,
                current_position=mark.window_end,
                rows_backfilled=mark.rows_backfilled,
                gaps_remaining=0,
                status="completed",
                priority=mark.priority,
                metadata=mark.metadata,
            )

    def fail_backfill(
        self,
        dataset: str,
        symbol: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark a backfill as failed."""
        mark = self.get_backfill_mark(dataset, symbol)
        if mark:
            metadata = mark.metadata or {}
            metadata["last_error"] = error
            self.set_backfill_mark(
                dataset=dataset,
                symbol=symbol,
                window_start=mark.window_start,
                window_end=mark.window_end,
                current_position=mark.current_position,
                rows_backfilled=mark.rows_backfilled,
                gaps_remaining=mark.gaps_remaining,
                status="failed",
                priority=mark.priority,
                metadata=metadata,
            )

    def delete_backfill_mark(self, dataset: str, symbol: Optional[str] = None) -> bool:
        """Delete a backfill mark."""
        key = f"{dataset}:{symbol}" if symbol else dataset

        if self._use_db_backend():
            try:
                conn = self._db_connect()
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM backfill_marks WHERE dataset = %s AND (symbol = %s OR (symbol IS NULL AND %s IS NULL))",
                        (dataset, symbol, symbol),
                    )
                    deleted = cur.rowcount
                conn.close()
                if deleted:
                    logger.info(f"[DB] Backfill mark deleted: {key}")
                    return True
            except Exception as e:
                logger.warning(f"DB backfill delete failed, falling back to local: {e}")

        data = self._load_json(self.backfill_marks_path)
        marks = data.get("marks", {})

        if key in marks:
            del marks[key]
            data["marks"] = marks
            self._atomic_write(self.backfill_marks_path, data)
            logger.info(f"Backfill mark deleted: {key}")
            return True
        return False


# Singleton instance for easy access
_manager: Optional[ManifestManager] = None


def get_manifest_manager() -> ManifestManager:
    """Get singleton ManifestManager instance."""
    global _manager
    if _manager is None:
        _manager = ManifestManager()
    return _manager


def init_manifests() -> None:
    """Initialize manifest files. Call at system startup."""
    get_manifest_manager()
    logger.info("Manifests initialized")
