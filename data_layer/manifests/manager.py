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
        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})

        key = f"{vendor}:{dataset}"
        now = datetime.utcnow().isoformat()

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
        data = self._load_json(self.bookmarks_path)
        bookmarks = data.get("bookmarks", {})
        return {k: ConnectorBookmark(**v) for k, v in bookmarks.items()}

    def get_last_date(self, vendor: str, dataset: str) -> Optional[str]:
        """Get last successfully fetched date for vendor/dataset."""
        bookmark = self.get_bookmark(vendor, dataset)
        return bookmark.last_date if bookmark else None

    def delete_bookmark(self, vendor: str, dataset: str) -> bool:
        """Delete a bookmark."""
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
        now = datetime.utcnow().isoformat()

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
        data = self._load_json(self.backfill_marks_path)
        marks = data.get("marks", {})

        key = f"{dataset}:{symbol}" if symbol else dataset
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
