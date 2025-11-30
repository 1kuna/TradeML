"""
Bookmark manager for tracking ingestion progress with atomic updates.

Bookmarks track the last successfully ingested timestamp/date for each data source,
enabling resume-from-last-checkpoint behavior across restarts.
"""

import json
from typing import Optional, Dict
from datetime import datetime, date
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import tempfile

try:  # pragma: no cover
    from botocore.exceptions import ClientError  # type: ignore
except Exception:  # pragma: no cover
    class ClientError(Exception):
        pass
from loguru import logger

from .s3_client import S3Client


@dataclass
class Bookmark:
    """Bookmark for a data source."""
    source: str
    table: str
    last_timestamp: str  # ISO format timestamp or date
    last_row_count: int
    updated_at: str


class BookmarkManager:
    """
    Manages bookmarks for data ingestion with atomic persistence.

    S3-backed (ETag) when s3_client is provided; otherwise uses a local JSON
    file with atomic replace semantics to support Pi/SSD-only mode.
    """

    def __init__(
        self,
        s3_client: Optional[S3Client] = None,
        bookmark_key: str = "manifests/bookmarks.json",
        local_path: Optional[str] = None,
    ):
        """
        Initialize bookmark manager.

        Args:
            s3_client: S3 client instance or None for local mode
            bookmark_key: S3 key for bookmark file
            local_path: Local file path (used when s3_client is None)
        """
        self.s3 = s3_client
        self.bookmark_key = bookmark_key
        self.local_path = Path(local_path) if local_path else Path(bookmark_key)
        self._cache: Optional[Dict] = None
        self._etag: Optional[str] = None

        logger.info(f"BookmarkManager initialized: {bookmark_key} (local={self.s3 is None})")

    def _load(self) -> Dict[str, Bookmark]:
        """Load bookmarks from storage."""
        if self.s3:
            try:
                data, etag = self.s3.get_json(self.bookmark_key)
                self._etag = etag
                self._cache = {k: Bookmark(**v) for k, v in data.items()}
                logger.debug(f"Loaded {len(self._cache)} bookmarks (ETag: {etag})")
                return self._cache
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                    logger.info("No existing bookmarks, starting fresh")
                    self._cache = {}
                    self._etag = None
                    return {}
                raise
        else:
            path = self.local_path
            if not path.exists():
                logger.info("No existing local bookmarks, starting fresh")
                self._cache = {}
                return {}
            try:
                data = json.loads(path.read_text())
                self._cache = {k: Bookmark(**v) for k, v in data.items()}
                logger.debug(f"Loaded {len(self._cache)} local bookmarks from {path}")
                return self._cache
            except Exception as e:
                logger.warning(f"Failed to load local bookmarks ({path}): {e}; starting fresh")
                self._cache = {}
                return {}

    def _save(self, bookmarks: Dict[str, Bookmark]) -> bool:
        """
        Save bookmarks to S3 with optimistic locking.

        Returns:
            True if saved successfully, False if ETag mismatch (concurrent update)
        """
        data = {k: asdict(v) for k, v in bookmarks.items()}

        if self.s3:
            try:
                if self._etag:
                    # Update with ETag precondition
                    new_etag = self.s3.put_json(
                        key=self.bookmark_key,
                        data=data,
                        if_match=self._etag,
                    )
                else:
                    # Create new file
                    new_etag = self.s3.put_json(
                        key=self.bookmark_key,
                        data=data,
                    )

                self._etag = new_etag
                self._cache = bookmarks
                logger.debug(f"Saved {len(bookmarks)} bookmarks (ETag: {new_etag})")
                return True

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == 'PreconditionFailed':
                    logger.warning("Bookmark save failed: concurrent update detected")
                    return False
                raise
        else:
            path = self.local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkstemp(prefix="bookmarks_", suffix=".json", dir=str(path.parent))[1])
            try:
                tmp.write_text(json.dumps(data, indent=2))
                shutil.move(str(tmp), str(path))
                self._cache = bookmarks
                logger.debug(f"Saved {len(bookmarks)} local bookmarks to {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save local bookmarks to {path}: {e}")
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                return False

    def get(self, source: str, table: str) -> Optional[Bookmark]:
        """
        Get bookmark for a source/table.

        Args:
            source: Data source name (e.g., 'alpaca', 'iex')
            table: Table name (e.g., 'equities_bars', 'options_nbbo')

        Returns:
            Bookmark if exists, None otherwise
        """
        if self._cache is None:
            self._load()

        key = f"{source}:{table}"
        return self._cache.get(key)

    def set(
        self,
        source: str,
        table: str,
        last_timestamp: str,
        row_count: int,
        max_retries: int = 3,
    ) -> bool:
        """
        Set bookmark for a source/table with retry on concurrent updates.

        Args:
            source: Data source name
            table: Table name
            last_timestamp: Last successfully ingested timestamp (ISO format)
            row_count: Number of rows in last batch
            max_retries: Max retries on ETag mismatch

        Returns:
            True if saved successfully
        """
        key = f"{source}:{table}"

        for attempt in range(max_retries):
            # Reload to get latest state
            bookmarks = self._load()

            # Update bookmark
            bookmarks[key] = Bookmark(
                source=source,
                table=table,
                last_timestamp=last_timestamp,
                last_row_count=row_count,
                updated_at=datetime.utcnow().isoformat(),
            )

            # Try to save
            if self._save(bookmarks):
                logger.info(f"Bookmark updated: {key} -> {last_timestamp} ({row_count} rows)")
                return True

            logger.warning(f"Bookmark update retry {attempt + 1}/{max_retries}")

        logger.error(f"Failed to update bookmark after {max_retries} retries: {key}")
        return False

    def get_last_timestamp(self, source: str, table: str) -> Optional[str]:
        """Get last ingested timestamp for source/table."""
        bookmark = self.get(source, table)
        return bookmark.last_timestamp if bookmark else None

    def get_all(self) -> Dict[str, Bookmark]:
        """Get all bookmarks."""
        if self._cache is None:
            self._load()
        return self._cache.copy()

    def delete(self, source: str, table: str, max_retries: int = 3) -> bool:
        """Delete a bookmark."""
        key = f"{source}:{table}"

        for attempt in range(max_retries):
            bookmarks = self._load()

            if key not in bookmarks:
                logger.debug(f"Bookmark not found: {key}")
                return True

            del bookmarks[key]

            if self._save(bookmarks):
                logger.info(f"Bookmark deleted: {key}")
                return True

            logger.warning(f"Bookmark delete retry {attempt + 1}/{max_retries}")

        logger.error(f"Failed to delete bookmark after {max_retries} retries: {key}")
        return False
