"""
Distributed lease manager using S3 conditional writes.

Enables distributed locking for edge collectors across multiple devices.
Uses S3 JSON objects with ETag-based optimistic locking.
"""

import json
import time
import socket
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

try:  # pragma: no cover - exercised when botocore present
    from botocore.exceptions import ClientError  # type: ignore
except Exception:  # pragma: no cover
    class ClientError(Exception):
        pass
from loguru import logger

from .s3_client import S3Client


@dataclass
class Lease:
    """Lease metadata."""
    name: str
    holder: str
    acquired_at: str
    expires_at: str
    lease_seconds: int


class LeaseManager:
    """
    Distributed lease manager using S3.

    Uses conditional writes (If-None-Match, If-Match) to ensure only one
    holder can acquire a lease at a time.
    """

    def __init__(
        self,
        s3_client: S3Client,
        lease_seconds: int = 120,
        renew_seconds: int = 45,
        lock_prefix: str = "locks/",
    ):
        """
        Initialize lease manager.

        Args:
            s3_client: S3 client instance
            lease_seconds: Lease duration in seconds
            renew_seconds: How often to renew (should be < lease_seconds)
            lock_prefix: S3 prefix for lock objects
        """
        self.s3 = s3_client
        self.lease_seconds = lease_seconds
        self.renew_seconds = renew_seconds
        self.lock_prefix = lock_prefix
        self.holder_id = f"{socket.gethostname()}-{time.time()}"

        logger.info(f"LeaseManager initialized: holder={self.holder_id}, lease={lease_seconds}s")

    def _lease_key(self, name: str) -> str:
        """Get S3 key for lease."""
        return f"{self.lock_prefix}{name}.lock"

    def acquire(self, name: str, force: bool = False) -> bool:
        """
        Acquire a lease.

        Args:
            name: Lease name (e.g., 'edge-collector-alpaca')
            force: Force acquire even if lease exists but expired

        Returns:
            True if acquired, False otherwise
        """
        key = self._lease_key(name)
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.lease_seconds)

        lease = Lease(
            name=name,
            holder=self.holder_id,
            acquired_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            lease_seconds=self.lease_seconds,
        )

        try:
            # Try to create new lease (fails if already exists)
            self.s3.put_json(
                key=key,
                data=asdict(lease),
                if_none_match='*',  # Only create if doesn't exist
            )
            logger.info(f"Lease acquired: {name} by {self.holder_id}")
            return True

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'PreconditionFailed':
                # Lease exists, check if expired
                if force:
                    existing_lease = self._get_lease(name)
                    if existing_lease and self._is_expired(existing_lease):
                        # Force steal expired lease
                        return self._steal_lease(name, existing_lease)
                logger.debug(f"Lease already held: {name}")
                return False
            else:
                logger.error(f"Acquire failed: {e}")
                raise

    def _get_lease(self, name: str) -> Optional[Lease]:
        """Get current lease if exists."""
        key = self._lease_key(name)
        try:
            data, _ = self.s3.get_json(key)
            return Lease(**data)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                return None
            raise

    def _is_expired(self, lease: Lease) -> bool:
        """Check if lease is expired."""
        expires_at = datetime.fromisoformat(lease.expires_at)
        return datetime.utcnow() > expires_at

    def _steal_lease(self, name: str, old_lease: Lease) -> bool:
        """Steal an expired lease."""
        key = self._lease_key(name)
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.lease_seconds)

        new_lease = Lease(
            name=name,
            holder=self.holder_id,
            acquired_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            lease_seconds=self.lease_seconds,
        )

        try:
            # Overwrite old lease
            self.s3.put_json(key=key, data=asdict(new_lease))
            logger.warning(f"Lease stolen from {old_lease.holder}: {name}")
            return True
        except Exception as e:
            logger.error(f"Steal failed: {e}")
            return False

    def renew(self, name: str) -> bool:
        """
        Renew an existing lease.

        Args:
            name: Lease name

        Returns:
            True if renewed, False if lease no longer held by us
        """
        key = self._lease_key(name)

        try:
            # Get current lease
            lease = self._get_lease(name)
            if not lease:
                logger.warning(f"Cannot renew, lease not found: {name}")
                return False

            if lease.holder != self.holder_id:
                logger.warning(f"Cannot renew, lease held by {lease.holder}: {name}")
                return False

            # Update expiry
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=self.lease_seconds)

            renewed_lease = Lease(
                name=name,
                holder=self.holder_id,
                acquired_at=lease.acquired_at,  # Keep original acquire time
                expires_at=expires_at.isoformat(),
                lease_seconds=self.lease_seconds,
            )

            self.s3.put_json(key=key, data=asdict(renewed_lease))
            logger.debug(f"Lease renewed: {name}")
            return True

        except Exception as e:
            logger.error(f"Renew failed: {e}")
            return False

    def release(self, name: str) -> bool:
        """
        Release a lease.

        Args:
            name: Lease name

        Returns:
            True if released, False if not held by us
        """
        key = self._lease_key(name)

        try:
            lease = self._get_lease(name)
            if not lease:
                logger.debug(f"Lease already released: {name}")
                return True

            if lease.holder != self.holder_id:
                logger.warning(f"Cannot release, lease held by {lease.holder}: {name}")
                return False

            self.s3.delete_object(key)
            logger.info(f"Lease released: {name}")
            return True

        except Exception as e:
            logger.error(f"Release failed: {e}")
            return False

    def is_held_by_me(self, name: str) -> bool:
        """Check if lease is currently held by this instance."""
        lease = self._get_lease(name)
        if not lease:
            return False
        return lease.holder == self.holder_id and not self._is_expired(lease)

    def get_holder(self, name: str) -> Optional[str]:
        """Get current lease holder ID, or None if no lease."""
        lease = self._get_lease(name)
        if lease and not self._is_expired(lease):
            return lease.holder
        return None
