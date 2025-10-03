"""
Lightweight concurrency helpers for bounded parallelism on Raspberry Pi.

Provides:
- PartitionLockManager: per-partition locks to serialize writes
- worker_count: resolve worker counts from environment with safe defaults
"""

from __future__ import annotations

import os
import threading
from typing import Dict


class PartitionLockManager:
    """Process-local lock manager keyed by partition path.

    Prevents concurrent writes to the same logical partition (e.g.,
    raw/<source>/<table>/date=YYYY-MM-DD) within a process.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._global = threading.Lock()

    def lock_for(self, key: str) -> threading.Lock:
        with self._global:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._locks[key] = lk
            return lk


_PLM = PartitionLockManager()


def partition_lock(key: str) -> threading.Lock:
    """Get a process-local lock for a partition key."""
    return _PLM.lock_for(key)


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return max(1, v)
    except Exception:
        return default


def worker_count(default: int = 4) -> int:
    """Global node worker count (bounded for RPi).

    Env: NODE_WORKERS (default 4)
    """
    return _int_env("NODE_WORKERS", default)


def max_inflight_for(source_name: str, default: int = 2) -> int:
    """Per-source inflight limit.

    Env: NODE_MAX_INFLIGHT_<SOURCE> (uppercased), default conservative.
    """
    key = f"NODE_MAX_INFLIGHT_{source_name.upper()}"
    return _int_env(key, default)

