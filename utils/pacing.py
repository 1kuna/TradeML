"""
Global request pacing utility to smooth outbound API calls across threads.

Implements per-vendor pacing using a simple inter-request interval schedule:
- Each vendor has a next-allowed timestamp; calls wait until that time.
- Small positive jitter de-synchronizes bursts in multi-threaded scenarios.

Environment knobs:
- REQUEST_PACING_ENABLED=true|false (default true)
- REQUEST_PACING_JITTER_MS=min,max (default 50,150)
"""

from __future__ import annotations

import os
import time
import random
import threading
from dataclasses import dataclass
from typing import Dict, Tuple


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _env_int_pair(name: str, default: Tuple[int, int]) -> Tuple[int, int]:
    v = os.getenv(name)
    if not v:
        return default
    try:
        parts = [int(x.strip()) for x in v.split(",")]
        if len(parts) == 2 and parts[0] >= 0 and parts[1] >= parts[0]:
            return (parts[0], parts[1])
    except Exception:
        pass
    return default


@dataclass
class _VendorState:
    interval_s: float
    next_allowed_ts: float


class RequestPacer:
    _instance = None
    _global_lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = _env_bool("REQUEST_PACING_ENABLED", True)
        self.jitter_ms_range = _env_int_pair("REQUEST_PACING_JITTER_MS", (50, 150))
        self._states: Dict[str, _VendorState] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "RequestPacer":
        if cls._instance is None:
            with cls._global_lock:
                if cls._instance is None:
                    cls._instance = RequestPacer()
        return cls._instance

    def _lock_for(self, vendor: str) -> threading.Lock:
        with self._locks_lock:
            lk = self._locks.get(vendor)
            if lk is None:
                lk = threading.Lock()
                self._locks[vendor] = lk
            return lk

    def _state_for(self, vendor: str, rps: float) -> _VendorState:
        st = self._states.get(vendor)
        if st is None:
            interval = 1.0 / max(1e-6, rps)
            st = _VendorState(interval_s=interval, next_allowed_ts=0.0)
            self._states[vendor] = st
        else:
            # Update interval if rps changed materially
            new_interval = 1.0 / max(1e-6, rps)
            if abs(new_interval - st.interval_s) / max(1e-6, st.interval_s) > 0.05:
                st.interval_s = new_interval
        return st

    def acquire(self, vendor: str, rps: float) -> None:
        if not self.enabled:
            return
        lk = self._lock_for(vendor)

        # Calculate sleep time and schedule next slot inside lock (fast)
        with lk:
            st = self._state_for(vendor, rps)
            now = time.time()
            wait_for = st.next_allowed_ts - now
            if wait_for > 0:
                # Apply small positive jitter to avoid lockstep
                jmin, jmax = self.jitter_ms_range
                jitter = random.uniform(jmin / 1000.0, jmax / 1000.0)
                sleep_time = wait_for + jitter
            else:
                sleep_time = 0
            # Schedule next slot BEFORE releasing lock
            # This ensures proper spacing even with concurrent threads
            base = max(now, st.next_allowed_ts)
            st.next_allowed_ts = base + st.interval_s

        # Sleep OUTSIDE lock - allows other threads to schedule their slots
        if sleep_time > 0:
            time.sleep(sleep_time)

