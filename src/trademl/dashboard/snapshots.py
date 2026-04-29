"""Background dashboard snapshot cache and refresh loop."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

SnapshotBuilder = Callable[[Any], dict[str, Any]]


class DashboardSnapshotManager:
    """Build and serve dashboard snapshots outside the HTTP request path."""

    snapshot_version = 1
    stream_interval_seconds = 2.0

    def __init__(self, *, settings: Any, builders: dict[str, SnapshotBuilder]) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._consecutive_failures: dict[str, int] = {
            "game": 0,
            "live": 0,
            "status": 0,
        }
        self._successful_builds: dict[str, bool] = {
            "game": False,
            "live": False,
            "status": False,
        }
        self._refresh_intervals: dict[str, float] = {
            "game": 10.0,
            "live": 5.0,
            "status": 15.0,
        }
        self._eager_channels = {"game", "live"}
        self._active_channels = set(self._eager_channels)
        self._builders = builders
        self._next_refresh_at: dict[str, float] = {}

        for channel in self._builders:
            self._load_snapshot_from_disk(channel)
        for channel in self._eager_channels:
            self._build_initial_snapshot(channel)
            self._next_refresh_at[channel] = (
                time.monotonic() + self._refresh_intervals[channel]
            )
        self._thread = threading.Thread(
            target=self._run_loop, name="dashboard-snapshots", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the background refresh loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self, channel: str) -> dict[str, Any] | None:
        """Return the latest built snapshot envelope for one channel."""
        with self._lock:
            record = self._records.get(channel)
            if record is None:
                return None
            return json.loads(json.dumps(record, default=str))

    def get_latest_or_not_ready(self, channel: str) -> tuple[HTTPStatus, dict[str, Any]]:
        """Return the latest payload or a compact not-ready response."""
        self._ensure_channel_active(channel)
        payload = self.get_latest(channel)
        if payload is None:
            return (
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": "snapshot_not_ready", "channel": channel},
            )
        return HTTPStatus.OK, payload

    def refresh_once(self, channel: str) -> None:
        """Refresh a single channel immediately."""
        self._ensure_channel_active(channel)
        self._refresh_channel(channel)
        self._next_refresh_at[channel] = (
            time.monotonic() + self._refresh_intervals[channel]
        )

    def health_summary(self) -> dict[str, Any]:
        """Return compact snapshot freshness information for operator status."""
        with self._lock:
            now = datetime.now(tz=UTC)
            summary: dict[str, Any] = {}
            for channel in self._builders:
                record = self._records.get(channel)
                meta = record.get("meta", {}) if isinstance(record, dict) else {}
                built_at = meta.get("built_at")
                age_seconds: int | None = None
                if built_at:
                    try:
                        age_seconds = max(
                            0,
                            int(
                                (
                                    now - datetime.fromisoformat(str(built_at))
                                ).total_seconds()
                            ),
                        )
                    except ValueError:
                        age_seconds = None
                summary[channel] = {
                    "built_at": built_at,
                    "stale": bool(meta.get("stale", False)),
                    "build_ms": meta.get("build_ms"),
                    "consecutive_failures": self._consecutive_failures.get(channel, 0),
                    "age_seconds": age_seconds,
                    "source": meta.get("source"),
                    "error": meta.get("error"),
                }
            return summary

    def _run_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            now = time.monotonic()
            for channel, interval in self._refresh_intervals.items():
                if channel not in self._active_channels:
                    continue
                next_refresh_at = self._next_refresh_at.get(channel, now + interval)
                if now >= next_refresh_at:
                    self._refresh_channel(channel)
                    self._next_refresh_at[channel] = time.monotonic() + interval

    def _ensure_channel_active(self, channel: str) -> None:
        if channel in self._active_channels:
            return
        self._active_channels.add(channel)
        if self.get_latest(channel) is None:
            self._build_initial_snapshot(channel)
        self._next_refresh_at[channel] = (
            time.monotonic() + self._refresh_intervals[channel]
        )

    def _build_initial_snapshot(self, channel: str) -> None:
        try:
            self._build_fresh_snapshot(channel)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "dashboard_snapshot_initial_build_failed channel=%s", channel
            )
            self._mark_existing_snapshot_stale(
                channel, error=str(exc), source="disk_cache"
            )

    def _refresh_channel(self, channel: str) -> None:
        try:
            self._build_fresh_snapshot(channel)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("dashboard_snapshot_refresh_failed channel=%s", channel)
            self._mark_existing_snapshot_stale(
                channel, error=str(exc), source="stale_last_good"
            )

    def _build_fresh_snapshot(self, channel: str) -> None:
        builder = self._builders[channel]
        build_started_at = datetime.now(tz=UTC)
        started = time.monotonic()
        payload = builder(self.settings)
        build_ms = round((time.monotonic() - started) * 1000.0, 2)
        response = self._wrap_snapshot(
            payload,
            built_at=datetime.now(tz=UTC),
            build_started_at=build_started_at,
            build_ms=build_ms,
            stale=False,
            source="fresh",
            error=None,
        )
        with self._lock:
            self._records[channel] = response
            self._successful_builds[channel] = True
            self._consecutive_failures[channel] = 0
        self._persist_snapshot(channel, response)

    def _mark_existing_snapshot_stale(
        self, channel: str, *, error: str, source: str
    ) -> None:
        with self._lock:
            current = self._records.get(channel)
            self._consecutive_failures[channel] = (
                self._consecutive_failures.get(channel, 0) + 1
            )
            if current is None:
                return
            response = json.loads(json.dumps(current, default=str))
            meta = response.setdefault("meta", {})
            meta["stale"] = True
            meta["source"] = source
            meta["error"] = error
            response["meta"] = meta
            self._records[channel] = response

    def _wrap_snapshot(
        self,
        payload: dict[str, Any],
        *,
        built_at: datetime,
        build_started_at: datetime,
        build_ms: float,
        stale: bool,
        source: str,
        error: str | None,
    ) -> dict[str, Any]:
        response = dict(payload)
        response["meta"] = {
            "built_at": built_at.isoformat(),
            "build_started_at": build_started_at.isoformat(),
            "build_ms": build_ms,
            "stale": stale,
            "source": source,
            "error": error,
            "version": self.snapshot_version,
        }
        return response

    def _snapshot_path(self, channel: str) -> Path:
        return self.settings.local_state / f"dashboard_{channel}_snapshot.json"

    def _persist_snapshot(self, channel: str, payload: dict[str, Any]) -> None:
        path = self._snapshot_path(channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(temp_path, path)

    def _load_snapshot_from_disk(self, channel: str) -> None:
        path = self._snapshot_path(channel)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "dashboard_snapshot_disk_load_failed channel=%s path=%s error=%s",
                channel,
                path,
                exc,
            )
            return
        if not isinstance(payload, dict):
            return
        meta = payload.setdefault("meta", {})
        meta.setdefault("built_at", None)
        meta.setdefault("build_started_at", None)
        meta.setdefault("build_ms", None)
        meta["stale"] = bool(meta.get("stale", False))
        meta["source"] = "disk_cache"
        meta["error"] = meta.get("error")
        meta["version"] = self.snapshot_version
        with self._lock:
            self._records[channel] = payload
