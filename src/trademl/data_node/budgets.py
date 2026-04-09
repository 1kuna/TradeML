"""Per-vendor request budget management."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading


UTC = timezone.utc
WINDOW_METRICS = (
    "outbound_requests",
    "local_budget_blocks",
    "remote_rate_limits",
    "permanent_failures",
    "empty_successes",
)


@dataclass(slots=True, frozen=True)
class VendorBudget:
    """Static per-vendor configuration."""

    rpm: int
    daily_cap: int


class BudgetManager:
    """Track per-vendor RPM and daily caps with FORWARD-task reserve support."""

    def __init__(
        self,
        config: dict[str, dict[str, int]],
        reserve_fraction: float = 0.10,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self.reserve_fraction = reserve_fraction
        self.vendor_limits = {
            vendor: VendorBudget(rpm=values["rpm"], daily_cap=values["daily_cap"]) for vendor, values in config.items()
        }
        self._lock = threading.RLock()
        self.request_windows: dict[str, deque[datetime]] = defaultdict(deque)
        self.daily_spend: dict[str, dict[str, int]] = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
        self.daily_requests: dict[str, dict[str, int]] = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
        self.telemetry_totals: dict[str, dict[str, int]] = defaultdict(self._empty_telemetry_totals)
        self.endpoint_telemetry: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        self.telemetry_windows: dict[str, dict[str, deque[datetime]]] = defaultdict(self._empty_telemetry_windows)
        self.day_anchor = datetime.now(tz=UTC).date()
        self.snapshot_path = Path(snapshot_path).expanduser() if snapshot_path else None
        self._restore_snapshot()
        self._persist_snapshot()

    @staticmethod
    def _empty_telemetry_totals() -> dict[str, int]:
        return {
            "outbound_requests": 0,
            "logical_units": 0,
            "request_cost_units": 0,
            "local_budget_blocks": 0,
            "remote_rate_limits": 0,
            "permanent_failures": 0,
            "empty_successes": 0,
        }

    @staticmethod
    def _empty_telemetry_windows() -> dict[str, deque[datetime]]:
        return {metric: deque() for metric in WINDOW_METRICS}

    def _ensure_endpoint_telemetry(self, vendor: str, endpoint: str) -> dict[str, int]:
        endpoint_map = self.endpoint_telemetry[vendor]
        if endpoint not in endpoint_map:
            endpoint_map[endpoint] = self._empty_telemetry_totals()
        return endpoint_map[endpoint]

    def _restore_snapshot(self) -> None:
        """Restore same-day budget state from the last persisted snapshot when available."""
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return
        try:
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        raw_anchor = payload.get("day_anchor")
        if not raw_anchor:
            return
        try:
            day_anchor = datetime.fromisoformat(f"{raw_anchor}T00:00:00+00:00").date()
        except ValueError:
            return
        if day_anchor != self.day_anchor:
            return
        vendors = payload.get("vendors")
        if not isinstance(vendors, dict):
            return
        for vendor in self.vendor_limits:
            snapshot = vendors.get(vendor, {})
            if not isinstance(snapshot, dict):
                continue
            spend = snapshot.get("daily_spend", {})
            if isinstance(spend, dict):
                self.daily_spend[vendor]["FORWARD"] = int(spend.get("FORWARD", 0) or 0)
                self.daily_spend[vendor]["OTHER"] = int(spend.get("OTHER", 0) or 0)
            requests = snapshot.get("daily_requests", {})
            if isinstance(requests, dict):
                self.daily_requests[vendor]["FORWARD"] = int(requests.get("FORWARD", 0) or 0)
                self.daily_requests[vendor]["OTHER"] = int(requests.get("OTHER", 0) or 0)
            window = self.request_windows[vendor]
            for raw_stamp in snapshot.get("window_timestamps", []) or []:
                try:
                    stamp = datetime.fromisoformat(str(raw_stamp))
                except ValueError:
                    continue
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                window.append(stamp)
            telemetry = snapshot.get("telemetry", {})
            if isinstance(telemetry, dict):
                totals = telemetry.get("totals", {})
                if isinstance(totals, dict):
                    vendor_totals = self.telemetry_totals[vendor]
                    for key in vendor_totals:
                        vendor_totals[key] = int(totals.get(key, 0) or 0)
                per_endpoint = telemetry.get("per_endpoint", {})
                if isinstance(per_endpoint, dict):
                    for endpoint, endpoint_totals in per_endpoint.items():
                        if not isinstance(endpoint_totals, dict):
                            continue
                        target = self._ensure_endpoint_telemetry(vendor, str(endpoint))
                        for key in target:
                            target[key] = int(endpoint_totals.get(key, 0) or 0)
                window_timestamps = telemetry.get("window_timestamps", {})
                if isinstance(window_timestamps, dict):
                    vendor_windows = self.telemetry_windows[vendor]
                    for metric in WINDOW_METRICS:
                        for raw_stamp in window_timestamps.get(metric, []) or []:
                            try:
                                stamp = datetime.fromisoformat(str(raw_stamp))
                            except ValueError:
                                continue
                            if stamp.tzinfo is None:
                                stamp = stamp.replace(tzinfo=UTC)
                            vendor_windows[metric].append(stamp)

    def _normalize(self, now: datetime | None) -> datetime:
        current = now or datetime.now(tz=UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        if current.date() != self.day_anchor:
            self.reset_daily(current)
        return current

    def _trim_window(self, vendor: str, now: datetime) -> None:
        window = self.request_windows[vendor]
        while window and (now - window[0]).total_seconds() >= 60:
            window.popleft()

    def _trim_telemetry_windows(self, vendor: str, now: datetime) -> None:
        windows = self.telemetry_windows[vendor]
        for metric in WINDOW_METRICS:
            window = windows[metric]
            while window and (now - window[0]).total_seconds() >= 60:
                window.popleft()

    def can_spend(
        self,
        vendor: str,
        task_kind: str = "OTHER",
        *,
        units: int = 1,
        now: datetime | None = None,
    ) -> bool:
        """Return whether a request can be issued for the vendor now."""
        with self._lock:
            current = self._normalize(now)
            self._trim_window(vendor, current)
            limits = self.vendor_limits[vendor]
            spend = self.daily_spend[vendor]
            requested_units = max(1, int(units))

            if len(self.request_windows[vendor]) >= limits.rpm:
                return False

            total_spend = spend["FORWARD"] + spend["OTHER"]
            reserved_units = max(1, int(limits.daily_cap * self.reserve_fraction))
            non_forward_ceiling = max(0, limits.daily_cap - reserved_units)

            if task_kind == "FORWARD":
                return total_spend + requested_units <= limits.daily_cap
            return (
                spend["OTHER"] + requested_units <= non_forward_ceiling
                and total_spend + requested_units <= limits.daily_cap
            )

    def _record_telemetry_event(
        self,
        *,
        vendor: str,
        endpoint: str,
        metric: str,
        now: datetime,
        request_units: int = 0,
        logical_units: int = 0,
    ) -> None:
        vendor_totals = self.telemetry_totals[vendor]
        endpoint_totals = self._ensure_endpoint_telemetry(vendor, endpoint)
        vendor_totals[metric] += 1
        endpoint_totals[metric] += 1
        if metric == "outbound_requests":
            vendor_totals["logical_units"] += max(0, int(logical_units))
            vendor_totals["request_cost_units"] += max(0, int(request_units))
            endpoint_totals["logical_units"] += max(0, int(logical_units))
            endpoint_totals["request_cost_units"] += max(0, int(request_units))
        if metric in WINDOW_METRICS:
            self.telemetry_windows[vendor][metric].append(now)

    def record_local_budget_block(
        self,
        vendor: str,
        *,
        endpoint: str = "__unknown__",
        now: datetime | None = None,
    ) -> None:
        """Record a local pre-request budget block."""
        with self._lock:
            current = self._normalize(now)
            self._trim_telemetry_windows(vendor, current)
            self._record_telemetry_event(vendor=vendor, endpoint=endpoint, metric="local_budget_blocks", now=current)
            self._persist_snapshot(now=current)

    def record_remote_rate_limit(
        self,
        vendor: str,
        *,
        endpoint: str = "__unknown__",
        now: datetime | None = None,
    ) -> None:
        """Record a remote 429 or equivalent throttle response."""
        with self._lock:
            current = self._normalize(now)
            self._trim_telemetry_windows(vendor, current)
            self._record_telemetry_event(vendor=vendor, endpoint=endpoint, metric="remote_rate_limits", now=current)
            self._persist_snapshot(now=current)

    def record_permanent_failure(
        self,
        vendor: str,
        *,
        endpoint: str = "__unknown__",
        now: datetime | None = None,
    ) -> None:
        """Record a permanent request failure."""
        with self._lock:
            current = self._normalize(now)
            self._trim_telemetry_windows(vendor, current)
            self._record_telemetry_event(vendor=vendor, endpoint=endpoint, metric="permanent_failures", now=current)
            self._persist_snapshot(now=current)

    def record_empty_success(
        self,
        vendor: str,
        *,
        endpoint: str = "__unknown__",
        now: datetime | None = None,
    ) -> None:
        """Record a valid empty response."""
        with self._lock:
            current = self._normalize(now)
            self._trim_telemetry_windows(vendor, current)
            self._record_telemetry_event(vendor=vendor, endpoint=endpoint, metric="empty_successes", now=current)
            self._persist_snapshot(now=current)

    def record_spend(
        self,
        vendor: str,
        task_kind: str = "OTHER",
        *,
        units: int = 1,
        endpoint: str = "__unknown__",
        logical_units: int = 1,
        now: datetime | None = None,
    ) -> None:
        """Record a completed request."""
        with self._lock:
            current = self._normalize(now)
            self._trim_window(vendor, current)
            self._trim_telemetry_windows(vendor, current)
            self.request_windows[vendor].append(current)
            bucket = "FORWARD" if task_kind == "FORWARD" else "OTHER"
            self.daily_requests[vendor][bucket] += 1
            self.daily_spend[vendor][bucket] += max(1, int(units))
            self._record_telemetry_event(
                vendor=vendor,
                endpoint=endpoint,
                metric="outbound_requests",
                now=current,
                request_units=max(1, int(units)),
                logical_units=max(0, int(logical_units)),
            )
            self._persist_snapshot(now=current)

    def reset_daily(self, now: datetime | None = None) -> None:
        """Reset all daily counters, typically at midnight local-to-UTC boundary."""
        with self._lock:
            current = now or datetime.now(tz=UTC)
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            self.day_anchor = current.date()
            self.daily_spend = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
            self.daily_requests = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
            self._persist_snapshot(now=current)

    def snapshot(self, now: datetime | None = None) -> dict[str, object]:
        """Return a JSON-serializable view of current budget usage."""
        with self._lock:
            current = self._normalize(now)
            vendors: dict[str, dict[str, object]] = {}
            for vendor, limits in self.vendor_limits.items():
                self._trim_window(vendor, current)
                self._trim_telemetry_windows(vendor, current)
                spend = self.daily_spend[vendor]
                request_counts = self.daily_requests[vendor]
                total_spend = spend["FORWARD"] + spend["OTHER"]
                total_requests = request_counts["FORWARD"] + request_counts["OTHER"]
                reserved_units = max(1, int(limits.daily_cap * self.reserve_fraction))
                non_forward_ceiling = max(0, limits.daily_cap - reserved_units)
                telemetry_totals = dict(self.telemetry_totals[vendor])
                telemetry_windows = self.telemetry_windows[vendor]
                vendors[vendor] = {
                    "rpm": limits.rpm,
                    "daily_cap": limits.daily_cap,
                    "reserve_fraction": self.reserve_fraction,
                    "reserved_units": reserved_units,
                    "non_forward_ceiling": non_forward_ceiling,
                    "window_used": len(self.request_windows[vendor]),
                    "window_timestamps": [stamp.isoformat() for stamp in self.request_windows[vendor]],
                    "daily_spend": {
                        "FORWARD": int(spend["FORWARD"]),
                        "OTHER": int(spend["OTHER"]),
                        "TOTAL": int(total_spend),
                    },
                    "daily_requests": {
                        "FORWARD": int(request_counts["FORWARD"]),
                        "OTHER": int(request_counts["OTHER"]),
                        "TOTAL": int(total_requests),
                    },
                    "telemetry": {
                        "totals": telemetry_totals,
                        "window_counts": {
                            metric: len(telemetry_windows[metric])
                            for metric in WINDOW_METRICS
                        },
                        "window_timestamps": {
                            metric: [stamp.isoformat() for stamp in telemetry_windows[metric]]
                            for metric in WINDOW_METRICS
                        },
                        "per_endpoint": {
                            endpoint: dict(endpoint_totals)
                            for endpoint, endpoint_totals in sorted(self.endpoint_telemetry[vendor].items())
                        },
                    },
                }
            return {
                "day_anchor": self.day_anchor.isoformat(),
                "checked_at": current.isoformat(),
                "writer_pid": os.getpid(),
                "vendors": vendors,
            }

    def _persist_snapshot(self, now: datetime | None = None) -> None:
        """Persist the current budget snapshot for the external dashboard process."""
        if self.snapshot_path is None:
            return
        payload = self.snapshot(now=now)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.snapshot_path.parent,
            prefix=f"{self.snapshot_path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            tmp_path = Path(handle.name)
        tmp_path.replace(self.snapshot_path)
