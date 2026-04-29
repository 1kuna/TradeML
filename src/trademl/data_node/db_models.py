"""Data-node SQLite row models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(tz=UTC)


@dataclass(slots=True)
class VendorAttempt:
    """Persistent per-vendor attempt state for canonical and supplemental tasks."""

    task_key: str
    task_family: str
    planner_group: str
    vendor: str
    lease_owner: str | None
    status: str
    attempts: int
    last_error: str | None
    next_eligible_at: str | None
    leased_at: str | None
    lease_expires_at: str | None
    rows_returned: int | None
    payload_json: str | None
    updated_at: str

    @property
    def payload(self) -> dict:
        """Deserialize the optional planner payload."""
        if not self.payload_json:
            return {}
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class PlannerTask:
    """Persistent planner-native work unit."""

    task_key: str
    task_family: str
    planner_group: str
    dataset: str
    tier: str
    priority: int
    start_date: str
    end_date: str
    symbols_json: str
    eligible_vendors_json: str
    output_name: str | None
    payload_json: str | None
    status: str
    lease_owner: str | None
    leased_at: str | None
    lease_expires_at: str | None
    next_eligible_at: str | None
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the symbol scope for the task."""
        try:
            raw = json.loads(self.symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def eligible_vendors(self) -> tuple[str, ...]:
        """Return the vendors allowed to serve the task."""
        try:
            raw = json.loads(self.eligible_vendors_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def payload(self) -> dict:
        """Deserialize the planner payload."""
        if not self.payload_json:
            return {}
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class PlannerTaskProgress:
    """Persistent progress state for a planner task."""

    task_key: str
    expected_units: int
    completed_units: int
    remaining_units: int
    completed_symbols_json: str | None
    remaining_symbols_json: str | None
    state_json: str | None
    updated_at: str

    @property
    def completed_symbols(self) -> tuple[str, ...]:
        """Return the symbols fully covered for the task."""
        if not self.completed_symbols_json:
            return ()
        try:
            raw = json.loads(self.completed_symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def remaining_symbols(self) -> tuple[str, ...]:
        """Return the symbols still missing coverage for the task."""
        if not self.remaining_symbols_json:
            return ()
        try:
            raw = json.loads(self.remaining_symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def state(self) -> dict:
        """Return the planner progress payload."""
        if not self.state_json:
            return {}
        try:
            return json.loads(self.state_json)
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class CanonicalUnit:
    """Durable canonical symbol-date coverage unit."""

    dataset: str
    symbol: str
    trading_date: str
    status: str
    source_name: str | None
    task_key: str | None
    written_at: str | None
    partition_revision: int
    last_error: str | None


@dataclass(slots=True)
class RawPartitionManifest:
    """Metadata describing the compacted raw partition for a trading date."""

    dataset: str
    trading_date: str
    partition_revision: int
    symbol_count: int
    row_count: int
    symbols_json: str | None
    content_hash: str | None
    last_compacted_at: str | None
    status: str

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the compacted symbol set for this partition."""
        if not self.symbols_json:
            return ()
        try:
            raw = json.loads(self.symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)


@dataclass(slots=True)
class VendorLaneHealth:
    """Durable scheduler health state for a vendor/dataset lane."""

    vendor: str
    dataset: str
    state: str
    cooldown_until: str | None
    last_state_change: str
    recent_outbound_requests: int
    recent_success_units: int
    recent_remote_429s: int
    recent_local_budget_blocks: int
    recent_empty_valid: int
    recent_permanent_failures: int
    updated_at: str

