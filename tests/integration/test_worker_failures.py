from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data_node.db import PartitionStatus, TaskKind
from data_node.worker import QueueWorker
from data_node.fetchers import FetchResult, FetchStatus


class _FakeBudget:
    def can_spend(self, vendor: str, kind: TaskKind, tokens: int = 1) -> bool:  # noqa: ARG002
        return True


def test_rate_limited_sets_backoff_and_defers_leasing(node_db, temp_data_root, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))

    task_id = node_db.enqueue_task(
        dataset="equities_eod",
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-01",
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )
    task = node_db.lease_next_task(node_id="node-test", now=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert task and task.id == task_id

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    result = FetchResult(status=FetchStatus.RATE_LIMITED, error="429 rate limit")
    worker.handle_result(task, result)

    # Backoff applied (~60s); leasing immediately should fail
    early = node_db.lease_next_task(node_id="node-test", now=datetime.now(timezone.utc) + timedelta(seconds=10))
    assert early is None

    later = node_db.lease_next_task(node_id="node-test", now=datetime.now(timezone.utc) + timedelta(hours=2))
    assert later is not None


def test_not_entitled_marks_vendor_ineligible(node_db, temp_data_root, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    task_id = node_db.enqueue_task(
        dataset="options_chains",
        symbol="MSFT",
        start_date="2024-01-05",
        end_date="2024-01-05",
        kind=TaskKind.FORWARD,
        priority=0,
    )
    task = node_db.lease_next_task(node_id="node-test")
    assert task and task.id == task_id

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    result = FetchResult(status=FetchStatus.NOT_ENTITLED, error="403", vendor_used="finnhub")
    worker.handle_result(task, result)

    assert ("options_chains", "finnhub") in worker._vendor_ineligible
    stats = node_db.get_queue_stats()
    assert stats["by_status"].get("FAILED", 0) == 0  # should retry later


def test_storage_not_implemented_sets_gap(node_db, temp_data_root, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    task_id = node_db.enqueue_task(
        dataset="fundamentals",
        symbol="AAPL",
        start_date="2024-01-10",
        end_date="2024-01-10",
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )
    task = node_db.lease_next_task(node_id="node-test")
    assert task and task.id == task_id

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    result = FetchResult(
        status=FetchStatus.EMPTY,
        rows=0,
        qc_code="STORAGE_NOT_IMPLEMENTED",
        vendor_used="finnhub",
    )
    worker.handle_result(task, result)

    parts = node_db.get_partition_status_batch(
        table_name="fundamentals",
        symbols=["AAPL"],
        start_date="2024-01-10",
        end_date="2024-01-10",
    )
    assert parts
    assert parts[0]["status"] == PartitionStatus.AMBER.value
