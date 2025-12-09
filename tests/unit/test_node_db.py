from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data_node.db import PartitionStatus, TaskKind, TaskStatus


def test_enqueue_lease_retry_and_fail(node_db):
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-05",
        kind=TaskKind.GAP,
        priority=1,
    )
    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="MSFT",
        start_date="2024-01-01",
        end_date="2024-01-05",
        kind=TaskKind.FORWARD,
        priority=5,
    )

    task = node_db.lease_next_task(node_id="node-1", now=now)
    assert task is not None
    assert task.priority == 1
    assert task.status == TaskStatus.LEASED

    # First failure → backoff, still pending
    node_db.mark_task_failed(task.id, "network glitch", backoff_until=now + timedelta(minutes=5), max_attempts=3)
    assert node_db.get_queue_stats()["by_status"]["PENDING"] >= 2

    # Backoff respected
    during_backoff = node_db.lease_next_task(node_id="node-1", now=now + timedelta(minutes=1))
    assert during_backoff is not None
    assert during_backoff.id != task.id
    node_db.mark_task_done(during_backoff.id)

    retry = node_db.lease_next_task(node_id="node-1", now=now + timedelta(minutes=10))
    assert retry and retry.id == task.id

    # Second failure with tight max_attempts → mark FAILED
    node_db.mark_task_failed(retry.id, "fatal", max_attempts=1)
    assert node_db.get_failed_count() == 1


def test_vendor_specific_leasing(node_db):
    now = datetime(2024, 2, 2, tzinfo=timezone.utc)
    node_db.enqueue_task(
        dataset="corp_actions",
        symbol="AAPL",
        start_date="2024-02-01",
        end_date="2024-02-01",
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )
    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="NVDA",
        start_date="2024-02-01",
        end_date="2024-02-01",
        kind=TaskKind.BOOTSTRAP,
        priority=1,
    )

    leased = node_db.lease_next_task_for_vendor(
        vendor="alpaca",
        datasets=["equities_eod"],
        node_id="node-2",
        now=now,
    )
    assert leased is not None
    assert leased.dataset == "equities_eod"
    assert leased.symbol == "NVDA"


def test_partition_coverage_and_batch_queries(node_db):
    # Insert two GREEN, one RED to validate coverage math
    node_db.upsert_partition_status(
        source_name="alpaca",
        table_name="equities_eod",
        symbol="AAPL",
        dt="2024-01-02",
        status=PartitionStatus.GREEN,
        qc_score=1.0,
        row_count=100,
        expected_rows=100,
        qc_code="OK",
    )
    node_db.upsert_partition_status(
        source_name="alpaca",
        table_name="equities_eod",
        symbol="MSFT",
        dt="2024-01-02",
        status=PartitionStatus.RED,
        qc_score=0.5,
        row_count=50,
        expected_rows=100,
        qc_code="LOW_ROWS",
    )

    coverage = node_db.get_green_coverage(
        table_name="equities_eod",
        symbols=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )
    assert coverage == 0.5

    batch = node_db.get_partition_status_batch(
        table_name="equities_eod",
        symbols=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )
    assert len(batch) == 2
    assert {row["symbol"] for row in batch} == {"AAPL", "MSFT"}
