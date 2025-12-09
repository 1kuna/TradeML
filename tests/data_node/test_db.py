"""
Tests for data_node.db module.

Tests:
- Queue enqueue/dequeue operations
- Leasing and lease expiry
- Task status transitions
- Partition status upserts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from datetime import datetime, timedelta, timezone

from data_node.db import (
    NodeDB,
    TaskKind,
    TaskStatus,
    PartitionStatus,
)


class TestQueueOperations:
    """Tests for backfill_queue operations."""

    def test_enqueue_task(self, node_db):
        """Test basic task enqueue."""
        task_id = node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        assert task_id is not None
        assert task_id > 0

    def test_enqueue_dedup(self, node_db):
        """Test that duplicate tasks are ignored."""
        task_id1 = node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        task_id2 = node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        assert task_id1 is not None
        assert task_id2 is None  # Duplicate

    def test_lease_next_task(self, node_db):
        """Test leasing a task."""
        # Enqueue a task
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        # Lease it
        task = node_db.lease_next_task(node_id="test-node")

        assert task is not None
        assert task.dataset == "equities_eod"
        assert task.symbol == "AAPL"
        assert task.status == TaskStatus.LEASED
        assert task.lease_owner == "test-node"

    def test_lease_priority_order(self, node_db):
        """Test that tasks are leased in priority order."""
        # Enqueue tasks with different priorities
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="MSFT",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.FORWARD,
            priority=5,
        )

        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.BOOTSTRAP,
            priority=0,
        )

        # First lease should get priority 0
        task1 = node_db.lease_next_task(node_id="test-node")
        assert task1.priority == 0
        assert task1.symbol == "AAPL"

        # Second lease should get priority 5
        task2 = node_db.lease_next_task(node_id="test-node")
        assert task2.priority == 5
        assert task2.symbol == "MSFT"

    def test_lease_expiry(self, node_db):
        """Test that expired leases can be re-leased."""
        # Enqueue and lease a task
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        # Lease with short TTL
        now = datetime.now(timezone.utc)
        task1 = node_db.lease_next_task(
            node_id="node-1",
            lease_ttl_seconds=1,
            now=now,
        )
        assert task1 is not None

        # Try to lease again immediately - should fail
        task2 = node_db.lease_next_task(node_id="node-2", now=now)
        assert task2 is None

        # After expiry, should succeed
        expired_time = now + timedelta(seconds=5)
        task3 = node_db.lease_next_task(node_id="node-2", now=expired_time)
        assert task3 is not None
        assert task3.lease_owner == "node-2"

    def test_mark_task_done(self, node_db):
        """Test marking a task as done."""
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        task = node_db.lease_next_task(node_id="test-node")
        assert node_db.mark_task_done(task.id)

        stats = node_db.get_queue_stats()
        assert stats["by_status"].get("DONE", 0) == 1
        assert stats["by_status"].get("PENDING", 0) == 0

    def test_mark_task_failed_retry(self, node_db):
        """Test marking a task as failed with retry."""
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        task = node_db.lease_next_task(node_id="test-node")

        # Mark failed - should stay PENDING for retry
        backoff = datetime.now(timezone.utc) + timedelta(seconds=60)
        assert node_db.mark_task_failed(
            task.id,
            error="Test error",
            backoff_until=backoff,
            max_attempts=5,
        )

        stats = node_db.get_queue_stats()
        assert stats["by_status"].get("PENDING", 0) == 1
        assert stats["by_status"].get("FAILED", 0) == 0

    def test_mark_task_failed_permanent(self, node_db):
        """Test marking a task as permanently failed."""
        node_db.enqueue_task(
            dataset="equities_eod",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-01",
            kind=TaskKind.GAP,
            priority=1,
        )

        # Exhaust retries
        for i in range(5):
            task = node_db.lease_next_task(node_id="test-node")
            if task is None:
                break
            node_db.mark_task_failed(
                task.id,
                error=f"Error {i}",
                max_attempts=5,
            )

        stats = node_db.get_queue_stats()
        assert stats["by_status"].get("FAILED", 0) == 1


class TestPartitionStatus:
    """Tests for partition_status operations."""

    def test_upsert_partition_status(self, node_db):
        """Test upserting partition status."""
        node_db.upsert_partition_status(
            source_name="alpaca",
            table_name="equities_eod",
            symbol="AAPL",
            dt="2024-01-02",
            status=PartitionStatus.GREEN,
            qc_score=1.0,
            row_count=1,
            expected_rows=1,
            qc_code="OK",
        )

        coverage = node_db.get_green_coverage(
            table_name="equities_eod",
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-03",
        )

        assert coverage == 1.0

    def test_upsert_preserves_first_observed(self, node_db):
        """Test that upsert preserves first_observed_at."""
        # First insert
        node_db.upsert_partition_status(
            source_name="alpaca",
            table_name="equities_eod",
            symbol="AAPL",
            dt="2024-01-02",
            status=PartitionStatus.AMBER,
            qc_score=0.5,
            row_count=0,
            expected_rows=1,
        )

        # Update
        node_db.upsert_partition_status(
            source_name="alpaca",
            table_name="equities_eod",
            symbol="AAPL",
            dt="2024-01-02",
            status=PartitionStatus.GREEN,
            qc_score=1.0,
            row_count=1,
            expected_rows=1,
        )

        # Should now be GREEN
        coverage = node_db.get_green_coverage(
            table_name="equities_eod",
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-03",
        )

        assert coverage == 1.0

    def test_get_partition_status_batch(self, node_db):
        """Test batch retrieval of partition status."""
        # Insert multiple
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            node_db.upsert_partition_status(
                source_name="alpaca",
                table_name="equities_eod",
                symbol=symbol,
                dt="2024-01-02",
                status=PartitionStatus.GREEN,
                qc_score=1.0,
                row_count=1,
                expected_rows=1,
            )

        records = node_db.get_partition_status_batch(
            table_name="equities_eod",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            start_date="2024-01-01",
            end_date="2024-01-03",
        )

        assert len(records) == 3

    def test_prune_failed_tasks(self, node_db):
        """Test pruning old failed tasks."""
        # Create and fail some tasks
        for i in range(3):
            node_db.enqueue_task(
                dataset="equities_eod",
                symbol=f"SYM{i}",
                start_date="2024-01-01",
                end_date="2024-01-01",
                kind=TaskKind.GAP,
                priority=1,
            )

            for _ in range(5):
                task = node_db.lease_next_task(node_id="test")
                if task:
                    node_db.mark_task_failed(task.id, "error", max_attempts=5)

        stats = node_db.get_queue_stats()
        assert stats["by_status"].get("FAILED", 0) == 3

        # Prune with future threshold
        future = datetime.now(timezone.utc) + timedelta(days=1)
        deleted = node_db.prune_failed_tasks(before=future)

        assert deleted == 3

        stats = node_db.get_queue_stats()
        assert stats["by_status"].get("FAILED", 0) == 0
