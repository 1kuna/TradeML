from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from trademl.data_node import db as db_module
from trademl.data_node.db import DataNodeDB


def test_enqueue_lease_done_cycle(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    task_id = database.enqueue_task("equities_eod", "AAPL", "2024-01-01", "2024-01-31", "GAP", 1)

    leased = database.lease_next_task()

    assert leased is not None
    assert leased.id == task_id
    assert leased.status == "LEASED"

    database.mark_task_done(task_id)
    assert database.lease_next_task() is None


def test_failed_task_respects_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    task_id = database.enqueue_task("equities_eod", "MSFT", "2024-02-01", "2024-02-29", "GAP", 1)
    first_lease = database.lease_next_task()
    assert first_lease is not None

    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)
    database.mark_task_failed(task_id, "429", backoff_minutes=30)

    assert database.lease_next_task(now=now + timedelta(minutes=29)) is None
    leased_again = database.lease_next_task(now=now + timedelta(minutes=31))
    assert leased_again is not None
    assert leased_again.id == task_id


def test_duplicate_tasks_rejected(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", "NVDA", "2024-01-01", "2024-01-05", "BOOTSTRAP", 5)

    with pytest.raises(sqlite3.IntegrityError):
        database.enqueue_task("equities_eod", "NVDA", "2024-01-01", "2024-01-05", "BOOTSTRAP", 5)


def test_duplicate_gap_tasks_with_null_symbol_are_rejected(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", None, "2024-01-01", "2024-01-01", "GAP", 1)

    with pytest.raises(sqlite3.IntegrityError):
        database.enqueue_task("equities_eod", None, "2024-01-01", "2024-01-01", "GAP", 1)


def test_priority_ordering(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", "LOW", "2024-01-01", "2024-01-05", "GAP", 5)
    database.enqueue_task("equities_eod", "MID", "2024-01-01", "2024-01-05", "GAP", 1)
    database.enqueue_task("equities_eod", "HIGH", "2024-01-01", "2024-01-05", "GAP", 0)

    leased = [database.lease_next_task(), database.lease_next_task(), database.lease_next_task()]

    assert [task.symbol for task in leased if task is not None] == ["HIGH", "MID", "LOW"]
