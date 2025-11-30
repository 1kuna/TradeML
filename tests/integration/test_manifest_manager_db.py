import os
from datetime import datetime

import pytest


class _StubCursor:
    def __init__(self, store):
        self.store = store
        self._fetch = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, sql, params=None):
        sql_lower = sql.lower()
        if "connector_bookmarks" in sql_lower and "insert into" in sql_lower:
            vendor, dataset, last_date, last_timestamp, row_count, status, updated_at, metadata = params
            key = (vendor, dataset)
            self.store["bookmarks"][key] = {
                "vendor": vendor,
                "dataset": dataset,
                "last_date": last_date,
                "last_timestamp": last_timestamp,
                "row_count": row_count,
                "status": status,
                "updated_at": updated_at,
                "metadata": metadata,
            }
        elif "connector_bookmarks" in sql_lower and "select" in sql_lower:
            if "where" in sql_lower:
                vendor, dataset = params
                key = (vendor, dataset)
                row = self.store["bookmarks"].get(key)
                self._fetch = [tuple(row.values())] if row else []
            else:
                self._fetch = [tuple(v.values()) for v in self.store["bookmarks"].values()]
        elif "connector_bookmarks" in sql_lower and "delete" in sql_lower:
            vendor, dataset = params
            key = (vendor, dataset)
            if key in self.store["bookmarks"]:
                del self.store["bookmarks"][key]
                self.rowcount = 1
            else:
                self.rowcount = 0
        elif "backfill_marks" in sql_lower and "insert into" in sql_lower:
            dataset, symbol, window_start, window_end, current_position, rows_backfilled, gaps_remaining, status, priority, created_at, updated_at, metadata = params
            key = (dataset, symbol)
            self.store["marks"][key] = {
                "dataset": dataset,
                "symbol": symbol,
                "window_start": window_start,
                "window_end": window_end,
                "current_position": current_position,
                "rows_backfilled": rows_backfilled,
                "gaps_remaining": gaps_remaining,
                "status": status,
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata,
            }
        elif "backfill_marks" in sql_lower and "select" in sql_lower:
            if "where" in sql_lower:
                dataset, symbol, symbol2 = params
                key = (dataset, symbol)
                row = self.store["marks"].get(key)
                self._fetch = [tuple(row.values())] if row else []
            else:
                self._fetch = [tuple(v.values()) for v in self.store["marks"].values()]
        elif "backfill_marks" in sql_lower and "delete" in sql_lower:
            dataset, symbol, symbol2 = params
            key = (dataset, symbol)
            if key in self.store["marks"]:
                del self.store["marks"][key]
                self.rowcount = 1
            else:
                self.rowcount = 0
        else:
            self._fetch = []

    def fetchone(self):
        return self._fetch[0] if self._fetch else None

    def fetchall(self):
        return self._fetch


class _StubConn:
    def __init__(self, store):
        self.store = store
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def cursor(self):
        return _StubCursor(self.store)

    def close(self):
        return None


@pytest.fixture
def stub_db(monkeypatch):
    from data_layer.manifests import manager as mgr

    store = {"bookmarks": {}, "marks": {}}

    def _fake_connect():
        return _StubConn(store)

    monkeypatch.setenv("MANIFESTS_BACKEND", "postgres")
    monkeypatch.setattr(mgr.ManifestManager, "_db_connect", staticmethod(_fake_connect))
    # Ensure table creation no-ops
    monkeypatch.setattr(mgr, "logger", mgr.logger)  # keep logger
    return store


def test_manifest_manager_db_roundtrip(stub_db):
    from data_layer.manifests.manager import ManifestManager

    mgr = ManifestManager()

    mgr.set_bookmark("alpaca", "equities_eod", "2024-01-01", row_count=10, status="success", metadata={"note": "ok"})
    bm = mgr.get_bookmark("alpaca", "equities_eod")
    assert bm is not None
    assert bm.last_date == "2024-01-01"
    assert bm.metadata == {"note": "ok"}

    mgr.set_backfill_mark("equities_eod", None, "2020-01-01", "2020-12-31", "2020-06-30", 100, 5, status="in_progress", priority=10)
    mark = mgr.get_backfill_mark("equities_eod")
    assert mark is not None
    assert mark.current_position == "2020-06-30"

    all_marks = mgr.get_all_backfill_marks()
    assert "equities_eod:None" not in all_marks  # keying uses dataset only when symbol None
    assert "equities_eod" in [k.split(":")[0] for k in all_marks.keys()]

    mgr.delete_bookmark("alpaca", "equities_eod")
    assert mgr.get_bookmark("alpaca", "equities_eod") is None

    mgr.delete_backfill_mark("equities_eod")
    assert mgr.get_backfill_mark("equities_eod") is None
