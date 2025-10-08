import os
import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd


class _DummyCalendar:
    def is_session(self, ts):
        return True

    def session_open(self, ts):
        return ts

    def session_close(self, ts):
        return ts

    def sessions_in_range(self, start, end):
        return [start, end]

    def previous_session(self, ts, n=1):
        return ts

    def next_session(self, ts, n=1):
        return ts


sys.modules.setdefault("exchange_calendars", types.SimpleNamespace(get_calendar=lambda *_args, **_kwargs: _DummyCalendar()))


def test_router_equities_gate(tmp_path):
    # Build a minimal ledger with GREEN for today on equities_eod for AAPL
    today = date.today()
    df = pd.DataFrame([
        {"source": "alpaca", "table_name": "equities_eod", "symbol": "AAPL", "dt": today, "status": "GREEN", "rows": 1, "expected_rows": 1, "qc_score": 1.0, "last_checked": pd.Timestamp.utcnow(), "notes": None}
    ])
    out = Path("data_layer/qc")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "partition_status.parquet", index=False)

    from ops.ssot.router import RouterContext, route
    ctx = RouterContext(df)
    sigs = route(today, "AAPL", ctx)
    assert "equities_xs" in sigs


def test_green_threshold_custom_cfg(tmp_path):
    # Use a custom training config that depends only on equities_eod for this test
    cfg = tmp_path / "equities_test.yml"
    cfg.write_text("dependencies: [equities_eod]\ngreen_threshold: { window_years: 1, min_ratio: 0.5 }\n")

    # Ledger with 60% GREEN over last year for equities_eod
    # For simplicity, just write one GREEN row and rely on the function reading by window
    today = date.today()
    df = pd.DataFrame([
        {"source": "alpaca", "table_name": "equities_eod", "symbol": "AAPL", "dt": today, "status": "GREEN", "rows": 1, "expected_rows": 1, "qc_score": 1.0, "last_checked": pd.Timestamp.utcnow(), "notes": None}
    ])
    out = Path("data_layer/qc")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "partition_status.parquet", index=False)

    from ops.ssot.train_gate import _meets_green_threshold
    ok = _meets_green_threshold(str(cfg), df)
    assert ok is True


def test_route_dataset_prefers_polygon():
    from ops.ssot.router import route_dataset

    order = route_dataset("equities_eod", date.today(), ["AAPL", "MSFT", "GOOGL"])
    assert order
    assert order[0] == "polygon"
