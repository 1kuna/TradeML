import os
import json
from datetime import datetime, timedelta, timezone

import pytest


def test_budget_manager_local_persist_and_reset(tmp_path, monkeypatch):
    os.environ.pop("STORAGE_BACKEND", None)
    # Point manifest to temp dir
    import ops.ssot.budget as budget_mod

    manifest_rel = f"manifests/test_budget_{datetime.now().timestamp()}.json"
    bm = budget_mod.BudgetManager({"alpaca": 5, "finnhub": 2}, s3_client=None, manifest_key=manifest_rel)

    assert bm.remaining("alpaca") == 5
    assert bm.try_consume("alpaca", 3) is True
    assert bm.remaining("alpaca") == 2
    assert bm.try_consume("alpaca", 3) is False
    assert bm.remaining("finnhub") == 2

    # Force reset by setting reset_at in the past
    bm.state["alpaca"].reset_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rem_before = bm.remaining("alpaca")
    assert rem_before == 5  # reset to full limit
