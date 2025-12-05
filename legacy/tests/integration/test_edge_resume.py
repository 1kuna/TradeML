import os
import io
import uuid
from datetime import datetime, timedelta
import pytest
import pandas as pd

pytestmark = pytest.mark.skipif(
    not os.getenv("S3_ENDPOINT"),
    reason="Requires S3/MinIO endpoint in environment",
)


class StubConnector:
    def __init__(self):
        pass

    def fetch_and_transform(self, symbols, start, end):
        # Produce a small dataframe for the requested day
        day = pd.to_datetime(start).date()
        rows = [
            {"date": day, "symbol": s, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for s in symbols
        ]
        return pd.DataFrame(rows)


def test_edge_resumes_from_bookmark(monkeypatch, tmp_path):
    from data_layer.storage.s3_client import get_s3_client
    from data_layer.storage.bookmarks import BookmarkManager
    from scripts.edge_collector import EdgeCollector

    os.environ["STORAGE_BACKEND"] = "s3"

    s3 = get_s3_client()

    # Unique bookmark key per test
    bm_key = f"manifests/test_bookmarks_{uuid.uuid4().hex}.json"
    bm = BookmarkManager(s3, bookmark_key=bm_key)

    source = "alpaca"
    table = "equities_bars"

    # Seed bookmark to yesterday
    yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
    assert bm.set(source, table, yesterday, row_count=10)

    # Create minimal config file
    cfg_path = tmp_path / "edge.yml"
    cfg_path.write_text("locks:\n  lease_seconds: 5\n  renew_seconds: 2\n  tasks:\n    - alpaca_bars\n")

    # Init collector and stub the connector
    ec = EdgeCollector(str(cfg_path))
    ec.bookmarks = bm
    ec.connectors = {"alpaca": StubConnector()}

    # Run one pass (should start from today, since yesterday is bookmarked)
    ec.collect_alpaca_bars()

    # Expect bookmark advanced to today or later
    last = bm.get_last_timestamp(source, table)
    assert last >= datetime.utcnow().date().isoformat()

