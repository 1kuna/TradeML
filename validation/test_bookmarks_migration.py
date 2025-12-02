import json
from datetime import datetime

from data_layer.storage.bookmarks import BookmarkManager


def test_legacy_string_bookmark_migrates(tmp_path):
    legacy = {"alpaca:equities_eod": "2025-11-30"}
    path = tmp_path / "bookmarks.json"
    path.write_text(json.dumps(legacy))

    mgr = BookmarkManager(s3_client=None, local_path=str(path))
    all_bm = mgr.get_all()

    assert "alpaca:equities_eod" in all_bm
    bm = all_bm["alpaca:equities_eod"]
    assert bm.last_timestamp == "2025-11-30"
    assert bm.source == "alpaca"
    assert bm.table == "equities_eod"

    # File should be rewritten to normalized schema
    data = json.loads(path.read_text())
    normalized = data["alpaca:equities_eod"]
    assert normalized["source"] == "alpaca"
    assert normalized["table"] == "equities_eod"
    assert "updated_at" in normalized
    datetime.fromisoformat(normalized["updated_at"])
