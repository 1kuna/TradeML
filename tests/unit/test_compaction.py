from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trademl.data_node.compaction import compact_archive_partitions


def test_compact_archive_partitions_preserves_rows_and_writes_telemetry(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    partition = data_root / "data" / "raw" / "ticker_news" / "date=2026-04-29"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-04-29", "2026-04-29"],
            "symbol": ["AAPL", "MSFT"],
            "news_id": ["n1", "n2"],
        }
    ).to_parquet(partition / "part-a.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2026-04-29", "2026-04-29"],
            "symbol": ["AAPL", "NVDA"],
            "news_id": ["n1", "n3"],
        }
    ).to_parquet(partition / "part-b.parquet", index=False)

    payload = compact_archive_partitions(data_root=data_root, datasets=["ticker_news"], max_partitions=1)

    compacted = pd.read_parquet(partition / "data.parquet")
    assert payload["summary"]["successes"] == 1
    assert payload["rows"][0]["duplicates_dropped"] == 1
    assert len(compacted) == 3
    assert not (partition / "part-a.parquet").exists()
    latest = data_root / "control" / "cluster" / "state" / "data" / "compaction" / "latest.json"
    assert json.loads(latest.read_text(encoding="utf-8"))["rows"][0]["dataset"] == "ticker_news"


def test_compact_archive_partitions_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    partition = data_root / "data" / "raw" / "equities_minute" / "date=2026-04-29"
    partition.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-04-29"], "symbol": ["AAPL"], "close": [1.0]}).to_parquet(
        partition / "part-a.parquet",
        index=False,
    )
    pd.DataFrame({"date": ["2026-04-29"], "symbol": ["AAPL"], "close": [1.0]}).to_parquet(
        partition / "part-b.parquet",
        index=False,
    )

    payload = compact_archive_partitions(data_root=data_root, datasets=["equities_minute"], dry_run=True)

    assert payload["rows"][0]["status"] == "dry_run"
    assert (partition / "part-a.parquet").exists()
    assert (partition / "part-b.parquet").exists()
    assert not (partition / "data.parquet").exists()
