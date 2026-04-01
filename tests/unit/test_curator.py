from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.data_node.curator import Curator


def test_curator_adjusts_split_and_dividend() -> None:
    raw = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "ABC", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "vwap": 100.0, "volume": 10},
            {"date": "2024-01-03", "symbol": "ABC", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "vwap": 50.0, "volume": 20},
            {"date": "2024-01-04", "symbol": "ABC", "open": 49.0, "high": 50.0, "low": 48.0, "close": 49.0, "vwap": 49.0, "volume": 21},
        ]
    )
    corp_actions = pd.DataFrame(
        [
            {"symbol": "ABC", "event_type": "split", "ex_date": "2024-01-03", "ratio": 0.5, "source": "test"},
            {"symbol": "ABC", "event_type": "dividend", "ex_date": "2024-01-04", "ratio": 1.0, "source": "test"},
        ]
    )

    result = Curator().apply_adjustments(raw_bars=raw, corp_actions=corp_actions)
    adjusted = result.frame.sort_values("date").reset_index(drop=True)

    # Split halves pre-ex-date prices; dividend scales dates before 2024-01-04 by (50 - 1) / 50.
    assert adjusted.loc[0, "close"] == 49.0
    assert adjusted.loc[0, "volume"] == 20.0
    assert set(result.adjustment_log["event_type"]) == {"split", "dividend"}


def test_write_curated_can_limit_rebuild_scope_and_persist_adjustment_log(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "ABC", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "vwap": 10.0, "volume": 100},
            {"date": "2024-01-03", "symbol": "ABC", "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0, "vwap": 20.0, "volume": 200},
        ]
    )
    corp_actions = pd.DataFrame(
        [{"symbol": "ABC", "event_type": "split", "ex_date": "2024-01-03", "ratio": 0.5, "source": "test"}]
    )

    result = Curator().write_curated(
        raw_bars=raw,
        corp_actions=corp_actions,
        output_root=tmp_path / "curated",
        changed_dates=["2024-01-03"],
        adjustment_log_path=tmp_path / "adjustments.parquet",
    )

    assert (tmp_path / "curated" / "date=2024-01-03" / "data.parquet").exists()
    assert not (tmp_path / "curated" / "date=2024-01-02" / "data.parquet").exists()
    assert (tmp_path / "adjustments.parquet").exists()
    assert not result.adjustment_log.empty
