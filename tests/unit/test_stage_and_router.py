from __future__ import annotations

from datetime import date, timedelta

from data_node.db import PartitionStatus
from data_node.stages import (
    StageConfig,
    StageDefinition,
    check_promotion,
    get_current_stage,
    save_stage_config,
)
from ops.ssot.router import route_dataset


def test_stage_promotion_seeds_next_stage(node_db, temp_data_root):
    stages = {
        0: StageDefinition(
            name="s0-small",
            universe_size=2,
            equities_eod_years=1,
            equities_minute_years=1,
            green_threshold=0.5,
        ),
        1: StageDefinition(
            name="s1-small",
            universe_size=3,
            equities_eod_years=2,
            equities_minute_years=2,
            green_threshold=0.5,
        ),
    }
    save_stage_config(StageConfig(current_stage=0, promoted_at=None, stages=stages))

    today = date.today()
    for sym in ["NVDA", "MSFT"]:
        for dt in (today - timedelta(days=1), today):
            node_db.upsert_partition_status(
                source_name="alpaca",
                table_name="equities_eod",
                symbol=sym,
                dt=dt.isoformat(),
                status=PartitionStatus.GREEN,
                qc_score=1.0,
                row_count=100,
                expected_rows=100,
            )
            node_db.upsert_partition_status(
                source_name="alpaca",
                table_name="equities_minute",
                symbol=sym,
                dt=dt.isoformat(),
                status=PartitionStatus.GREEN,
                qc_score=1.0,
                row_count=100,
                expected_rows=100,
            )

    promoted = check_promotion(node_db)
    assert promoted is True
    assert get_current_stage() == 1
    assert node_db.get_queue_stats()["by_status"].get("PENDING", 0) > 0


def test_route_dataset_prefers_unique_weighted_sources(temp_data_root):
    cfg_path = temp_data_root / "endpoints.yml"
    cfg_path.write_text(
        """
providers:
  alpaca:
    weight: 1.0
    rpm: 200
    datasets:
      equities_eod:
        unique: true
        weight: 2
  finnhub:
    weight: 1.0
    rpm: 60
    datasets:
      equities_eod:
        unique: false
        weight: 1
"""
    )

    ordered = route_dataset(
        "equities_eod",
        want_date=date.today(),
        universe=["AAPL", "MSFT"],
        endpoints_path=str(cfg_path),
    )

    assert ordered[0] == "alpaca"
    assert set(ordered) == {"alpaca", "finnhub"}
