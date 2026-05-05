from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.data_node.db import DataNodeDB
from trademl.data_node.saturation import LaneControlInput, SaturationControllerV2
from trademl.fleet.data_quality import run_data_quality_audit


def test_saturation_controller_scales_alpaca_above_one_lane() -> None:
    controller = SaturationControllerV2(target_utilization=0.98, vendor_caps={"alpaca": 16})

    decision = controller.decide(
        LaneControlInput(
            vendor="alpaca",
            dataset="equities_minute",
            base_width=1,
            eligible_tasks=500,
            active_width=0,
            rpm=200,
            remaining_minute=200,
            remaining_daily=100000,
            p95_latency_ms=5000,
            rows_per_credit=10000,
        ),
        resource_state={"status": "ok", "cpu_count": 4, "load_average_1m": 1.0},
    )

    assert decision.target_width > 1
    assert decision.target_width <= 16
    assert decision.action == "scale_up"


def test_saturation_controller_paces_low_rpm_and_backs_off_on_429() -> None:
    controller = SaturationControllerV2(target_utilization=0.98)

    paced = controller.decide(
        LaneControlInput(
            vendor="twelve_data",
            dataset="equities_minute",
            base_width=1,
            eligible_tasks=20,
            active_width=0,
            rpm=8,
            remaining_minute=8,
            remaining_daily=600,
            p95_latency_ms=8000,
        ),
        resource_state={"status": "ok", "cpu_count": 4, "load_average_1m": 1.0},
    )
    throttled = controller.decide(
        LaneControlInput(
            vendor="alpaca",
            dataset="stock_trades",
            base_width=1,
            eligible_tasks=20,
            active_width=0,
            rpm=200,
            remaining_minute=200,
            remaining_daily=100000,
            recent_429s=1,
        ),
        resource_state={"status": "ok", "cpu_count": 4, "load_average_1m": 1.0},
    )

    assert paced.target_width == 1
    assert paced.action == "paced"
    assert throttled.action == "backoff"
    assert throttled.reason == "recent_remote_rate_limit"


def test_controller_and_data_quality_sqlite_rollups(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "node.sqlite")

    db.record_controller_decision(
        vendor="alpaca",
        dataset="equities_minute",
        eligible_tasks=12,
        target_width=8,
        active_width=2,
        budget_remaining_minute=100,
        budget_remaining_daily=1000,
        latency_ms=2500,
        rows_per_credit=5000,
        action="scale_up",
        reason="budget_backlog_latency_allow_more_lanes",
    )
    db.record_data_quality_check(
        dataset="equities_minute",
        check_name="source_quality",
        verdict="OK",
        status="ok",
        rows_checked=10,
        partitions_checked=1,
    )

    controller = db.summarize_controller_decisions(minutes=60)
    quality = db.summarize_data_quality_checks(hours=24)

    assert controller["rows"][0]["action"] == "scale_up"
    assert controller["rows"][0]["target_width"] == 8
    assert quality["rows"][0]["dataset"] == "equities_minute"
    assert quality["rows"][0]["verdict"] == "OK"


def test_data_quality_detects_schema_mismatch_and_known_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    minute = data_root / "data" / "raw" / "equities_minute" / "date=2026-05-05"
    minute.mkdir(parents=True)
    pd.DataFrame([{"symbol": "AAPL", "open": 1.0}]).to_parquet(minute / "data.parquet", index=False)
    availability = data_root / "control" / "cluster" / "state" / "data" / "source_availability"
    availability.mkdir(parents=True)
    (availability / "latest.json").write_text(
        """
        {
          "datasets": {
            "fundamentals_tiingo": {
              "known_unavailable": true,
              "state": "ENTITLEMENT_UNAVAILABLE",
              "reason": "free plan"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    payload = run_data_quality_audit(
        data_root=data_root,
        datasets=["equities_minute", "fundamentals_tiingo"],
    )
    rows = {row["dataset"]: row for row in payload["rows"]}

    assert rows["equities_minute"]["verdict"] == "CRITICAL"
    assert rows["equities_minute"]["status"] == "schema_mismatch"
    assert rows["fundamentals_tiingo"]["verdict"] == "INFO"
    assert rows["fundamentals_tiingo"]["status"] == "known_unavailable"


def test_data_quality_checks_sibling_nas_alias_for_archive_sources(tmp_path: Path) -> None:
    data_root = tmp_path / "nas"
    alias_root = tmp_path / "trademl_nas"
    events = alias_root / "data" / "raw" / "alpaca_market_events" / "date=2026-05-05"
    events.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "source_name": "alpaca",
                "symbol": "AAPL",
                "vendor_ts": "2026-05-05T13:30:00Z",
                "price": 100.0,
                "bid_price": 99.9,
                "ask_price": 100.1,
            }
        ]
    ).to_parquet(events / "data.parquet", index=False)
    data_root.mkdir()

    payload = run_data_quality_audit(data_root=data_root, datasets=["stock_trades", "stock_quotes"])
    rows = {row["dataset"]: row for row in payload["rows"]}

    assert rows["stock_trades"]["verdict"] == "OK"
    assert rows["stock_quotes"]["verdict"] == "OK"
