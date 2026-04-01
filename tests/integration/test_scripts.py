from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _write_training_dataset(root: Path) -> None:
    dates = pd.bdate_range("2024-01-01", periods=520)
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    raw_rows = []
    for idx, symbol in enumerate(symbols):
        close = 50 + idx * 10 + np.linspace(0, 30, len(dates)) + np.sin(np.arange(len(dates)) / 10)
        open_ = close * 0.999
        for date, open_price, close_price in zip(dates, open_, close, strict=False):
            raw_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": float(open_price),
                    "high": float(close_price * 1.01),
                    "low": float(close_price * 0.99),
                    "close": float(close_price),
                    "vwap": float((open_price + close_price) / 2),
                    "volume": 1_000_000 + idx * 10_000,
                }
            )
    panel = pd.DataFrame(raw_rows)
    curated_root = root / "data" / "curated" / "equities_ohlcv_adj"
    curated_root.mkdir(parents=True, exist_ok=True)
    for day, day_frame in panel.groupby("date"):
        partition = curated_root / f"date={day.strftime('%Y-%m-%d')}"
        partition.mkdir(parents=True, exist_ok=True)
        day_frame.to_parquet(partition / "data.parquet", index=False)

    qc_root = root / "data" / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame(
        {
            "source": "alpaca",
            "dataset": "equities_eod",
            "date": [date.strftime("%Y-%m-%d") for date in dates],
            "status": "GREEN",
            "row_count": len(symbols),
            "expected_rows": len(symbols),
            "qc_code": "OK",
            "note": None,
            "updated_at": pd.Timestamp.utcnow(),
        }
    )
    qc.to_parquet(qc_root / "partition_status.parquet", index=False)


def test_train_script_emits_report(tmp_path: Path) -> None:
    data_root = tmp_path / "workspace"
    _write_training_dataset(data_root)
    config = {
        "data": {"green_threshold": 0.9},
        "features": {
            "price": {"momentum": [5, 20, 60, 126], "reversal": [1, 5], "drawdown": [20, 60]},
            "volatility": {"realized": [20, 60], "idiosyncratic": [60]},
            "liquidity": {"adv_dollar": [20], "amihud": [20]},
            "controls": {"log_price": True},
        },
        "preprocessing": {"missing_threshold": 0.30},
        "validation": {"initial_train_years": 1, "step": "6_months"},
        "portfolio": {"cost_stress_multiplier": 2.0},
    }
    config_path = tmp_path / "train.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "src/scripts/train.py",
            "--data-root",
            str(data_root),
            "--config",
            str(config_path),
            "--output-root",
            str(data_root),
            "--report-date",
            "2026-03-31",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    assert (data_root / "reports" / "daily" / "2026-03-31.json").exists()
    assert (data_root / "reports" / "daily" / "2026-03-31.md").exists()


def test_backtest_script_writes_outputs(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=4).tolist() * 2,
            "symbol": ["AAPL"] * 4 + ["MSFT"] * 4,
            "close": [100, 101, 102, 103, 50, 51, 52, 53],
        }
    )
    targets = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")],
            "symbol": ["AAPL", "MSFT"],
            "score": [1.0, 1.0],
            "target_weight": [1.0, 1.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
            "symbol": ["AAPL", "MSFT"],
            "prediction": [0.7, 0.2],
            "label_5d": [0.03, -0.01],
        }
    )
    prices_path = tmp_path / "prices.parquet"
    targets_path = tmp_path / "targets.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "backtest"
    prices.to_parquet(prices_path, index=False)
    targets.to_parquet(targets_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "src/scripts/backtest.py",
            "--prices",
            str(prices_path),
            "--targets",
            str(targets_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=Path.cwd(),
    )

    assert (output_dir / "equity_curve.parquet").exists()
    assert (output_dir / "trade_log.parquet").exists()
    assert (output_dir / "cost_attribution.parquet").exists()
    assert (output_dir / "ic_time_series.parquet").exists()
    assert (output_dir / "decile_returns.parquet").exists()


def test_pi_wizard_initializes_state(tmp_path: Path) -> None:
    root = tmp_path / "pi"
    env_file = tmp_path / "wizard.env"
    fstab_path = tmp_path / "fstab"
    config_path = tmp_path / "node.yml"
    stage_symbols = [f"SYM{index:03d}" for index in range(100)]
    result = subprocess.run(
        [
            sys.executable,
            "src/scripts/pi_data_node_wizard.py",
            "--root",
            str(root),
            "--config",
            str(config_path),
            "--stage-years",
            "5",
            "--nas-mount",
            str(tmp_path / "nas"),
            "--collection-time-et",
            "17:00",
            "--maintenance-hour-local",
            "3",
            "--env-file",
            str(env_file),
            "--fstab-path",
            str(fstab_path),
            *[item for symbol in stage_symbols for item in ("--stage-symbol", symbol)],
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["task_count"] == 100
    assert (root / "control" / "node.sqlite").exists()
    assert (root / "stage.yml").exists()
    stage = yaml.safe_load((root / "stage.yml").read_text(encoding="utf-8"))
    assert stage["symbols"] == stage_symbols
    assert stage["schedule"]["collection_time_et"] == "17:00"
    assert stage["schedule"]["maintenance_hour_local"] == 3
    assert env_file.exists()
    assert "NAS_MOUNT=" in env_file.read_text(encoding="utf-8")
    assert "NAS_SHARE=" in env_file.read_text(encoding="utf-8")
    assert fstab_path.exists()
    assert str(tmp_path / "nas") in fstab_path.read_text(encoding="utf-8")
    node_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert node_cfg["node"]["nas_share"] == "//nas/trademl"
    assert node_cfg["node"]["collection_time_et"] == "17:00"
    assert node_cfg["node"]["maintenance_hour_local"] == 3
