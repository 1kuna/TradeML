"""Backtest runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trademl.backtest.engine import run_backtest
from trademl.costs.models import apply_costs


def main() -> int:
    """Run a deterministic backtest from parquet inputs."""
    parser = argparse.ArgumentParser(description="Run the TradeML backtest.")
    parser.add_argument("--prices", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    targets = pd.read_parquet(args.targets)
    result = run_backtest(prices, targets, apply_costs, {"initial_capital": args.initial_capital, "cost_spread_bps": 5.0})

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_parquet(output / "equity_curve.parquet", index=False)
    result.trade_log.to_parquet(output / "trade_log.parquet", index=False)
    print(json.dumps({"equity_rows": len(result.equity_curve), "trade_rows": len(result.trade_log)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
