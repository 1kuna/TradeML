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
    parser.add_argument("--corp-actions", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    targets = pd.read_parquet(args.targets)
    corp_actions = pd.read_parquet(args.corp_actions) if args.corp_actions else None
    predictions = pd.read_parquet(args.predictions) if args.predictions else None
    result = run_backtest(
        prices,
        targets,
        apply_costs,
        {"initial_capital": args.initial_capital, "cost_spread_bps": 5.0},
        corp_actions=corp_actions,
        prediction_frame=predictions,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_parquet(output / "equity_curve.parquet", index=False)
    result.trade_log.to_parquet(output / "trade_log.parquet", index=False)
    result.cost_attribution.to_parquet(output / "cost_attribution.parquet", index=False)
    result.ic_time_series.to_parquet(output / "ic_time_series.parquet", index=False)
    result.decile_returns.to_parquet(output / "decile_returns.parquet", index=False)
    print(
        json.dumps(
            {
                "equity_rows": len(result.equity_curve),
                "trade_rows": len(result.trade_log),
                "cost_rows": len(result.cost_attribution),
                "ic_rows": len(result.ic_time_series),
                "decile_rows": len(result.decile_returns),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
