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
    summary = _summarize_backtest(
        result=result,
        output=output,
        initial_capital=float(args.initial_capital),
    )
    (output / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output / "backtest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary))
    return 0


def _summarize_backtest(*, result, output: Path, initial_capital: float) -> dict[str, object]:
    """Build a machine-readable summary for automation and operator review."""
    final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else float(initial_capital)
    total_cost = float(result.cost_attribution["cost"].sum()) if not result.cost_attribution.empty else 0.0
    turnover = (
        float(result.trade_log["trade_value"].abs().sum()) / float(initial_capital)
        if not result.trade_log.empty and initial_capital > 0
        else 0.0
    )
    net_return = (final_equity / float(initial_capital)) - 1.0 if initial_capital > 0 else 0.0
    gross_return = net_return + (total_cost / float(initial_capital) if initial_capital > 0 else 0.0)
    mean_rank_ic = float(result.ic_time_series["rank_ic"].mean()) if not result.ic_time_series.empty else 0.0
    return {
        "equity_rows": int(len(result.equity_curve)),
        "trade_rows": int(len(result.trade_log)),
        "cost_rows": int(len(result.cost_attribution)),
        "ic_rows": int(len(result.ic_time_series)),
        "decile_rows": int(len(result.decile_returns)),
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "gross_return": gross_return,
        "net_return": net_return,
        "cost_total": total_cost,
        "stressed_cost": total_cost,
        "turnover": turnover,
        "mean_rank_ic": mean_rank_ic,
        "paths": {
            "output_dir": str(output),
            "equity_curve": str(output / "equity_curve.parquet"),
            "trade_log": str(output / "trade_log.parquet"),
            "cost_attribution": str(output / "cost_attribution.parquet"),
            "ic_time_series": str(output / "ic_time_series.parquet"),
            "decile_returns": str(output / "decile_returns.parquet"),
            "summary": str(output / "backtest_summary.json"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
