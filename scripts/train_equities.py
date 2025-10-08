#!/usr/bin/env python3
"""CLI wrapper around the equities_xs pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from ops.pipelines.equities_xs import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equities cross-sectional training pipeline")
    parser.add_argument("--start", required=True, help="ISO start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="ISO end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", required=True, help="Universe symbols")
    parser.add_argument("--label", choices=["horizon", "triple_barrier"], default="horizon")
    parser.add_argument("--k", type=int, default=5, help="Horizon days for horizon labels")
    parser.add_argument("--tp", type=float, default=2.0, help="TP sigma for triple barrier")
    parser.add_argument("--sl", type=float, default=1.0, help="SL sigma for triple barrier")
    parser.add_argument("--max_h", type=int, default=10, help="Max holding for triple barrier")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)
    parser.add_argument("--gross-cap", type=float, default=1.0)
    parser.add_argument("--max-name", type=float, default=0.05)
    parser.add_argument("--kelly", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    args = parse_args()

    cfg = PipelineConfig(
        start_date=args.start,
        end_date=args.end,
        universe=args.symbols,
        label_type=args.label,
        horizon_days=args.k,
        tp_sigma=args.tp,
        sl_sigma=args.sl,
        max_h=args.max_h,
        n_folds=args.folds,
        embargo_days=args.embargo,
        initial_capital=args.capital,
        spread_bps=args.spread_bps,
        gross_cap=args.gross_cap,
        max_name=args.max_name,
        kelly_fraction=args.kelly,
    )
    result = run_pipeline(cfg)
    logger.info("Training complete; backtest Sharpe=%.3f", result["backtest_metrics"].sharpe_ratio)


if __name__ == "__main__":
    main()
