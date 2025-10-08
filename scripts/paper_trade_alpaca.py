#!/usr/bin/env python3
"""Submit target weights to Alpaca for paper trading."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from data_layer.curated.loaders import load_price_panel
from execution.brokers.alpaca_client import AlpacaBrokerClient, AlpacaCredentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send target weights to Alpaca")
    parser.add_argument("--report", type=str, help="Path to equities_<date>.json report")
    parser.add_argument("--asof", type=str, help="Override as-of date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--fractional", action="store_true", help="Allow fractional share trading")
    return parser.parse_args()


def _find_latest_report(asof: str | None) -> Path:
    reports_dir = Path("ops/reports")
    if asof:
        candidate = reports_dir / f"equities_{asof}.json"
        if candidate.exists():
            return candidate
    files = sorted(reports_dir.glob("equities_*.json"))
    if not files:
        raise FileNotFoundError("No equities reports found in ops/reports")
    return files[-1]


def _load_positions(report_path: Path) -> tuple[pd.DataFrame, str]:
    payload = json.loads(report_path.read_text())
    positions = pd.DataFrame(payload.get("positions", []))
    if positions.empty:
        raise RuntimeError(f"No positions found in {report_path}")
    if "target_w" not in positions.columns:
        raise ValueError("Report missing 'target_w' column")
    return positions, payload.get("asof")


def _price_map(asof: str, universe: pd.Series) -> Dict[str, float]:
    panel = load_price_panel(universe.tolist(), asof, asof)
    panel = panel[panel["date"] == pd.to_datetime(asof).date()][["symbol", "close"]]
    return dict(zip(panel["symbol"], panel["close"]))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    args = parse_args()
    report_path = Path(args.report) if args.report else _find_latest_report(args.asof)
    positions, report_asof = _load_positions(report_path)
    asof = args.asof or report_asof
    if not asof:
        raise ValueError("As-of date could not be determined; pass --asof")

    price_map = _price_map(asof, positions["symbol"])
    creds = AlpacaCredentials(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        api_secret=os.getenv("ALPACA_API_SECRET", ""),
        paper=os.getenv("ALPACA_PAPER", "true").lower() != "false",
    )
    if not creds.api_key or not creds.api_secret:
        raise ValueError("Alpaca credentials missing; set ALPACA_API_KEY and ALPACA_API_SECRET")

    client = AlpacaBrokerClient(creds)
    weights = positions[["symbol", "target_w"]].copy()
    weights["date"] = pd.to_datetime(asof).date()
    orders = client.submit_orders(
        asof=pd.to_datetime(asof),
        target_weights=weights,
        policy_cfg={
            "notional": args.capital,
            "price_map": price_map,
            "fractional": args.fractional,
        },
    )
    logger.info("Submitted %d orders via Alpaca", len(orders))


if __name__ == "__main__":
    main()
