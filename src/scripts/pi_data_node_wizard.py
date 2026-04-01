"""Pi data-node setup wizard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from trademl.data_node.db import DataNodeDB


STAGE0_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK.B", "LLY", "JPM", "XOM",
    "UNH", "V", "MA", "AVGO", "HD", "COST", "PG", "JNJ", "ORCL", "NFLX",
    "ABBV", "BAC", "KO", "MRK", "CVX", "PEP", "TMO", "WMT", "ADBE", "CSCO",
    "AMD", "CRM", "MCD", "LIN", "ACN", "QCOM", "DHR", "TXN", "ABT", "PM",
    "WFC", "IBM", "GE", "NOW", "GS", "INTU", "MS", "AMAT", "ISRG", "CAT",
    "DIS", "BLK", "RTX", "SPGI", "BKNG", "SCHW", "T", "PGR", "C", "AMGN",
    "COP", "HON", "LOW", "ELV", "MDT", "VRTX", "PANW", "INTC", "BA", "GILD",
    "DE", "ADI", "LRCX", "SYK", "MMC", "PLD", "CB", "TMUS", "NKE", "MU",
    "SO", "CI", "UPS", "MDLZ", "REGN", "AXP", "PYPL", "FI", "KLAC", "ICE",
    "SHW", "DUK", "TT", "SNPS", "USB", "ZTS", "AON", "CSX", "MO", "EQIX",
]


def run_wizard(*, root: Path, config_path: Path, stage_years: int = 5) -> dict:
    """Initialize Pi-local state and seed Stage 0 bootstrap tasks."""
    local_state = root / "control"
    local_state.mkdir(parents=True, exist_ok=True)
    db = DataNodeDB(local_state / "node.sqlite")
    bookmarks = root / "bookmarks.json"
    stage_file = root / "stage.yml"
    bookmarks.write_text(json.dumps({"stage": 0, "symbols_seeded": len(STAGE0_SYMBOLS)}, indent=2), encoding="utf-8")
    stage_file.write_text(yaml.safe_dump({"current": 0, "symbols": STAGE0_SYMBOLS, "years": stage_years}), encoding="utf-8")

    end_date = "2026-03-31"
    start_date = f"{int(end_date[:4]) - stage_years}-{end_date[5:]}"
    for symbol in STAGE0_SYMBOLS:
        try:
            db.enqueue_task("equities_eod", symbol, start_date, end_date, "BOOTSTRAP", 5)
        except Exception:
            continue

    return {"local_state": str(local_state), "task_count": len(STAGE0_SYMBOLS), "config_path": str(config_path)}


def main() -> int:
    """CLI entry point for initializing Pi node state."""
    parser = argparse.ArgumentParser(description="Initialize TradeML Pi node state.")
    parser.add_argument("--root", default="~/trademl")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--stage-years", type=int, default=5)
    args = parser.parse_args()

    result = run_wizard(root=Path(args.root).expanduser(), config_path=Path(args.config), stage_years=args.stage_years)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
