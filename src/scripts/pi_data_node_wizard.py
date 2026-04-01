"""Pi data-node setup wizard."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
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
    stage_file.write_text(
        yaml.safe_dump(
            {
                "current": 0,
                "symbols": STAGE0_SYMBOLS,
                "years": stage_years,
                "environment": _detect_environment(),
                "schedule": {"collection_time_et": "16:30", "maintenance_hour_local": 2},
            }
        ),
        encoding="utf-8",
    )

    end_date = datetime.now(UTC).date().isoformat()
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
    parser.add_argument("--nas-mount", default=None)
    parser.add_argument("--collection-time-et", default="16:30")
    parser.add_argument("--maintenance-hour-local", type=int, default=2)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--alpaca-api-key", default="")
    parser.add_argument("--alpaca-api-secret", default="")
    parser.add_argument("--finnhub-api-key", default="")
    parser.add_argument("--alpha-vantage-api-key", default="")
    parser.add_argument("--fred-api-key", default="")
    parser.add_argument("--fmp-api-key", default="")
    parser.add_argument("--massive-api-key", default="")
    parser.add_argument("--start-node", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    env_values = {
        "NAS_MOUNT": args.nas_mount or _prompt("NAS mount", "/mnt/trademl"),
        "COLLECTION_TIME_ET": args.collection_time_et,
        "MAINTENANCE_HOUR_LOCAL": str(args.maintenance_hour_local),
        "ALPACA_API_KEY": args.alpaca_api_key or _prompt("Alpaca API key", os.getenv("ALPACA_API_KEY", "")),
        "ALPACA_API_SECRET": args.alpaca_api_secret or _prompt("Alpaca API secret", os.getenv("ALPACA_API_SECRET", "")),
        "FINNHUB_API_KEY": args.finnhub_api_key or _prompt("Finnhub API key", os.getenv("FINNHUB_API_KEY", "")),
        "ALPHA_VANTAGE_API_KEY": args.alpha_vantage_api_key or _prompt("Alpha Vantage API key", os.getenv("ALPHA_VANTAGE_API_KEY", "")),
        "FRED_API_KEY": args.fred_api_key or _prompt("FRED API key", os.getenv("FRED_API_KEY", "")),
        "FMP_API_KEY": args.fmp_api_key or _prompt("FMP API key", os.getenv("FMP_API_KEY", "")),
        "MASSIVE_API_KEY": args.massive_api_key or _prompt("Massive API key", os.getenv("MASSIVE_API_KEY", "")),
    }
    nas_write_ok = _test_nas_mount(Path(env_values["NAS_MOUNT"]))
    env_path = Path(args.env_file).expanduser() if args.env_file else root / ".env"
    _write_env_file(env_path, env_values)
    result = run_wizard(root=root, config_path=Path(args.config), stage_years=args.stage_years)
    result.update({"nas_write_ok": nas_write_ok, "env_file": str(env_path)})
    if args.start_node:
        subprocess.Popen(
            [os.sys.executable, "-m", "trademl.data_node", "--config", args.config, "--root", str(root), "--date", end_date := datetime.now(UTC).date().isoformat(), "--symbols", "AAPL", "MSFT"],  # noqa: S603
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result["node_started"] = True
    print(json.dumps(result))
    return 0

def _prompt(label: str, default: str) -> str:
    if not os.isatty(0):
        return default
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _detect_environment() -> dict[str, object]:
    memory_mb = 0
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        memory_mb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    return {"machine": platform.machine(), "platform": platform.platform(), "memory_mb": memory_mb}


def _test_nas_mount(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
