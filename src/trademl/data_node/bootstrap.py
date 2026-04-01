"""Shared Stage 0 bootstrap defaults for worker setup flows."""

from __future__ import annotations

from typing import Any


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


def resolve_bootstrap_stage(local_config: dict[str, Any], local_stage: dict[str, Any]) -> tuple[int, list[str], int]:
    """Resolve a usable bootstrap stage from local stage.yml or config defaults."""
    current = int(local_stage.get("current", local_config.get("stage", {}).get("current", 0)))
    stage_key = f"stage_{current}"
    stage_cfg = local_config.get("stage", {}).get(stage_key, {})

    stage_symbols = list(local_stage.get("symbols", []))
    if not stage_symbols:
        configured_symbols = stage_cfg.get("symbols", [])
        if isinstance(configured_symbols, list):
            stage_symbols = [str(symbol) for symbol in configured_symbols if str(symbol).strip()]
        else:
            count = int(configured_symbols or len(STAGE0_SYMBOLS))
            stage_symbols = STAGE0_SYMBOLS[: max(1, min(count, len(STAGE0_SYMBOLS)))]

    years = int(local_stage.get("years", stage_cfg.get("eod_years", 5)))
    return current, stage_symbols, years
