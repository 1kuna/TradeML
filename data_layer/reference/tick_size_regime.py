"""
Tick size regime reference data manager.

Provides PIT-safe lookups for tick size regimes (Reg NMS, Tick Pilot, etc.)
to enable microstructure feature computation and execution modeling.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
from loguru import logger

from ..schemas import TICK_SIZE_REGIME_SCHEMA


class TickSizeRegime:
    """
    Manage tick size regime reference data with PIT discipline.

    Key principles:
    1. Track regime changes with effective dates
    2. Support symbol-specific and market-wide rules
    3. PIT-safe lookups (only use rules effective at query date)
    """

    DEFAULT_PATH = Path("data_layer/reference/tick_size_regime/data.parquet")

    # Standard tick sizes
    REG_NMS_TICK = 0.01        # $0.01 for prices >= $1.00
    SUB_PENNY_TICK = 0.0001    # $0.0001 for prices < $1.00
    TICK_PILOT_TICK = 0.05     # $0.05 for Tick Pilot symbols

    # Regime names
    REG_NMS = "REG_NMS"
    SUB_PENNY = "SUB_PENNY"
    TICK_PILOT = "TICK_PILOT"
    STANDARD = "STANDARD"

    # Key dates
    REG_NMS_EFFECTIVE = date(2007, 10, 1)
    TICK_PILOT_START = date(2016, 10, 3)
    TICK_PILOT_END = date(2018, 10, 2)

    def __init__(self, path: Optional[Path] = None):
        """
        Initialize tick size regime manager.

        Args:
            path: Path to tick size regime parquet file
        """
        self.path = path or self.DEFAULT_PATH
        self._cache: Optional[pd.DataFrame] = None
        self._tick_pilot_symbols: Optional[set] = None
        logger.debug(f"TickSizeRegime initialized with path: {self.path}")

    def _load(self) -> pd.DataFrame:
        """Load tick size regime data, using cache if available."""
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            # Return default market-wide rules
            logger.debug(f"Tick size regime file not found: {self.path}, using defaults")
            return self._get_default_rules()

        df = pd.read_parquet(self.path)

        # Ensure date type
        if "effective_date" in df.columns and not df.empty:
            df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date

        self._cache = df
        logger.info(f"Loaded {len(df)} tick size regime entries")
        return df

    def _get_default_rules(self) -> pd.DataFrame:
        """Get default market-wide tick size rules."""
        rules = [
            {
                "effective_date": self.REG_NMS_EFFECTIVE,
                "symbol": None,
                "tick_size": self.REG_NMS_TICK,
                "regime_name": self.REG_NMS,
                "notes": "Reg NMS minimum tick for prices >= $1.00",
            },
        ]
        return pd.DataFrame(rules)

    def invalidate_cache(self) -> None:
        """Clear the cache to force reload on next access."""
        self._cache = None
        self._tick_pilot_symbols = None

    def get_tick_size(
        self,
        symbol: str,
        as_of_date: date,
        price: Optional[float] = None,
    ) -> float:
        """
        Get the tick size for a symbol at a specific date.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to query (PIT)
            price: Optional price for sub-penny determination

        Returns:
            Tick size in dollars
        """
        df = self._load()

        # Check for symbol-specific rule first
        symbol_rules = df[
            (df["symbol"] == symbol) &
            (df["effective_date"] <= as_of_date)
        ].sort_values("effective_date", ascending=False)

        if not symbol_rules.empty:
            return float(symbol_rules.iloc[0]["tick_size"])

        # Fall back to market-wide rule
        market_rules = df[
            (df["symbol"].isna()) &
            (df["effective_date"] <= as_of_date)
        ].sort_values("effective_date", ascending=False)

        if not market_rules.empty:
            base_tick = float(market_rules.iloc[0]["tick_size"])
        else:
            base_tick = self.REG_NMS_TICK

        # Apply sub-penny rule for prices < $1.00
        if price is not None and price < 1.00:
            return self.SUB_PENNY_TICK

        return base_tick

    def get_regime_name(self, symbol: str, as_of_date: date) -> str:
        """
        Get the regime name for a symbol at a specific date.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to query (PIT)

        Returns:
            Regime name (e.g., "REG_NMS", "TICK_PILOT")
        """
        df = self._load()

        # Check for symbol-specific rule first
        symbol_rules = df[
            (df["symbol"] == symbol) &
            (df["effective_date"] <= as_of_date)
        ].sort_values("effective_date", ascending=False)

        if not symbol_rules.empty:
            return str(symbol_rules.iloc[0]["regime_name"])

        # Fall back to market-wide
        return self.REG_NMS

    def is_tick_constrained(
        self,
        symbol: str,
        as_of_date: date,
        price: float,
        threshold_pct: float = 0.5,
    ) -> bool:
        """
        Check if a price is near a tick boundary.

        Useful for identifying when spread-based execution costs
        are constrained by tick size.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to query
            price: Current price
            threshold_pct: Percentage of tick size to consider "near"

        Returns:
            True if price is within threshold_pct of a tick boundary
        """
        tick = self.get_tick_size(symbol, as_of_date, price)

        # Distance to nearest tick boundary
        remainder = price % tick
        dist_to_lower = remainder
        dist_to_upper = tick - remainder

        min_dist = min(dist_to_lower, dist_to_upper)
        threshold = tick * threshold_pct

        return min_dist <= threshold

    def was_tick_pilot_symbol(self, symbol: str, as_of_date: date) -> bool:
        """
        Check if a symbol was in the Tick Pilot program on a specific date.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to query (PIT)

        Returns:
            True if symbol was in Tick Pilot and date is within pilot period
        """
        if not (self.TICK_PILOT_START <= as_of_date <= self.TICK_PILOT_END):
            return False

        df = self._load()

        pilot_rules = df[
            (df["symbol"] == symbol) &
            (df["regime_name"] == self.TICK_PILOT) &
            (df["effective_date"] <= as_of_date)
        ]

        return not pilot_rules.empty

    def get_tick_pilot_symbols(self) -> set:
        """
        Get all symbols that were in the Tick Pilot program.

        Returns:
            Set of symbols
        """
        if self._tick_pilot_symbols is not None:
            return self._tick_pilot_symbols

        df = self._load()
        pilot_df = df[df["regime_name"] == self.TICK_PILOT]

        self._tick_pilot_symbols = set(pilot_df["symbol"].dropna().unique())
        return self._tick_pilot_symbols

    def add_rule(
        self,
        effective_date: date,
        tick_size: float,
        regime_name: str,
        symbol: Optional[str] = None,
        notes: Optional[str] = None,
        source_name: str = "manual",
    ) -> None:
        """
        Add a tick size rule.

        Args:
            effective_date: When the rule becomes effective
            tick_size: Tick size in dollars
            regime_name: Regime identifier
            symbol: Symbol (None for market-wide)
            notes: Optional notes
            source_name: Source of the data
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            df = pd.read_parquet(self.path)
        else:
            df = pd.DataFrame()

        new_row = {
            "effective_date": effective_date,
            "symbol": symbol,
            "tick_size": tick_size,
            "regime_name": regime_name,
            "notes": notes,
            "source_name": source_name,
            "ingested_at": datetime.now(timezone.utc),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        df = df.sort_values("effective_date").reset_index(drop=True)
        df.to_parquet(self.path, index=False)

        self.invalidate_cache()
        logger.info(f"Added tick size rule: {regime_name} for {symbol or 'market-wide'} effective {effective_date}")

    def bulk_load_tick_pilot_symbols(self, symbols: List[str]) -> None:
        """
        Bulk load Tick Pilot symbol list.

        Args:
            symbols: List of symbols in the Tick Pilot program
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            df = pd.read_parquet(self.path)
        else:
            df = pd.DataFrame()

        rows = []
        for symbol in symbols:
            rows.append({
                "effective_date": self.TICK_PILOT_START,
                "symbol": symbol,
                "tick_size": self.TICK_PILOT_TICK,
                "regime_name": self.TICK_PILOT,
                "notes": "SEC Tick Size Pilot Program",
                "source_name": "sec_tick_pilot",
                "ingested_at": datetime.now(timezone.utc),
            })

        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)

        # Deduplicate
        df = df.drop_duplicates(subset=["effective_date", "symbol", "regime_name"], keep="last")
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        df = df.sort_values("effective_date").reset_index(drop=True)
        df.to_parquet(self.path, index=False)

        self.invalidate_cache()
        logger.info(f"Bulk loaded {len(symbols)} Tick Pilot symbols")

    def initialize_defaults(self) -> None:
        """
        Initialize with default market-wide rules.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        rules = [
            {
                "effective_date": self.REG_NMS_EFFECTIVE,
                "symbol": None,
                "tick_size": self.REG_NMS_TICK,
                "regime_name": self.REG_NMS,
                "notes": "Reg NMS minimum tick size for prices >= $1.00",
                "source_name": "sec_reg_nms",
                "ingested_at": datetime.now(timezone.utc),
            },
        ]

        df = pd.DataFrame(rules)
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        df.to_parquet(self.path, index=False)

        self.invalidate_cache()
        logger.info("Initialized default tick size regime rules")


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage tick size regime data")
    parser.add_argument("--action", choices=["init", "lookup", "stats"], default="stats")
    parser.add_argument("--symbol", type=str, help="Symbol to lookup")
    parser.add_argument("--date", type=str, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--price", type=float, help="Price for sub-penny check")

    args = parser.parse_args()
    regime = TickSizeRegime()

    if args.action == "init":
        regime.initialize_defaults()
        print("[OK] Initialized default tick size regime rules")

    elif args.action == "lookup":
        if not args.symbol or not args.date:
            print("Error: --symbol and --date required for lookup")
        else:
            as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
            tick = regime.get_tick_size(args.symbol, as_of, args.price)
            regime_name = regime.get_regime_name(args.symbol, as_of)

            print(f"\n{args.symbol} as of {as_of}:")
            print(f"  Tick size: ${tick:.4f}")
            print(f"  Regime: {regime_name}")

            if args.price:
                constrained = regime.is_tick_constrained(args.symbol, as_of, args.price)
                print(f"  Price ${args.price:.2f} tick-constrained: {constrained}")

    else:  # stats
        df = regime._load()
        if df.empty:
            print("\n[INFO] Using default rules (no custom data loaded)")
            print(f"  Default tick: ${regime.REG_NMS_TICK}")
            print(f"  Sub-penny tick: ${regime.SUB_PENNY_TICK}")
        else:
            print(f"\n[OK] Tick size regime stats:")
            print(f"  Total rules: {len(df)}")
            print(f"  Market-wide rules: {len(df[df['symbol'].isna()])}")
            print(f"  Symbol-specific rules: {len(df[df['symbol'].notna()])}")
            print(f"\n  By regime:")
            print(df.groupby("regime_name").size().to_string())
