"""
Delistings reference data manager.

Provides PIT-safe lookups for delisted securities to eliminate survivorship bias.
Merges data from multiple sources (FMP, Alpha Vantage) with priority-based deduplication.
"""

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from loguru import logger

from ..schemas import DataType, get_schema, DELISTINGS_SCHEMA


class DelistingsManager:
    """
    Manage delisting reference data with PIT discipline.

    Key principles:
    1. Merge multiple sources with priority (FMP > AV)
    2. PIT-safe lookups (only use data known at query date)
    3. Support for survivorship-bias-free universe construction
    """

    DEFAULT_PATH = Path("data_layer/reference/delistings/data.parquet")

    def __init__(self, path: Optional[Path] = None):
        """
        Initialize delistings manager.

        Args:
            path: Path to delistings parquet file (default: data_layer/reference/delistings/data.parquet)
        """
        self.path = path or self.DEFAULT_PATH
        self._cache: Optional[pd.DataFrame] = None
        logger.debug(f"DelistingsManager initialized with path: {self.path}")

    def _load(self) -> pd.DataFrame:
        """Load delistings data, using cache if available."""
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            logger.warning(f"Delistings file not found: {self.path}")
            return pd.DataFrame(columns=["symbol", "delist_date", "reason", "source_name"])

        df = pd.read_parquet(self.path)

        # Ensure delist_date is date type
        if "delist_date" in df.columns and not df.empty:
            df["delist_date"] = pd.to_datetime(df["delist_date"]).dt.date

        self._cache = df
        logger.info(f"Loaded {len(df)} delistings from {self.path}")
        return df

    def invalidate_cache(self) -> None:
        """Clear the cache to force reload on next access."""
        self._cache = None

    def is_delisted(self, symbol: str, as_of_date: date) -> bool:
        """
        Check if a symbol was delisted on or before a given date.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to check (PIT)

        Returns:
            True if symbol was delisted on or before as_of_date
        """
        df = self._load()
        if df.empty:
            return False

        match = df[(df["symbol"] == symbol) & (df["delist_date"] <= as_of_date)]
        return not match.empty

    def get_delist_date(self, symbol: str) -> Optional[date]:
        """
        Get the delisting date for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Delisting date or None if not delisted
        """
        df = self._load()
        if df.empty:
            return None

        match = df[df["symbol"] == symbol]
        if match.empty:
            return None

        return match["delist_date"].iloc[0]

    def get_delisted_symbols(self, as_of_date: date) -> Set[str]:
        """
        Get all symbols delisted on or before a given date.

        Args:
            as_of_date: Date to check (PIT)

        Returns:
            Set of delisted symbols
        """
        df = self._load()
        if df.empty:
            return set()

        delisted = df[df["delist_date"] <= as_of_date]["symbol"].unique()
        return set(delisted)

    def get_active_symbols(self, symbols: List[str], as_of_date: date) -> List[str]:
        """
        Filter a list of symbols to only those active (not delisted) as of a date.

        Args:
            symbols: List of symbols to filter
            as_of_date: Date to check (PIT)

        Returns:
            List of symbols that were not delisted as of as_of_date
        """
        delisted = self.get_delisted_symbols(as_of_date)
        return [s for s in symbols if s not in delisted]

    def get_all_delistings(self) -> pd.DataFrame:
        """
        Get all delistings data.

        Returns:
            DataFrame with all delistings
        """
        return self._load().copy()

    @staticmethod
    def merge_sources(
        fmp_df: pd.DataFrame,
        av_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Merge delistings from FMP and Alpha Vantage with priority deduplication.

        FMP has priority 1 (highest), AV has priority 2.
        When both sources have the same symbol, FMP data is kept.

        Args:
            fmp_df: DataFrame from FMP connector
            av_df: DataFrame from Alpha Vantage connector
            output_path: Optional path to write merged result

        Returns:
            Merged DataFrame with deduplicated delistings
        """
        # Add source priority
        if not fmp_df.empty:
            fmp_df = fmp_df.copy()
            if "source_priority" not in fmp_df.columns:
                fmp_df["source_priority"] = 1

        if not av_df.empty:
            av_df = av_df.copy()
            if "source_priority" not in av_df.columns:
                av_df["source_priority"] = 2

        # Combine
        dfs = [df for df in [fmp_df, av_df] if not df.empty]
        if not dfs:
            return pd.DataFrame(columns=["symbol", "delist_date", "reason", "source_name", "source_priority"])

        combined = pd.concat(dfs, ignore_index=True)

        # Deduplicate by symbol, keeping highest priority (lowest number)
        combined = combined.sort_values(["symbol", "source_priority"])
        deduplicated = combined.drop_duplicates(subset=["symbol"], keep="first")

        # Ensure required columns
        required_cols = ["symbol", "delist_date", "reason", "source_name"]
        for col in required_cols:
            if col not in deduplicated.columns:
                deduplicated[col] = None

        # Clean up
        deduplicated = deduplicated.sort_values("delist_date", ascending=False).reset_index(drop=True)

        logger.info(
            f"Merged {len(fmp_df) if not fmp_df.empty else 0} FMP + "
            f"{len(av_df) if not av_df.empty else 0} AV delistings → {len(deduplicated)} unique"
        )

        # Optionally write
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            deduplicated.to_parquet(output_path, index=False)
            logger.info(f"Wrote merged delistings to {output_path}")

        return deduplicated


def fetch_and_merge_delistings(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch delistings from all configured sources and merge.

    Args:
        output_path: Path to write merged result (default: DelistingsManager.DEFAULT_PATH)

    Returns:
        Merged DataFrame
    """
    from ..connectors.fmp_connector import FMPConnector
    from ..connectors.alpha_vantage_connector import AlphaVantageConnector

    output_path = output_path or DelistingsManager.DEFAULT_PATH

    fmp_df = pd.DataFrame()
    av_df = pd.DataFrame()

    # Fetch from FMP
    try:
        fmp = FMPConnector()
        fmp_df = fmp.fetch_delisted_companies()
        logger.info(f"Fetched {len(fmp_df)} delistings from FMP")
    except Exception as e:
        logger.warning(f"Failed to fetch FMP delistings: {e}")

    # Fetch from Alpha Vantage
    try:
        av = AlphaVantageConnector()
        av_df = av.fetch_listing_status(state="delisted")
        logger.info(f"Fetched {len(av_df)} delistings from Alpha Vantage")
    except Exception as e:
        logger.warning(f"Failed to fetch AV delistings: {e}")

    # Merge and save
    merged = DelistingsManager.merge_sources(fmp_df, av_df, output_path)
    return merged


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage delistings reference data")
    parser.add_argument("--action", choices=["fetch", "lookup", "stats"], default="stats")
    parser.add_argument("--symbol", type=str, help="Symbol to lookup")
    parser.add_argument("--date", type=str, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data_layer/reference/delistings/data.parquet")

    args = parser.parse_args()

    if args.action == "fetch":
        df = fetch_and_merge_delistings(Path(args.output))
        print(f"\n[OK] Fetched and merged {len(df)} delistings")
        print(f"\nMost recent delistings:")
        print(df.head(10)[["symbol", "delist_date", "reason", "source_name"]])

    elif args.action == "lookup":
        manager = DelistingsManager(Path(args.output))
        if args.symbol:
            as_of = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
            is_del = manager.is_delisted(args.symbol, as_of)
            delist_date = manager.get_delist_date(args.symbol)
            print(f"\n{args.symbol} as of {as_of}:")
            print(f"  Delisted: {is_del}")
            print(f"  Delist date: {delist_date}")
        else:
            print("Error: --symbol required for lookup")

    else:  # stats
        manager = DelistingsManager(Path(args.output))
        df = manager.get_all_delistings()
        if df.empty:
            print("\n[WARN] No delistings data found. Run --action fetch first.")
        else:
            print(f"\n[OK] Delistings stats:")
            print(f"  Total: {len(df)}")
            print(f"  Date range: {df['delist_date'].min()} to {df['delist_date'].max()}")
            print(f"\n  By source:")
            print(df.groupby("source_name").size().to_string())
            print(f"\n  By reason:")
            print(df.groupby("reason").size().head(10).to_string())
