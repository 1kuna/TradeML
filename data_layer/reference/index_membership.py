"""
Index membership reference data manager.

Provides PIT-safe lookups for index constituents (S&P 500, Russell 1000/2000, etc.)
to enable proper universe filtering for historical training and backtesting.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Dict

import pandas as pd
from loguru import logger

from ..schemas import INDEX_MEMBERSHIP_SCHEMA


class IndexMembershipManager:
    """
    Manage index membership reference data with PIT discipline.

    Key principles:
    1. Track ADD/REMOVE events with effective dates
    2. PIT-safe lookups (reconstruct membership at any historical date)
    3. Support multiple indices (SP500, R1000, R2000)
    """

    DEFAULT_PATH = Path("data_layer/reference/index_membership")

    # Known index names
    SP500 = "SP500"
    R1000 = "R1000"
    R2000 = "R2000"
    R3000 = "R3000"

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize index membership manager.

        Args:
            base_path: Base path for index membership data
                       (default: data_layer/reference/index_membership)
        """
        self.base_path = base_path or self.DEFAULT_PATH
        self._cache: Dict[str, pd.DataFrame] = {}
        logger.debug(f"IndexMembershipManager initialized with path: {self.base_path}")

    def _get_index_path(self, index_name: str) -> Path:
        """Get path to index-specific parquet file."""
        return self.base_path / f"{index_name.lower()}.parquet"

    def _load_index(self, index_name: str) -> pd.DataFrame:
        """Load index membership events, using cache if available."""
        if index_name in self._cache:
            return self._cache[index_name]

        path = self._get_index_path(index_name)
        if not path.exists():
            logger.warning(f"Index membership file not found: {path}")
            return pd.DataFrame(columns=["date", "index_name", "symbol", "action"])

        df = pd.read_parquet(path)

        # Ensure date is date type
        if "date" in df.columns and not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        self._cache[index_name] = df
        logger.info(f"Loaded {len(df)} events for index {index_name}")
        return df

    def invalidate_cache(self, index_name: Optional[str] = None) -> None:
        """Clear the cache to force reload on next access."""
        if index_name:
            self._cache.pop(index_name, None)
        else:
            self._cache.clear()

    def get_constituents(self, index_name: str, as_of_date: date) -> Set[str]:
        """
        Get index constituents as of a specific date (PIT).

        Reconstructs membership by replaying ADD/REMOVE events up to as_of_date.

        Args:
            index_name: Index name (e.g., "SP500", "R1000")
            as_of_date: Date to query (PIT)

        Returns:
            Set of symbols that were index constituents on as_of_date
        """
        df = self._load_index(index_name)
        if df.empty:
            return set()

        # Filter events up to as_of_date
        events = df[df["date"] <= as_of_date].copy()
        if events.empty:
            return set()

        # Replay events to build membership
        members: Set[str] = set()

        for _, row in events.iterrows():
            symbol = row["symbol"]
            action = row["action"].upper()

            if action == "ADD":
                members.add(symbol)
            elif action == "REMOVE":
                members.discard(symbol)

        return members

    def is_member(self, index_name: str, symbol: str, as_of_date: date) -> bool:
        """
        Check if a symbol was an index member on a specific date.

        Args:
            index_name: Index name
            symbol: Ticker symbol
            as_of_date: Date to check (PIT)

        Returns:
            True if symbol was a member of the index on as_of_date
        """
        members = self.get_constituents(index_name, as_of_date)
        return symbol in members

    def get_membership_history(
        self,
        index_name: str,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get membership history for a symbol in an index.

        Args:
            index_name: Index name
            symbol: Ticker symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with ADD/REMOVE events for the symbol
        """
        df = self._load_index(index_name)
        if df.empty:
            return pd.DataFrame(columns=["date", "index_name", "symbol", "action"])

        # Filter to symbol
        history = df[df["symbol"] == symbol].copy()

        # Apply date filters
        if start_date:
            history = history[history["date"] >= start_date]
        if end_date:
            history = history[history["date"] <= end_date]

        return history.reset_index(drop=True)

    def filter_by_index(
        self,
        symbols: List[str],
        index_name: str,
        as_of_date: date,
    ) -> List[str]:
        """
        Filter a list of symbols to only those in a specific index.

        Args:
            symbols: List of symbols to filter
            index_name: Index to filter by
            as_of_date: Date for membership check (PIT)

        Returns:
            List of symbols that are index members
        """
        members = self.get_constituents(index_name, as_of_date)
        return [s for s in symbols if s in members]

    def add_event(
        self,
        index_name: str,
        symbol: str,
        event_date: date,
        action: str,
        source_name: str = "manual",
    ) -> None:
        """
        Add a membership event.

        Args:
            index_name: Index name
            symbol: Ticker symbol
            event_date: Date of the event
            action: "ADD" or "REMOVE"
            source_name: Source of the data
        """
        action = action.upper()
        if action not in ("ADD", "REMOVE"):
            raise ValueError(f"Action must be ADD or REMOVE, got: {action}")

        path = self._get_index_path(index_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing or create new
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = pd.DataFrame(columns=["date", "index_name", "symbol", "action", "source_name", "ingested_at"])

        # Add new event
        new_row = {
            "date": event_date,
            "index_name": index_name,
            "symbol": symbol,
            "action": action,
            "source_name": source_name,
            "ingested_at": datetime.now(timezone.utc),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Sort and save
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(path, index=False)

        # Invalidate cache
        self.invalidate_cache(index_name)
        logger.info(f"Added {action} event for {symbol} in {index_name} on {event_date}")

    def bulk_load(
        self,
        index_name: str,
        events_df: pd.DataFrame,
        replace: bool = False,
    ) -> None:
        """
        Bulk load membership events.

        Args:
            index_name: Index name
            events_df: DataFrame with columns: date, symbol, action
            replace: If True, replace existing data; if False, append
        """
        path = self._get_index_path(index_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare new events
        events = events_df.copy()
        events["index_name"] = index_name
        if "source_name" not in events.columns:
            events["source_name"] = "bulk_load"
        if "ingested_at" not in events.columns:
            events["ingested_at"] = datetime.now(timezone.utc)

        # Ensure date type
        events["date"] = pd.to_datetime(events["date"]).dt.date

        if replace or not path.exists():
            df = events
        else:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, events], ignore_index=True)

        # Deduplicate (same symbol, date, action)
        df = df.drop_duplicates(subset=["date", "symbol", "action"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(path, index=False)

        # Invalidate cache
        self.invalidate_cache(index_name)
        logger.info(f"Bulk loaded {len(events)} events for {index_name}")

    def get_all_events(self, index_name: str) -> pd.DataFrame:
        """Get all membership events for an index."""
        return self._load_index(index_name).copy()


def fetch_sp500_constituents_fmp() -> pd.DataFrame:
    """
    Fetch S&P 500 historical constituents from FMP API.

    Returns:
        DataFrame with columns: date, symbol, action
    """
    try:
        from ..connectors.fmp_connector import FMPConnector

        fmp = FMPConnector()
        # FMP provides /v3/historical/sp500_constituent
        raw = fmp._fetch_raw("historical/sp500_constituent")

        if not isinstance(raw, list):
            logger.warning("Unexpected response format from FMP sp500_constituent")
            return pd.DataFrame()

        events = []
        for item in raw:
            # Each item has: dateAdded, symbol (for additions)
            # and removedTicker, dateRemoved (for removals)
            if item.get("dateAdded") and item.get("symbol"):
                events.append({
                    "date": datetime.strptime(item["dateAdded"], "%Y-%m-%d").date(),
                    "symbol": item["symbol"],
                    "action": "ADD",
                    "source_name": "fmp",
                })
            if item.get("removedTicker") and item.get("date"):
                events.append({
                    "date": datetime.strptime(item["date"], "%Y-%m-%d").date(),
                    "symbol": item["removedTicker"],
                    "action": "REMOVE",
                    "source_name": "fmp",
                })

        df = pd.DataFrame(events)
        if not df.empty:
            df["ingested_at"] = datetime.utcnow()
        return df

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 constituents from FMP: {e}")
        return pd.DataFrame()


def fetch_and_load_sp500(manager: Optional[IndexMembershipManager] = None) -> pd.DataFrame:
    """
    Fetch S&P 500 constituents and load into manager.

    Args:
        manager: IndexMembershipManager instance (creates default if None)

    Returns:
        DataFrame of loaded events
    """
    manager = manager or IndexMembershipManager()
    events = fetch_sp500_constituents_fmp()

    if not events.empty:
        manager.bulk_load(IndexMembershipManager.SP500, events, replace=True)

    return events


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage index membership data")
    parser.add_argument("--action", choices=["fetch", "lookup", "stats"], default="stats")
    parser.add_argument("--index", type=str, default="SP500", help="Index name")
    parser.add_argument("--symbol", type=str, help="Symbol to lookup")
    parser.add_argument("--date", type=str, help="As-of date (YYYY-MM-DD)")

    args = parser.parse_args()
    manager = IndexMembershipManager()

    if args.action == "fetch":
        if args.index.upper() == "SP500":
            events = fetch_and_load_sp500(manager)
            print(f"\n[OK] Fetched and loaded {len(events)} S&P 500 events")
        else:
            print(f"[WARN] Fetch not implemented for {args.index}")

    elif args.action == "lookup":
        if not args.date:
            print("Error: --date required for lookup")
        else:
            as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
            members = manager.get_constituents(args.index, as_of)
            print(f"\n{args.index} constituents as of {as_of}: {len(members)} symbols")
            if args.symbol:
                is_member = args.symbol in members
                print(f"{args.symbol} is member: {is_member}")
            else:
                print(f"Sample: {list(members)[:20]}")

    else:  # stats
        events = manager.get_all_events(args.index)
        if events.empty:
            print(f"\n[WARN] No data for {args.index}. Run --action fetch first.")
        else:
            print(f"\n[OK] {args.index} stats:")
            print(f"  Total events: {len(events)}")
            print(f"  Date range: {events['date'].min()} to {events['date'].max()}")
            print(f"  ADDs: {len(events[events['action'] == 'ADD'])}")
            print(f"  REMOVEs: {len(events[events['action'] == 'REMOVE'])}")
