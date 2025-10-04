"""
Universe construction module.

Builds tradable universe with:
- Liquidity filtering (ADV threshold)
- Price filtering (minimum price)
- Survivorship bias elimination (includes delisted)
- Point-in-time snapshots
"""

from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path


class UniverseConstructor:
    """
    Build and maintain tradable universe with PIT discipline.

    Key principles:
    1. ADV calculated using trailing N-day window
    2. Include delisted names (survivorship bias elimination)
    3. Point-in-time snapshots (universe as-of date)
    4. Price and liquidity filters
    """

    def __init__(
        self,
        min_price: float = 5.0,
        min_adv_usd: float = 1_000_000.0,
        adv_window: int = 60,
        universe_size: int = 1000,
    ):
        """
        Initialize universe constructor.

        Args:
            min_price: Minimum stock price (USD)
            min_adv_usd: Minimum average daily volume (USD)
            adv_window: Trailing days for ADV calculation
            universe_size: Max number of symbols in universe
        """
        self.min_price = min_price
        self.min_adv_usd = min_adv_usd
        self.adv_window = adv_window
        self.universe_size = universe_size

        logger.info(
            f"Universe constructor initialized: min_price=${min_price}, "
            f"min_adv=${min_adv_usd:,.0f}, window={adv_window}d, size={universe_size}"
        )

    def calculate_adv(
        self,
        prices: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate average daily volume (ADV) in USD.

        Args:
            prices: OHLCV DataFrame with 'date', 'symbol', 'close', 'volume'
            window: Rolling window (default: self.adv_window)

        Returns:
            DataFrame with 'adv_usd' column
        """
        if window is None:
            window = self.adv_window

        logger.info(f"Calculating {window}-day ADV for {prices['symbol'].nunique()} symbols")

        # Calculate dollar volume for each day
        prices = prices.copy()
        prices['dollar_volume'] = prices['close'] * prices['volume']

        # Calculate rolling ADV per symbol
        adv_list = []

        for symbol in prices['symbol'].unique():
            symbol_data = prices[prices['symbol'] == symbol].sort_values('date')

            # Rolling mean of dollar volume
            symbol_data['adv_usd'] = symbol_data['dollar_volume'].rolling(
                window=window,
                min_periods=max(1, window // 2)  # Allow partial windows
            ).mean()

            adv_list.append(symbol_data[['date', 'symbol', 'adv_usd', 'close', 'volume']])

        result = pd.concat(adv_list, ignore_index=True)
        logger.info(f"Calculated ADV for {len(result)} symbol-date pairs")

        return result

    def apply_filters(
        self,
        data: pd.DataFrame,
        as_of_date: date,
        delistings: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Apply price and liquidity filters for a specific date.

        Args:
            data: DataFrame with ADV calculations
            as_of_date: Date to build universe for
            delistings: Delisting data (optional, for survivorship check)

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Applying filters for universe as-of {as_of_date}")

        # Get data for as_of_date
        snapshot = data[data['date'] == as_of_date].copy()

        if snapshot.empty:
            logger.warning(f"No data found for {as_of_date}")
            return pd.DataFrame()

        initial_count = len(snapshot)

        # Filter 1: Price >= minimum
        snapshot = snapshot[snapshot['close'] >= self.min_price]
        logger.info(f"Price filter: {len(snapshot)}/{initial_count} passed (>= ${self.min_price})")

        # Filter 2: ADV >= minimum
        snapshot = snapshot[snapshot['adv_usd'] >= self.min_adv_usd]
        logger.info(f"ADV filter: {len(snapshot)}/{initial_count} passed (>= ${self.min_adv_usd:,.0f})")

        # Filter 3: Not delisted (optional check)
        if delistings is not None and not delistings.empty:
            delisted_symbols = delistings[
                delistings['delist_date'] <= as_of_date
            ]['symbol'].unique()

            pre_delist_count = len(snapshot)
            snapshot = snapshot[~snapshot['symbol'].isin(delisted_symbols)]
            logger.info(f"Delisting filter: {len(snapshot)}/{pre_delist_count} active")

        return snapshot

    def select_top_n(
        self,
        data: pd.DataFrame,
        n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Select top N symbols by ADV.

        Args:
            data: Filtered DataFrame
            n: Number of symbols (default: self.universe_size)

        Returns:
            Top N symbols
        """
        if n is None:
            n = self.universe_size

        # Sort by ADV descending and take top N
        top_n = data.nlargest(n, 'adv_usd')

        logger.info(f"Selected top {len(top_n)} symbols by ADV")

        return top_n

    def build_universe(
        self,
        prices: pd.DataFrame,
        as_of_date: date,
        delistings: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Build universe for a specific date.

        Args:
            prices: Historical OHLCV data
            as_of_date: Date to build universe for
            delistings: Optional delisting data

        Returns:
            List of symbols in universe
        """
        logger.info(f"Building universe for {as_of_date}")

        # Calculate ADV
        adv_data = self.calculate_adv(prices)

        # Apply filters
        filtered = self.apply_filters(adv_data, as_of_date, delistings)

        if filtered.empty:
            logger.warning(f"No symbols passed filters for {as_of_date}")
            return []

        # Select top N
        universe = self.select_top_n(filtered)

        symbols = sorted(universe['symbol'].unique().tolist())

        logger.info(f"Universe for {as_of_date}: {len(symbols)} symbols")
        return symbols

    def build_historical_universes(
        self,
        prices: pd.DataFrame,
        start_date: date,
        end_date: date,
        rebalance_freq: str = 'Q',  # 'D', 'W', 'M', 'Q'
        delistings: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build historical universe snapshots.

        Args:
            prices: Historical OHLCV data
            start_date: Start date
            end_date: End date
            rebalance_freq: Rebalance frequency ('D', 'W', 'M', 'Q')
            delistings: Optional delisting data

        Returns:
            DataFrame with universe snapshots
        """
        logger.info(
            f"Building historical universes from {start_date} to {end_date}, "
            f"freq={rebalance_freq}"
        )

        # Generate rebalance dates
        rebal_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

        snapshots = []

        for rebal_date in rebal_dates:
            rebal_date = rebal_date.date()

            # Build universe for this date
            symbols = self.build_universe(prices, rebal_date, delistings)

            if symbols:
                for symbol in symbols:
                    snapshots.append({
                        'date': rebal_date,
                        'symbol': symbol,
                        'universe_name': f'top{self.universe_size}',
                    })

        if not snapshots:
            logger.warning("No historical universes generated")
            return pd.DataFrame()

        df = pd.DataFrame(snapshots)
        logger.info(f"Generated {len(df)} universe entries across {len(rebal_dates)} dates")

        return df


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Build tradable universe")
    parser.add_argument("--prices", type=str, required=True, help="Path to price parquet")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--start-date", type=str, help="Start date for historical")
    parser.add_argument("--end-date", type=str, help="End date for historical")
    parser.add_argument("--freq", type=str, default="Q", help="Rebalance frequency (D/W/M/Q)")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-adv", type=float, default=1_000_000.0)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data_layer/reference/universe")

    args = parser.parse_args()

    # Load prices
    prices = pd.read_parquet(args.prices)
    logger.info(f"Loaded {len(prices)} price bars for {prices['symbol'].nunique()} symbols")

    # Initialize constructor
    constructor = UniverseConstructor(
        min_price=args.min_price,
        min_adv_usd=args.min_adv,
        universe_size=args.size
    )

    if args.date:
        # Single date universe
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        symbols = constructor.build_universe(prices, target_date)

        print(f"\n[OK] Universe for {target_date}:")
        print(f"     {len(symbols)} symbols")
        print(f"     {', '.join(symbols[:20])}{'...' if len(symbols) > 20 else ''}")

    elif args.start_date and args.end_date:
        # Historical universes
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

        df = constructor.build_historical_universes(prices, start, end, args.freq)

        if not df.empty:
            output_path = f"{args.output}/historical_{start}_{end}.parquet"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)

            print(f"\n[OK] Historical universes generated:")
            print(f"     {len(df)} entries across {df['date'].nunique()} dates")
            print(f"     Written to {output_path}")

            # Summary
            print(f"\n     Sample dates:")
            for d in sorted(df['date'].unique())[:5]:
                count = len(df[df['date'] == d])
                print(f"       {d}: {count} symbols")
    else:
        print("Error: Provide either --date or --start-date and --end-date")
