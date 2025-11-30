"""
Corporate actions pipeline with point-in-time adjustments.

Handles:
- Stock splits (forward and reverse)
- Cash dividends
- Special dividends
- Spin-offs (future)

Ensures PIT discipline - adjustments applied ONLY using information available at backtest time.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class CorporateActionsProcessor:
    """
    Process corporate actions and generate adjusted price series.

    Key principles:
    1. Adjustments flow backwards in time (ex-date is boundary)
    2. Cumulative adjustment factor tracked
    3. PIT-safe: use only events known as of backtest date
    4. Volume adjusted for splits (not dividends)
    """

    def __init__(self, events: Optional[pd.DataFrame] = None):
        """Initialize corporate actions processor."""
        self.events = events
        logger.info("Corporate actions processor initialized")

    def load_events(self, path: str) -> pd.DataFrame:
        """
        Load corporate actions from Parquet.

        Args:
            path: Path to corp actions parquet file

        Returns:
            DataFrame with events
        """
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} corporate action events from {path}")

        # Ensure ex_date is date type
        if 'ex_date' in df.columns:
            df['ex_date'] = pd.to_datetime(df['ex_date']).dt.date

        return df

    def load_events_from_dir(self, root: str = "data_layer/reference/corp_actions") -> pd.DataFrame:
        """
        Load and concatenate all corporate actions parquet files under a directory.

        Args:
            root: Directory containing *_corp_actions.parquet files

        Returns:
            DataFrame with all events; empty DataFrame if none found.
        """
        root_path = Path(root)
        frames: List[pd.DataFrame] = []
        for p in sorted(root_path.glob("*.parquet")):
            try:
                frames.append(self.load_events(str(p)))
            except Exception as e:
                logger.warning(f"Failed to load corp actions from {p}: {e}")
        if not frames:
            logger.warning(f"No corporate actions parquet files found under {root}")
            return pd.DataFrame()

        events = pd.concat(frames, ignore_index=True)
        self.events = events
        return events

    def calculate_split_adjustment(
        self,
        events: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Calculate cumulative split adjustment factors.

        Args:
            events: Corp actions DataFrame
            symbol: Symbol to calculate for

        Returns:
            DataFrame with dates and adjustment factors.
            Columns:
                - ex_date: split effective date
                - split_ratio: per-split ratio (e.g., 4.0 for 4:1)
                - cumulative_factor: product of split ratios from that ex_date backward
        """
        # Filter to splits for this symbol
        splits = events[
            (events['symbol'] == symbol) &
            (events['event_type'] == 'split')
        ].copy()

        if splits.empty:
            return pd.DataFrame(columns=['ex_date', 'split_ratio', 'cumulative_factor'])

        # Sort by ex_date descending so cumulative factors are built from the most recent split backwards
        splits['ex_date'] = pd.to_datetime(splits['ex_date']).dt.date
        splits = splits.sort_values('ex_date', ascending=False)

        # Keep both the per-event ratio and the cumulative factor that downstream
        # callers (and tests) rely on for PIT adjustments.
        splits['split_ratio'] = splits['ratio']
        splits['cumulative_factor'] = splits['split_ratio'].cumprod()

        adjustments = splits[['ex_date', 'split_ratio', 'cumulative_factor']].reset_index(drop=True)
        logger.info(f"Calculated {len(adjustments)} split adjustments for {symbol}")
        return adjustments

    def get_adjustment_factor(self, symbol: str, asof: date, events: Optional[pd.DataFrame] = None) -> float:
        """
        Point-in-time cumulative split factor applicable strictly before `asof`.

        Args:
            symbol: Ticker
            asof: Date for which the factor should apply (PIT – only splits after this date)
            events: Optional overrides for events DataFrame

        Returns:
            Cumulative split factor; 1.0 if no applicable splits.
        """
        events_df = events if events is not None else self.events
        if events_df is None or events_df.empty:
            return 1.0

        splits = events_df[
            (events_df["symbol"] == symbol) &
            (events_df["event_type"] == "split")
        ].copy()
        if splits.empty:
            return 1.0

        splits["ex_date"] = pd.to_datetime(splits["ex_date"]).dt.date
        # Only apply splits with ex_date strictly after the current bar (PIT discipline)
        applicable = splits[splits["ex_date"] > asof]
        if applicable.empty:
            return 1.0

        factor = float(applicable["ratio"].prod())
        return factor

    def get_dividend_amount(self, symbol: str, ex_date: date, events: Optional[pd.DataFrame] = None) -> float:
        """
        Return per-share cash dividend amount on ex_date for symbol.

        Args:
            symbol: Ticker
            ex_date: Ex-dividend date
            events: Optional overrides for events DataFrame

        Returns:
            Dividend amount if present, else 0.0
        """
        events_df = events if events is not None else self.events
        if events_df is None or events_df.empty:
            return 0.0

        divs = events_df[
            (events_df["symbol"] == symbol) &
            (events_df["event_type"] == "dividend")
        ].copy()
        if divs.empty:
            return 0.0

        divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.date
        match = divs[divs["ex_date"] == ex_date]
        if match.empty:
            return 0.0

        try:
            return float(match.iloc[0].get("amount", 0.0) or 0.0)
        except Exception:
            return 0.0

    def apply_split_adjustments(
        self,
        prices: pd.DataFrame,
        adjustments: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply split adjustments to price data.

        Args:
            prices: OHLCV DataFrame with 'date' column
            adjustments: Adjustment factors with required columns:
                - ex_date
                - split_ratio
                - cumulative_factor (computed if absent)

        Returns:
            Adjusted prices DataFrame
        """
        if adjustments.empty:
            # No adjustments needed
            adj = prices.copy()
            adj['adjustment_factor'] = 1.0
            adj['last_adjustment_date'] = None
            return adj

        adj = prices.copy()
        adj_factors = adjustments.copy()

        # Ensure date is proper datetime for comparison
        if isinstance(adj['date'].dtype, pd.CategoricalDtype):
            adj['date'] = pd.to_datetime(adj['date'].astype(str)).dt.date
        elif not isinstance(adj['date'].iloc[0], date):
            adj['date'] = pd.to_datetime(adj['date']).dt.date

        adj['adjustment_factor'] = 1.0
        adj['last_adjustment_date'] = None

        # Validate schema
        if 'ex_date' not in adj_factors.columns:
            raise ValueError("Split adjustments must include an 'ex_date' column.")
        if 'split_ratio' not in adj_factors.columns:
            raise ValueError("Split adjustments must include a 'split_ratio' column.")

        adj_factors['ex_date'] = pd.to_datetime(adj_factors['ex_date']).dt.date

        if 'cumulative_factor' not in adj_factors.columns:
            raise ValueError("Split adjustments must include a 'cumulative_factor' column.")

        # Apply in chronological order so last_adjustment_date reflects the latest split
        adj_factors = adj_factors.sort_values('ex_date', ascending=True).reset_index(drop=True)

        # Apply adjustments backwards in time
        for _, row in adj_factors.iterrows():
            ex_date = row['ex_date']
            factor = row['split_ratio']

            # Adjust all prices BEFORE ex_date
            mask = adj['date'] < ex_date

            adj.loc[mask, 'open'] *= factor
            adj.loc[mask, 'high'] *= factor
            adj.loc[mask, 'low'] *= factor
            adj.loc[mask, 'close'] *= factor
            if 'vwap' in adj.columns:
                adj.loc[mask, 'vwap'] *= factor

            # Adjust volume (inverse of price adjustment)
            adj.loc[mask, 'volume'] = (adj.loc[mask, 'volume'] / factor).astype(int)

            # Track adjustment factor
            adj.loc[mask, 'adjustment_factor'] *= factor
            adj.loc[mask, 'last_adjustment_date'] = ex_date

        return adj

    def calculate_dividend_return(
        self,
        events: pd.DataFrame,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Calculate total return series including dividends.

        Args:
            events: Corp actions DataFrame
            prices: Price DataFrame (unadjusted)
            symbol: Symbol

        Returns:
            DataFrame with dividend return column added
        """
        # Filter to dividends for this symbol
        divs = events[
            (events['symbol'] == symbol) &
            (events['event_type'] == 'dividend')
        ].copy()

        if divs.empty:
            prices['dividend_return'] = 0.0
            return prices

        # Merge dividends with prices by ex_date = date
        result = prices.copy()
        result['dividend_return'] = 0.0

        for _, div in divs.iterrows():
            ex_date = div['ex_date']
            amount = div['amount']

            # Find closing price on day before ex_date
            prev_close = result[result['date'] < ex_date]['close'].iloc[-1] if len(result[result['date'] < ex_date]) > 0 else np.nan

            if not np.isnan(prev_close) and prev_close > 0:
                div_return = amount / prev_close
                result.loc[result['date'] == ex_date, 'dividend_return'] = div_return

        return result

    def generate_adjusted_series(
        self,
        raw_prices: pd.DataFrame,
        corp_actions: Optional[pd.DataFrame],
        symbol: str
    ) -> pd.DataFrame:
        """
        Generate fully adjusted price series.

        Args:
            raw_prices: Raw OHLCV data
            corp_actions: All corporate actions for symbol
            symbol: Symbol to adjust

        Returns:
            Adjusted DataFrame with metadata
        """
        logger.info(f"Generating adjusted series for {symbol}")

        # Use cached events if none supplied
        events_df = corp_actions if corp_actions is not None else self.events
        if events_df is None:
            events_df = pd.DataFrame()

        # Calculate split adjustments
        split_adj = self.calculate_split_adjustment(events_df, symbol)

        # Apply split adjustments to prices
        adjusted = self.apply_split_adjustments(raw_prices, split_adj)

        # Add dividend returns (not price-adjusted, but informational)
        adjusted = self.calculate_dividend_return(events_df, adjusted, symbol)

        # Store original close for reference
        adjusted['close_raw'] = raw_prices['close']

        # Rename adjusted columns
        adjusted = adjusted.rename(columns={
            'open': 'open_adj',
            'high': 'high_adj',
            'low': 'low_adj',
            'close': 'close_adj',
            'vwap': 'vwap_adj',
            'volume': 'volume_adj'
        })

        # Ensure adjustment_factor and last_adjustment_date are present
        if 'adjustment_factor' not in adjusted.columns:
            adjusted['adjustment_factor'] = 1.0
        if 'last_adjustment_date' not in adjusted.columns:
            adjusted['last_adjustment_date'] = None

        logger.info(f"Adjusted {len(adjusted)} bars for {symbol}")

        return adjusted

    def validate_adjustments(
        self,
        raw: pd.DataFrame,
        adjusted: pd.DataFrame,
        events: pd.DataFrame,
        symbol: str
    ) -> Dict[str, bool]:
        """
        Validate that adjustments were applied correctly.

        Args:
            raw: Raw prices
            adjusted: Adjusted prices
            events: Corporate actions
            symbol: Symbol

        Returns:
            Dict with validation results
        """
        results = {}

        # Ensure dates are comparable
        raw = raw.copy()
        if isinstance(raw['date'].dtype, pd.CategoricalDtype):
            raw['date'] = pd.to_datetime(raw['date'].astype(str)).dt.date
        elif not isinstance(raw['date'].iloc[0], date):
            raw['date'] = pd.to_datetime(raw['date']).dt.date

        # Check: Adjusted prices should equal raw on latest date (no future adjustments)
        latest_date = raw['date'].max()
        raw_latest = raw[raw['date'] == latest_date]['close'].iloc[0]
        adj_latest = adjusted[adjusted['date'] == latest_date]['close_adj'].iloc[0]

        results['latest_price_match'] = np.isclose(raw_latest, adj_latest, rtol=1e-6)

        # Check: Split ratios applied correctly
        splits = events[(events['symbol'] == symbol) & (events['event_type'] == 'split')]

        if not splits.empty:
            for _, split in splits.iterrows():
                ex_date = split['ex_date']
                ratio = split['ratio']

                # Get price day before and day of ex_date
                before_raw = raw[raw['date'] < ex_date]['close'].iloc[-1] if len(raw[raw['date'] < ex_date]) > 0 else None
                on_raw = raw[raw['date'] == ex_date]['close'].iloc[0] if len(raw[raw['date'] == ex_date]) > 0 else None

                before_adj = adjusted[adjusted['date'] < ex_date]['close_adj'].iloc[-1] if len(adjusted[adjusted['date'] < ex_date]) > 0 else None
                on_adj = adjusted[adjusted['date'] == ex_date]['close_adj'].iloc[0] if len(adjusted[adjusted['date'] == ex_date]) > 0 else None

                if all([before_raw, on_raw, before_adj, on_adj]):
                    # Check that raw dropped by ~ratio, adj stayed continuous
                    raw_ratio = before_raw / on_raw
                    adj_continuous = np.isclose(before_adj, on_adj, rtol=0.1)

                    results[f'split_{ex_date}_ratio'] = np.isclose(raw_ratio, ratio, rtol=0.1)
                    results[f'split_{ex_date}_continuity'] = adj_continuous

        logger.info(f"Validation results for {symbol}: {results}")
        return results


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Process corporate actions")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to process")
    parser.add_argument("--prices", type=str, required=True, help="Path to raw price parquet")
    parser.add_argument("--events", type=str, required=True, help="Path to corp actions parquet")
    parser.add_argument("--output", type=str, default="data_layer/curated/equities_ohlcv_adj")

    args = parser.parse_args()

    processor = CorporateActionsProcessor()

    # Load data
    prices = pd.read_parquet(args.prices)
    prices = prices[prices['symbol'] == args.symbol].sort_values('date').reset_index(drop=True)

    events = processor.load_events(args.events)

    # Generate adjusted series
    adjusted = processor.generate_adjusted_series(prices, events, args.symbol)

    # Validate
    validation = processor.validate_adjustments(prices, adjusted, events, args.symbol)

    # Write output
    output_path = f"{args.output}/{args.symbol}_adj.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_parquet(output_path, index=False)

    print(f"[OK] Generated adjusted series for {args.symbol}")
    print(f"     {len(adjusted)} bars written to {output_path}")
    print(f"     Validation: {validation}")
