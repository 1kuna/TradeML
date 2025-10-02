"""
Data Quality Check Suite.

Validates:
- Schema conformance
- Timestamp monotonicity
- Outlier detection
- Coverage gaps
- Point-in-time safety
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger
import pyarrow.parquet as pq
from pathlib import Path

from ..schemas import get_schema, DataType


class DataQualityChecker:
    """
    Comprehensive data quality validation.

    Checks:
    1. Schema validation (columns, types)
    2. Monotonicity (timestamps increasing)
    3. Outliers (>5σ price moves)
    4. Coverage gaps (missing dates)
    5. PIT safety (no future data)
    """

    def __init__(self, sigma_threshold: float = 5.0):
        """
        Initialize QC checker.

        Args:
            sigma_threshold: Outlier threshold in standard deviations
        """
        self.sigma_threshold = sigma_threshold
        logger.info(f"Data quality checker initialized (σ threshold: {sigma_threshold})")

    def check_schema(
        self,
        df: pd.DataFrame,
        data_type: DataType
    ) -> Dict[str, Any]:
        """
        Validate DataFrame schema.

        Args:
            df: DataFrame to validate
            data_type: Expected data type

        Returns:
            Dict with validation results
        """
        results = {
            'check': 'schema_validation',
            'passed': True,
            'issues': []
        }

        schema = get_schema(data_type)
        expected_cols = set(schema.names)
        actual_cols = set(df.columns)

        # Check missing columns
        missing = expected_cols - actual_cols
        if missing:
            results['passed'] = False
            results['issues'].append(f"Missing columns: {missing}")

        # Check extra columns
        extra = actual_cols - expected_cols
        if extra:
            results['issues'].append(f"Extra columns: {extra}")

        # Check required metadata
        required_metadata = ['ingested_at', 'source_name', 'source_uri']
        for col in required_metadata:
            if col in actual_cols:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results['passed'] = False
                    results['issues'].append(f"{col} has {null_count} nulls")

        logger.info(f"Schema check: {'PASS' if results['passed'] else 'FAIL'}")
        return results

    def check_monotonicity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Check that timestamps are monotonically increasing per symbol.

        Args:
            df: DataFrame with timestamps
            timestamp_col: Name of timestamp column

        Returns:
            Dict with validation results
        """
        results = {
            'check': 'monotonicity',
            'passed': True,
            'issues': []
        }

        if timestamp_col not in df.columns:
            results['passed'] = False
            results['issues'].append(f"Timestamp column '{timestamp_col}' not found")
            return results

        # Check per symbol if symbol column exists
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_index()
                timestamps = symbol_data[timestamp_col].values

                # Check for duplicates
                if len(timestamps) != len(set(timestamps)):
                    results['issues'].append(f"{symbol}: Duplicate timestamps found")

                # Check for backwards movement
                for i in range(1, len(timestamps)):
                    if timestamps[i] < timestamps[i-1]:
                        results['passed'] = False
                        results['issues'].append(
                            f"{symbol}: Timestamp regression at index {i}"
                        )
                        break
        else:
            # Global check
            timestamps = df[timestamp_col].values
            if len(timestamps) != len(set(timestamps)):
                results['issues'].append("Duplicate timestamps found")

            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    results['passed'] = False
                    results['issues'].append(f"Timestamp regression at index {i}")
                    break

        logger.info(f"Monotonicity check: {'PASS' if results['passed'] else 'FAIL'}")
        return results

    def check_outliers(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        return_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect outlier price moves (>Nσ).

        Args:
            df: DataFrame with prices
            price_col: Price column to check
            return_threshold: Outlier threshold (default: self.sigma_threshold)

        Returns:
            Dict with validation results
        """
        if return_threshold is None:
            return_threshold = self.sigma_threshold

        results = {
            'check': 'outlier_detection',
            'passed': True,
            'outliers': []
        }

        if price_col not in df.columns:
            results['passed'] = False
            results['issues'] = [f"Price column '{price_col}' not found"]
            return results

        # Calculate returns per symbol
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('date')

                if len(symbol_data) < 2:
                    continue

                # Calculate log returns
                prices = symbol_data[price_col].values
                returns = np.diff(np.log(prices))

                # Detect outliers
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)

                if std_ret > 0:
                    z_scores = np.abs((returns - mean_ret) / std_ret)
                    outlier_indices = np.where(z_scores > return_threshold)[0]

                    for idx in outlier_indices:
                        date_val = symbol_data.iloc[idx + 1]['date'] if 'date' in symbol_data.columns else idx + 1
                        results['outliers'].append({
                            'symbol': symbol,
                            'date': date_val,
                            'return': returns[idx],
                            'z_score': z_scores[idx]
                        })

        if len(results['outliers']) > 0:
            results['passed'] = False
            logger.warning(f"Found {len(results['outliers'])} outliers (>{return_threshold}σ)")
        else:
            logger.info("Outlier check: PASS (no outliers)")

        return results

    def check_coverage(
        self,
        df: pd.DataFrame,
        start_date: date,
        end_date: date,
        expected_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check for missing dates or symbols.

        Args:
            df: DataFrame to check
            start_date: Expected start date
            end_date: Expected end date
            expected_symbols: Expected symbols (optional)

        Returns:
            Dict with validation results
        """
        from data_layer.reference.calendars import get_calendar

        results = {
            'check': 'coverage',
            'passed': True,
            'gaps': []
        }

        cal = get_calendar("XNYS")
        expected_dates = cal.get_trading_days(start_date, end_date)

        if 'symbol' in df.columns and expected_symbols:
            # Check per symbol
            for symbol in expected_symbols:
                symbol_data = df[df['symbol'] == symbol]
                actual_dates = set(symbol_data['date'].unique())

                missing_dates = set(expected_dates) - actual_dates
                if missing_dates:
                    results['gaps'].append({
                        'symbol': symbol,
                        'missing_dates': sorted(missing_dates),
                        'count': len(missing_dates)
                    })
        else:
            # Global check
            actual_dates = set(df['date'].unique())
            missing_dates = set(expected_dates) - actual_dates

            if missing_dates:
                results['gaps'].append({
                    'missing_dates': sorted(missing_dates),
                    'count': len(missing_dates)
                })

        if len(results['gaps']) > 0:
            results['passed'] = False
            total_gaps = sum(g['count'] for g in results['gaps'])
            logger.warning(f"Coverage gaps found: {total_gaps} missing dates")
        else:
            logger.info("Coverage check: PASS")

        return results

    def check_pit_safety(
        self,
        df: pd.DataFrame,
        data_date_col: str = 'date',
        ingestion_col: str = 'ingested_at'
    ) -> Dict[str, Any]:
        """
        Verify point-in-time safety (no future data leakage).

        Args:
            df: DataFrame to check
            data_date_col: Column with data date
            ingestion_col: Column with ingestion timestamp

        Returns:
            Dict with validation results
        """
        from data_layer.reference.calendars import get_calendar

        results = {
            'check': 'pit_safety',
            'passed': True,
            'violations': []
        }

        if data_date_col not in df.columns or ingestion_col not in df.columns:
            results['passed'] = False
            results['violations'].append("Required columns missing")
            return results

        cal = get_calendar("XNYS")

        for idx, row in df.iterrows():
            data_date = row[data_date_col]
            ingestion_ts = row[ingestion_col]

            if pd.isna(data_date) or pd.isna(ingestion_ts):
                continue

            # Convert to date if needed
            if isinstance(data_date, str):
                data_date = datetime.strptime(data_date, "%Y-%m-%d").date()
            elif isinstance(data_date, pd.Timestamp):
                data_date = data_date.date()

            # Get session close time for data_date
            session_times = cal.get_session_times(data_date)

            if session_times:
                _, close_time = session_times

                # Ingestion must be >= close_time
                if isinstance(ingestion_ts, str):
                    ingestion_ts = pd.to_datetime(ingestion_ts)

                # Make timezone-aware comparison
                if ingestion_ts.tz is None:
                    ingestion_ts = ingestion_ts.tz_localize('UTC')
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=ingestion_ts.tzinfo)

                if ingestion_ts < close_time:
                    results['passed'] = False
                    results['violations'].append({
                        'index': idx,
                        'data_date': data_date,
                        'ingestion_ts': ingestion_ts,
                        'session_close': close_time
                    })

        if len(results['violations']) > 0:
            logger.error(f"PIT violations: {len(results['violations'])} found")
        else:
            logger.info("PIT safety check: PASS")

        return results

    def run_all_checks(
        self,
        df: pd.DataFrame,
        data_type: DataType,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        expected_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run all QC checks.

        Args:
            df: DataFrame to validate
            data_type: Data type for schema check
            start_date: For coverage check
            end_date: For coverage check
            expected_symbols: For coverage check

        Returns:
            Dict with all results
        """
        logger.info("Running comprehensive data quality checks...")

        results = {
            'timestamp': datetime.now(),
            'data_type': data_type.value,
            'row_count': len(df),
            'checks': []
        }

        # 1. Schema
        results['checks'].append(self.check_schema(df, data_type))

        # 2. Monotonicity
        if 'date' in df.columns or 'ts_ns' in df.columns:
            ts_col = 'date' if 'date' in df.columns else 'ts_ns'
            results['checks'].append(self.check_monotonicity(df, ts_col))

        # 3. Outliers
        if 'close' in df.columns:
            results['checks'].append(self.check_outliers(df))

        # 4. Coverage
        if start_date and end_date and 'date' in df.columns:
            results['checks'].append(
                self.check_coverage(df, start_date, end_date, expected_symbols)
            )

        # 5. PIT safety
        if 'date' in df.columns and 'ingested_at' in df.columns:
            results['checks'].append(self.check_pit_safety(df))

        # Summary
        all_passed = all(check['passed'] for check in results['checks'])
        results['overall_status'] = 'PASS' if all_passed else 'FAIL'

        logger.info(f"QC Summary: {results['overall_status']} ({len(results['checks'])} checks)")

        return results


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run data quality checks")
    parser.add_argument("--data", type=str, required=True, help="Path to data parquet")
    parser.add_argument("--type", type=str, required=True,
                        help="Data type (e.g., EQUITY_BARS)")
    parser.add_argument("--start-date", type=str, help="Expected start date")
    parser.add_argument("--end-date", type=str, help="Expected end date")
    parser.add_argument("--output", type=str, default="data_layer/qc/reports")

    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} rows from {args.data}")

    # Initialize checker
    checker = DataQualityChecker()

    # Parse dates if provided
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    # Run checks
    data_type = DataType[args.type]
    results = checker.run_all_checks(df, data_type, start, end)

    # Print results
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Data: {args.data}")
    print(f"Type: {data_type.value}")
    print(f"Rows: {len(df)}")
    print(f"Status: {results['overall_status']}")
    print(f"\nChecks:")

    for check in results['checks']:
        status = "✓ PASS" if check['passed'] else "✗ FAIL"
        print(f"  {status} {check['check']}")

        if 'issues' in check and check['issues']:
            for issue in check['issues']:
                print(f"      - {issue}")

        if 'outliers' in check and check['outliers']:
            print(f"      - {len(check['outliers'])} outliers detected")
            for outlier in check['outliers'][:5]:  # Show first 5
                print(f"        {outlier}")

        if 'gaps' in check and check['gaps']:
            for gap in check['gaps'][:3]:  # Show first 3
                print(f"      - {gap}")

    # Save report
    output_path = f"{args.output}/qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Report saved to {output_path}")
