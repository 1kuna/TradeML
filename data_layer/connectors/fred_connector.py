"""
FRED (Federal Reserve Economic Data) connector for risk-free rates and macro data.

Free tier: Unlimited requests (rate-limited)
API Docs: https://fred.stlouisfed.org/docs/api/

Supports:
- Treasury rates (DGS1, DGS10, etc.)
- ALFRED vintages (real-time vs revised data)
- Macro time series (GDP, CPI, etc.)
"""

import os
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectorError
from ..schemas import DataType, get_schema


class FREDConnector(BaseConnector):
    """
    Connector for FRED economic data.

    Provides:
    - Risk-free rates (Treasuries)
    - Macro time series
    - ALFRED vintages for point-in-time accuracy
    """

    API_URL = "https://api.stlouisfed.org/fred"

    # Common series IDs
    TREASURY_SERIES = {
        "1m": "DGS1MO",    # 1-Month Treasury
        "3m": "DGS3MO",    # 3-Month Treasury
        "6m": "DGS6MO",    # 6-Month Treasury
        "1y": "DGS1",      # 1-Year Treasury
        "2y": "DGS2",      # 2-Year Treasury
        "5y": "DGS5",      # 5-Year Treasury
        "10y": "DGS10",    # 10-Year Treasury
        "30y": "DGS30",    # 30-Year Treasury
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_sec: float = 1.7,  # ~102 rpm (85% of 120 rpm heuristic)
    ):
        """
        Initialize FRED connector.

        Args:
            api_key: FRED API key (or from environment)
            rate_limit_per_sec: Max requests per second
        """
        api_key = api_key or os.getenv("FRED_API_KEY")

        if not api_key:
            raise ConnectorError(
                "FRED API key not found. Set FRED_API_KEY in .env"
            )

        super().__init__(
            source_name="fred",
            api_key=api_key,
            base_url=self.API_URL,
            rate_limit_per_sec=rate_limit_per_sec,
        )

        logger.info("FRED connector initialized")

    def _fetch_raw(
        self,
        endpoint: str,
        series_id: Optional[str] = None,
        **params
    ) -> Dict:
        """
        Fetch raw data from FRED API.

        Args:
            endpoint: API endpoint (series/observations, series/info, etc.)
            series_id: FRED series ID
            **params: Additional API parameters

        Returns:
            Raw JSON response
        """
        url = f"{self.API_URL}/{endpoint}"

        api_params = {
            "api_key": self.api_key,
            "file_type": "json",
            **params
        }

        if series_id:
            api_params["series_id"] = series_id

        try:
            response = self._get(url, params=api_params)
            data = response.json()

            # Check for API errors
            if "error_code" in data:
                raise ConnectorError(f"FRED error: {data.get('error_message', 'Unknown error')}")

            return data

        except Exception as e:
            logger.error(f"FRED API error: {e}")
            raise ConnectorError(f"Failed to fetch from FRED: {e}")

    def _transform(
        self,
        raw_data: Dict,
        series_id: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform FRED observations to our schema.

        Args:
            raw_data: Raw API response
            series_id: FRED series ID

        Returns:
            DataFrame conforming to MACRO_SCHEMA
        """
        observations = raw_data.get("observations", [])

        rows = []
        for obs in observations:
            # Skip missing values
            if obs["value"] == ".":
                continue

            row = {
                "series_id": series_id,
                "date": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                "value": float(obs["value"]),
                "vintage_date": None,  # Set by ALFRED if needed
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch time series observations.

        Args:
            series_id: FRED series ID (e.g., 'DGS10')
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with time series data
        """
        logger.info(f"Fetching FRED series {series_id}")

        params = {}
        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")

        raw_data = self._fetch_raw(
            endpoint="series/observations",
            series_id=series_id,
            **params
        )

        df = self._transform(raw_data, series_id=series_id)

        if not df.empty:
            source_uri = f"fred://series/{series_id}"
            df = self._add_metadata(df, source_uri=source_uri)

        return df

    def fetch_treasury_curve(
        self,
        tenors: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch entire Treasury yield curve.

        Args:
            tenors: List of tenors (e.g., ['1m', '3m', '10y']) or None for all
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with all Treasury rates
        """
        if tenors is None:
            tenors = list(self.TREASURY_SERIES.keys())

        logger.info(f"Fetching Treasury curve for tenors: {tenors}")

        all_series = []

        for tenor in tenors:
            if tenor not in self.TREASURY_SERIES:
                logger.warning(f"Unknown tenor: {tenor}, skipping")
                continue

            series_id = self.TREASURY_SERIES[tenor]
            df = self.fetch_series(series_id, start_date, end_date)

            if not df.empty:
                df["tenor"] = tenor
                all_series.append(df)

        if not all_series:
            return pd.DataFrame()

        # Combine all series
        combined = pd.concat(all_series, ignore_index=True)
        combined = combined.sort_values(["date", "tenor"]).reset_index(drop=True)

        return combined

    def fetch_alfred_vintages(
        self,
        series_id: str,
        vintage_dates: Optional[List[date]] = None,
    ) -> pd.DataFrame:
        """
        Fetch ALFRED vintages (real-time vs revised data).

        Args:
            series_id: FRED series ID
            vintage_dates: List of vintage dates to fetch

        Returns:
            DataFrame with vintages
        """
        logger.info(f"Fetching ALFRED vintages for {series_id}")

        if vintage_dates is None:
            # Get all available vintages
            raw_data = self._fetch_raw(
                endpoint="series/vintagedates",
                series_id=series_id
            )
            vintage_dates_str = raw_data.get("vintage_dates", [])
            vintage_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in vintage_dates_str]

        all_vintages = []

        for vintage_date in vintage_dates:
            params = {
                "vintage_dates": vintage_date.strftime("%Y-%m-%d")
            }

            raw_data = self._fetch_raw(
                endpoint="series/observations",
                series_id=series_id,
                **params
            )

            df = self._transform(raw_data, series_id=series_id)

            if not df.empty:
                df["vintage_date"] = vintage_date
                all_vintages.append(df)

        if not all_vintages:
            return pd.DataFrame()

        combined = pd.concat(all_vintages, ignore_index=True)
        combined = combined.sort_values(["date", "vintage_date"]).reset_index(drop=True)

        source_uri = f"fred://alfred/{series_id}"
        combined = self._add_metadata(combined, source_uri=source_uri)

        return combined

    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata about a series.

        Args:
            series_id: FRED series ID

        Returns:
            Dict with series metadata
        """
        logger.info(f"Fetching info for {series_id}")

        raw_data = self._fetch_raw(
            endpoint="series",
            series_id=series_id
        )

        return raw_data.get("seriess", [{}])[0]


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch FRED economic data")
    parser.add_argument("--series", type=str, help="FRED series ID (e.g., DGS10)")
    parser.add_argument("--treasury", action="store_true", help="Fetch full Treasury curve")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data_layer/reference/macro")

    args = parser.parse_args()

    connector = FREDConnector()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    if args.treasury:
        # Fetch Treasury curve
        df = connector.fetch_treasury_curve(start_date=start, end_date=end)
        if not df.empty:
            output_path = f"{args.output}/treasury_curve.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.MACRO),
            )
            print(f"[OK] Fetched {len(df)} Treasury observations to {output_path}")
    elif args.series:
        # Fetch specific series
        df = connector.fetch_series(args.series, start_date=start, end_date=end)
        if not df.empty:
            output_path = f"{args.output}/{args.series}.parquet"
            connector.write_parquet(
                df,
                path=output_path,
                schema=get_schema(DataType.MACRO),
            )
            print(f"[OK] Fetched {len(df)} observations for {args.series} to {output_path}")
        else:
            print(f"[WARN] No data found for {args.series}")
    else:
        print("Error: --series or --treasury required")
