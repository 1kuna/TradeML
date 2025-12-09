"""
Base connector class with retry logic, checksums, and lineage tracking.

All data connectors must inherit from BaseConnector and implement:
- _fetch_raw(): fetch data from vendor API
- _transform(): transform raw payload to our schema
"""

import hashlib
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import NameResolutionError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from utils.pacing import RequestPacer


class BaseConnector(ABC):
    """
    Base class for all data connectors.

    Provides:
    - HTTP retry with exponential backoff
    - Response checksum generation
    - Metadata tracking (ingested_at, source_uri, source_name)
    - Parquet writing with schema validation
    """

    def __init__(
        self,
        source_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit_per_sec: float = 2.0,
        max_retries: int = 5,
    ):
        """
        Initialize connector.

        Args:
            source_name: Name of data source (e.g., 'alpaca', 'iex')
            api_key: API key for authentication
            base_url: Base URL for API
            rate_limit_per_sec: Max requests per second
            max_retries: Max retry attempts on failure
        """
        self.source_name = source_name
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit_per_sec = rate_limit_per_sec
        self.max_retries = max_retries

        # Local fallback rate limiting (per instance); global pacer is preferred
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / rate_limit_per_sec

        # Setup HTTP session with retry
        self.session = self._create_session()

        logger.info(f"Initialized {source_name} connector")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()

        # Retry strategy: retry on 429, 500, 502, 503, 504
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def reset_session(self):
        """Rebuild the HTTP session after network faults to force new connections."""
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._create_session()
        logger.warning(f"{self.source_name}: HTTP session reset after network failure")

    def _rate_limit(self):
        """Enforce rate limiting between requests (global pacer + local fallback)."""
        # First enforce per-vendor RPM via budget manager (token bucket)
        try:
            from data_node.budgets import get_budget_manager
            from data_node.db import TaskKind

            bm = get_budget_manager()
            # Block until a token is available for this vendor
            while not bm.try_spend(self.source_name, TaskKind.FORWARD):
                status = bm.get_budget_status(self.source_name)
                hard_rpm = status.get("hard_rpm", 1) or 1
                wait_step = max(0.2, 60.0 / max(hard_rpm, 1))
                logger.debug(f"{self.source_name}: waiting for budget token ({status.get('tokens_rpm', 0):.2f} tokens)")
                time.sleep(wait_step)
        except Exception:
            # Budget tracking optional - do not block fetch if budget manager unavailable
            pass

        try:
            RequestPacer.instance().acquire(self.source_name, self.rate_limit_per_sec)
            # Update local timestamp for completeness
            self._last_request_time = time.time()
            return
        except Exception:
            pass
        # Fallback to simple per-instance spacing if pacer unavailable
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response:
        """
        Make GET request with rate limiting and retry.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: on failure after retries
        """
        self._rate_limit()

        import time, random
        logger.debug(f"HTTP GET begin: {url} params={params}")
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            logger.debug(f"HTTP GET status: {response.status_code} for {url}")

            # Track budget per HTTP request BEFORE checking status
            # This counts ALL API calls (success, 429, 5xx) for accurate rate limiting
            try:
                from data_node.budgets import get_budget_manager
                get_budget_manager().spend(self.source_name)
            except Exception:
                pass  # Budget tracking optional - don't fail requests

            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = int(retry_after)
                    except Exception:
                        sleep_s = self._min_request_interval * (1 + random.random())
                else:
                    sleep_s = min(60, (self._min_request_interval or 1.0) * (2 + random.random()))
                logger.warning(f"429 rate limited for {url}; sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                logger.debug(f"HTTP GET retry status: {response.status_code} for {url}")
                # Count retry attempt in budget too
                try:
                    from data_node.budgets import get_budget_manager
                    get_budget_manager().spend(self.source_name)
                except Exception:
                    pass

            # Handle server errors (5xx) with retry
            if response.status_code >= 500:
                sleep_s = min(30, 2 * (1 + random.random()))  # 2-4s with jitter
                logger.warning(f"{response.status_code} server error for {url}; sleeping {sleep_s:.2f}s and retrying")
                time.sleep(sleep_s)
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                logger.debug(f"HTTP GET retry status: {response.status_code} for {url}")
                # Count retry attempt in budget too
                try:
                    from data_node.budgets import get_budget_manager
                    get_budget_manager().spend(self.source_name)
                except Exception:
                    pass

            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if isinstance(getattr(e, "reason", None), NameResolutionError) or "Failed to resolve" in str(e):
                # DNS failures can cascade; slow down hard so caller can decide to freeze the vendor.
                sleep_s = min(30, 2 * (1 + random.random()))
                logger.error(f"Name resolution failure for {url}; sleeping {sleep_s:.1f}s before raising")
                time.sleep(sleep_s)
            logger.error(f"Request failed for {url}: {e}")
            raise

    def _post(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response:
        """Make POST request with rate limiting and retry."""
        self._rate_limit()

        try:
            response = self.session.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed for {url}: {e}")
            raise

    @staticmethod
    def _compute_checksum(data: bytes) -> str:
        """
        Compute SHA256 checksum of raw data.

        Args:
            data: Raw bytes

        Returns:
            Hex digest string
        """
        return hashlib.sha256(data).hexdigest()

    @abstractmethod
    def _fetch_raw(self, **kwargs) -> Any:
        """
        Fetch raw data from vendor API.

        Must be implemented by subclasses.

        Returns:
            Raw response data (dict, list, bytes, etc.)
        """
        pass

    @abstractmethod
    def _transform(self, raw_data: Any, **kwargs) -> pd.DataFrame:
        """
        Transform raw vendor data to our schema.

        Must be implemented by subclasses.

        Args:
            raw_data: Output from _fetch_raw()
            **kwargs: Additional context (e.g., symbol, date range)

        Returns:
            pandas DataFrame conforming to schema
        """
        pass

    def _add_metadata(
        self,
        df: pd.DataFrame,
        source_uri: str,
        transform_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add metadata columns for lineage tracking.

        Args:
            df: DataFrame to augment
            source_uri: API endpoint or file path
            transform_id: Hash of transformation logic (optional)

        Returns:
            DataFrame with metadata columns
        """
        df = df.copy()
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
        df["source_name"] = self.source_name
        df["source_uri"] = source_uri

        if transform_id:
            df["transform_id"] = transform_id

        return df

    def fetch_and_transform(self, **kwargs) -> pd.DataFrame:
        """
        Fetch raw data and transform to schema (template method).

        Args:
            **kwargs: Parameters for _fetch_raw and _transform

        Returns:
            Transformed DataFrame with metadata
        """
        logger.info(f"Fetching data from {self.source_name}...")

        # Fetch raw data
        raw_data = self._fetch_raw(**kwargs)

        # Transform to schema
        source_uri = kwargs.get("source_uri", self.base_url or self.source_name)
        df = self._transform(raw_data, **kwargs)

        # Add metadata
        df = self._add_metadata(df, source_uri=source_uri)

        logger.info(f"Fetched and transformed {len(df)} rows from {self.source_name}")
        return df

    def write_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
        schema: Optional[pa.Schema] = None,
        partition_cols: Optional[list] = None,
    ):
        """
        Write DataFrame to Parquet with optional schema validation.

        Args:
            df: DataFrame to write
            path: Output path (file or directory if partitioned)
            schema: PyArrow schema for validation (optional)
            partition_cols: Columns to partition by (e.g., ['date', 'symbol'])
        """
        # If configured, write to S3 instead of local filesystem
        storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()
        if storage_backend == "s3":
            try:
                from data_layer.storage.s3_client import get_s3_client
            except Exception as e:
                logger.error(f"S3 backend requested but unavailable: {e}")
                storage_backend = "local"

        if storage_backend == "s3":
            self._write_parquet_s3(df, path, schema=schema, partition_cols=partition_cols)
            return

        # Local filesystem write
        path = Path(path)

        if partition_cols:
            part_cols = [c for c in partition_cols if c in df.columns]
            if not part_cols:
                logger.warning("Partition columns not found in DataFrame; writing unpartitioned")
                partition_cols = None
            else:
                for _, sub in df.groupby(part_cols, dropna=False, as_index=False):
                    # Build partition directory path
                    part_path = Path(path)
                    for c in part_cols:
                        v = sub.iloc[0][c]
                        if hasattr(v, "isoformat"):
                            v = v.isoformat()
                        part_path = part_path / f"{c}={v}"

                    part_path.mkdir(parents=True, exist_ok=True)
                    tmp_file = part_path / f".tmp-{uuid.uuid4().hex}.parquet"
                    final_file = part_path / "data.parquet"

                    table = pa.Table.from_pandas(sub, schema=schema, preserve_index=False)
                    try:
                        pq.write_table(table, str(tmp_file), compression="snappy")
                        os.replace(tmp_file, final_file)  # Atomic on same filesystem
                    finally:
                        if tmp_file.exists():
                            try:
                                tmp_file.unlink()
                            except Exception:
                                pass
                logger.info(f"Wrote partitioned Parquet to {path}")
                return

        # Fallback: single file
        target_dir = path.parent if path.suffix else path
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = target_dir / f".tmp-{uuid.uuid4().hex}{''.join(path.suffixes) if path.suffixes else '.parquet'}"
        final_file = path if path.suffix else target_dir / "data.parquet"

        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        try:
            pq.write_table(table, str(tmp_file), compression="snappy")
            os.replace(tmp_file, final_file)
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
        logger.info(f"Wrote Parquet to {final_file}")

    def _write_parquet_s3(
        self,
        df: pd.DataFrame,
        path: Path | str,
        schema: Optional[pa.Schema] = None,
        partition_cols: Optional[list] = None,
    ):
        """
        Write DataFrame to S3 using the repo's S3Client.

        Notes:
            - Maps local-style roots under data_layer/ to S3 prefixes:
              data_layer/raw -> raw, data_layer/curated -> curated, otherwise reference
            - When partition_cols are provided, writes one object per partition
              at <root>/<col1>=<v1>/.../data.parquet
        """
        from data_layer.storage.s3_client import get_s3_client

        s3 = get_s3_client()
        path_str = str(path)

        # Normalize root mapping
        if path_str.startswith("data_layer/raw"):
            root = path_str.replace("data_layer/raw", "raw", 1)
        elif path_str.startswith("data_layer/curated"):
            root = path_str.replace("data_layer/curated", "curated", 1)
        elif path_str.startswith("s3://"):
            # Allow explicit s3://bucket/prefix (ignore client bucket)
            # Strip scheme and bucket if it matches current to get key prefix
            # Expected form: s3://<bucket>/<prefix>
            parts = path_str[5:].split("/", 1)
            root = parts[1] if len(parts) > 1 else ""
        else:
            root = path_str

        # Convert to bytes once per partition
        if partition_cols:
            # Partition and write per unique combination
            part_cols = [c for c in partition_cols if c in df.columns]
            if not part_cols:
                logger.warning("Partition columns not found in DataFrame; writing unpartitioned")
                partition_cols = None
            else:
                grouped = df.groupby(part_cols, dropna=False, as_index=False)
                for _, sub in grouped:
                    # Build key like root/col1=v1/col2=v2/data.parquet
                    sub_key = root.rstrip("/")
                    for c in part_cols:
                        v = sub.iloc[0][c]
                        # Format dates
                        if hasattr(v, 'isoformat'):
                            v = v.isoformat()
                        sub_key += f"/{c}={v}"

                    buf = pa.BufferOutputStream()
                    table = pa.Table.from_pandas(sub, schema=schema, preserve_index=False)
                    pq.write_table(table, buf, compression="snappy")
                    s3.put_object(key=f"{sub_key}/data.parquet", data=buf.getvalue().to_pybytes())
                logger.info(f"Wrote partitioned dataset to s3://{s3.bucket}/{root}")
                return

        # Fallback: single object
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf, compression="snappy")

        # If path ends with .parquet, keep it; else use data.parquet
        key = root
        if not key.endswith(".parquet"):
            key = key.rstrip("/") + "/data.parquet"
        s3.put_object(key=key, data=buf.getvalue().to_pybytes())
        logger.info(f"Wrote Parquet to s3://{s3.bucket}/{key}")

    def validate_response(self, response: requests.Response, expected_keys: Optional[list] = None):
        """
        Validate JSON response contains expected keys.

        Args:
            response: Response object
            expected_keys: List of required keys in JSON

        Raises:
            ValueError: if validation fails
        """
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        if expected_keys:
            missing = set(expected_keys) - set(data.keys())
            if missing:
                raise ValueError(f"Response missing required keys: {missing}")

        return data


class ConnectorError(Exception):
    """Base exception for connector errors."""
    pass


class RateLimitError(ConnectorError):
    """Raised when rate limit is exceeded."""
    pass


class DataQualityError(ConnectorError):
    """Raised when fetched data fails QC checks."""
    pass
