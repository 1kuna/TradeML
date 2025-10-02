import os
import io
import uuid
from datetime import datetime
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("S3_ENDPOINT"),
    reason="Requires S3/MinIO endpoint in environment",
)


def put_parquet_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def test_curator_idempotent(tmp_path):
    from data_layer.storage.s3_client import get_s3_client
    from scripts.curator import Curator

    os.environ["STORAGE_BACKEND"] = "s3"
    s3 = get_s3_client()

    # Unique prefixes for test isolation
    prefix_raw = f"raw/it_{uuid.uuid4().hex}/equities_bars"
    prefix_out = f"curated/it_{uuid.uuid4().hex}/equities_ohlcv_adj"

    # Create one date partition
    ds = datetime.utcnow().date().isoformat()
    df = pd.DataFrame([
        {"date": ds, "symbol": "AAPL", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ])
    s3.put_object(f"{prefix_raw}/date={ds}/data.parquet", put_parquet_bytes(df))

    # Write config file
    cfg = tmp_path / "curator.yml"
    cfg.write_text(
        f"""
watermark:
  bookmark_key: manifests/it_watermarks_{uuid.uuid4().hex}.json
jobs:
  - name: equities_bars_ohlcv
    source: alpaca
    table: equities_bars
    input_prefix: {prefix_raw}
    output_prefix: {prefix_out}
    partition: date
    idempotent: true
"""
    )

    c = Curator(str(cfg))
    c.run()

    # Run again: should not duplicate or error
    c.run()

    # Verify curated object exists once
    assert s3.object_exists(f"{prefix_out}/date={ds}/data.parquet") is True

