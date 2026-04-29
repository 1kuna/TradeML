"""Dataset schema registry for append-safe raw/archive parquet writes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True, frozen=True)
class ArchiveDatasetSchema:
    """Schema coercion rules for one partitioned archive dataset."""

    string_columns: tuple[str, ...] = ()
    timestamp_columns: tuple[str, ...] = ()
    date_column: str = "date"


ARCHIVE_SCHEMAS: dict[str, ArchiveDatasetSchema] = {
    "equities_minute": ArchiveDatasetSchema(
        string_columns=("source_name", "symbol", "vendor_ts", "currency"),
        timestamp_columns=("timestamp", "vendor_ts", "ingested_at"),
    ),
    "ticker_news": ArchiveDatasetSchema(
        string_columns=(
            "source_name",
            "symbol",
            "symbols",
            "news_id",
            "id",
            "url",
            "headline",
            "summary",
            "source",
            "author",
            "image_url",
        ),
        timestamp_columns=("published_at", "crawled_at", "ingested_at"),
    ),
}


def normalize_archive_frame(output_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce an archive frame to the registry schema before append."""
    normalized = frame.copy()
    if normalized.empty:
        return normalized
    schema = ARCHIVE_SCHEMAS.get(output_name, ArchiveDatasetSchema())
    if schema.date_column in normalized.columns:
        normalized[schema.date_column] = pd.to_datetime(
            normalized[schema.date_column], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    for column in schema.string_columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("string")
    for column in schema.timestamp_columns:
        if column in normalized.columns and column not in schema.string_columns:
            normalized[column] = pd.to_datetime(
                normalized[column], errors="coerce", utc=True
            )
    return normalized
