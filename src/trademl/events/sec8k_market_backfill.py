"""Bounded market-data backfill for SEC 8-K event labels."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Protocol
import uuid

import pandas as pd

from trademl.connectors.base import (
    BudgetBlockedConnectorError,
    PermanentConnectorError,
    RemoteRateLimitConnectorError,
)
from trademl.data_node.archive_schema import normalize_archive_frame
from trademl.data_node.provider_contracts import default_vendor_limits
from trademl.data_node.runtime import build_connector
from trademl.events.form4_labels import (
    Form4LabelConfig,
    needed_form4_market_selection,
    resolve_form4_tradable_at,
)
from trademl.events.sec8k import _timestamp_placebo_candidates
from trademl.events.sec8k_semantic import read_sec8k_item_event_candidates


SEC8K_MARKET_BACKFILL_VERSION = "sec8k_market_backfill_v1"
SEC8K_MARKET_BACKFILL_CANDIDATE_COLUMNS = (
    "event_id",
    "ticker",
    "accepted_at_utc",
    "eligibility_pass",
    "event_type",
)
SEC8K_MARKET_BACKFILL_CANDIDATE_SOURCES = (
    "sec8k_item_events",
    "sec_event_semantic_candidates",
)


class Sec8KMarketConnector(Protocol):
    """Small connector surface required by SEC 8-K market-data backfill."""

    vendor_name: str

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch normalized market data."""


def run_sec8k_market_backfill_from_env(
    *,
    data_root: Path,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    round_trip_cost_bps: float = 50.0,
    limit_events: int | None = None,
    include_timestamp_placebo: bool = True,
    max_fetch_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    daily_symbol_batch_size: int = 100,
    candidate_source: str = "sec8k_item_events",
    target_items: tuple[str, ...] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
) -> dict[str, object]:
    """Backfill SEC 8-K label market data using the configured Alpaca connector."""
    connector = build_connector(
        vendor="alpaca",
        env_values=os.environ,
        vendor_limits=default_vendor_limits(),
    )
    if connector is None:
        raise RuntimeError("Alpaca data credentials required for SEC 8-K market backfill")
    return run_sec8k_market_backfill(
        data_root=data_root,
        connector=connector,
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
        limit_events=limit_events,
        include_timestamp_placebo=include_timestamp_placebo,
        max_fetch_attempts=max_fetch_attempts,
        rate_limit_pause_seconds=rate_limit_pause_seconds,
        daily_symbol_batch_size=daily_symbol_batch_size,
        candidate_source=candidate_source,
        target_items=target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
    )


def run_sec8k_market_backfill(
    *,
    data_root: Path,
    connector: Sec8KMarketConnector,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    round_trip_cost_bps: float = 50.0,
    limit_events: int | None = None,
    include_timestamp_placebo: bool = True,
    max_fetch_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    daily_symbol_batch_size: int = 100,
    candidate_source: str = "sec8k_item_events",
    sleep_fn: Callable[[float], None] = time.sleep,
    target_items: tuple[str, ...] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
) -> dict[str, object]:
    """Fetch only market-data slices required by current SEC 8-K candidate labels."""
    root = Path(data_root).expanduser()
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    normalized_candidate_source = _candidate_source(candidate_source)
    candidates_path = _candidate_source_path(root=root, candidate_source=normalized_candidate_source)
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing SEC 8-K candidate parquet: {candidates_path}")
    candidates = (
        read_sec8k_item_event_candidates(candidates_path)
        if normalized_candidate_source == "sec8k_item_events"
        else pd.read_parquet(candidates_path)
    )
    candidates = _filter_backfill_source_candidates(
        candidates=candidates,
        target_items=target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
    )
    primary_candidates = _limited_primary_candidates(
        candidates=candidates,
        limit_events=limit_events,
    )
    backfill_candidates = _market_backfill_candidates(
        primary_candidates=primary_candidates,
        include_timestamp_placebo=include_timestamp_placebo,
    )

    config = Form4LabelConfig(
        horizons=tuple(sorted(set(int(item) for item in horizons))),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    needed = needed_form4_market_selection(candidates=backfill_candidates, config=config)
    minute_symbols_by_date = _minute_symbols_by_date(
        candidates=backfill_candidates,
        config=config,
    )
    candidate_symbols = sorted(needed["symbols"])
    daily_symbols = sorted(set(candidate_symbols) | set(config.benchmark_symbols))
    minute_results: list[dict[str, object]] = []
    daily_results: list[dict[str, object]] = []
    retry_events: list[dict[str, object]] = []
    daily_fetch_batches = 0

    for date_value, symbols_for_date in sorted(minute_symbols_by_date.items()):
        frame, fetch_retry_events = _fetch_market_frame_with_retries(
            connector=connector,
            dataset="equities_minute",
            symbols=symbols_for_date,
            start_date=date_value,
            end_date=date_value,
            max_attempts=max_fetch_attempts,
            rate_limit_pause_seconds=rate_limit_pause_seconds,
            sleep_fn=sleep_fn,
        )
        retry_events.extend(fetch_retry_events)
        paths = _append_partitioned_archive_frame(
            root=root,
            output_name="equities_minute",
            frame=frame,
        )
        rows_by_symbol = _rows_by_symbol(frame)
        for symbol in symbols_for_date:
            symbol_rows = rows_by_symbol.get(symbol, 0)
            minute_results.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "rows": int(symbol_rows),
                    "written_path_summary": _summarize_paths(paths)
                    if symbol_rows
                    else _summarize_paths([]),
                    "status": "available" if symbol_rows else "empty",
                }
            )

    if needed["daily_dates"] and daily_symbols:
        start_date = min(needed["daily_dates"])
        end_date = max(needed["daily_dates"])
        for symbol_batch in _chunked(
            daily_symbols,
            chunk_size=max(1, int(daily_symbol_batch_size)),
        ):
            daily_fetch_batches += 1
            frame, fetch_retry_events = _fetch_market_frame_with_retries(
                connector=connector,
                dataset="equities_eod",
                symbols=symbol_batch,
                start_date=start_date,
                end_date=end_date,
                max_attempts=max_fetch_attempts,
                rate_limit_pause_seconds=rate_limit_pause_seconds,
                sleep_fn=sleep_fn,
            )
            retry_events.extend(fetch_retry_events)
            paths = _append_partitioned_archive_frame(
                root=root,
                output_name="equities_eod",
                frame=frame,
            )
            rows_by_symbol = _rows_by_symbol(frame)
            for symbol in symbol_batch:
                symbol_rows = rows_by_symbol.get(symbol, 0)
                daily_results.append(
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "symbol": symbol,
                        "rows": int(symbol_rows),
                        "written_path_summary": _summarize_paths(paths)
                        if symbol_rows
                        else _summarize_paths([]),
                        "status": "available" if symbol_rows else "empty",
                    }
                )

    empty_minute = [
        {"date": item["date"], "symbol": item["symbol"]}
        for item in minute_results
        if int(item["rows"]) == 0
    ]
    empty_daily_symbols = [
        str(item["symbol"]) for item in daily_results if int(item["rows"]) == 0
    ]
    payload: dict[str, object] = {
        "version": SEC8K_MARKET_BACKFILL_VERSION,
        "checked_at": checked_at,
        "data_root": str(root),
        "vendor": getattr(connector, "vendor_name", "unknown"),
        "candidate_source": normalized_candidate_source,
        "target_items": list(_normalize_target_items(target_items)),
        "accepted_from": _date_bound(accepted_from),
        "accepted_to": _date_bound(accepted_to),
        "candidate_symbols": candidate_symbols,
        "daily_symbols": daily_symbols,
        "include_timestamp_placebo": include_timestamp_placebo,
        "max_fetch_attempts": max_fetch_attempts,
        "rate_limit_pause_seconds": rate_limit_pause_seconds,
        "daily_symbol_batch_size": daily_symbol_batch_size,
        "daily_fetch_batches": daily_fetch_batches,
        "primary_candidate_count": int(len(primary_candidates)),
        "eligible_primary_candidate_count": int(
            primary_candidates["eligibility_pass"].fillna(False).astype(bool).sum()
        )
        if "eligibility_pass" in primary_candidates
        else 0,
        "backfill_candidate_count": int(len(backfill_candidates)),
        "eligible_backfill_candidate_count": int(
            backfill_candidates["eligibility_pass"].fillna(False).astype(bool).sum()
        )
        if "eligibility_pass" in backfill_candidates
        else 0,
        "timestamp_placebo_candidate_count": int(
            max(0, len(backfill_candidates) - len(primary_candidates))
        ),
        "needed_minute_dates": sorted(needed["minute_dates"]),
        "minute_symbol_date_request_count": int(
            sum(len(symbols) for symbols in minute_symbols_by_date.values())
        ),
        "minute_symbol_counts_by_date": {
            date_value: len(symbols)
            for date_value, symbols in sorted(minute_symbols_by_date.items())
        },
        "needed_daily_dates": sorted(needed["daily_dates"]),
        "minute": minute_results,
        "daily": daily_results,
        "retry_events": retry_events,
        "retry_event_count": len(retry_events),
        "minute_rows": sum(int(item["rows"]) for item in minute_results),
        "daily_rows": sum(int(item["rows"]) for item in daily_results),
        "empty_minute": empty_minute,
        "empty_daily_symbols": empty_daily_symbols,
        "verdict": _coverage_verdict(
            candidate_symbols=candidate_symbols,
            empty_minute=empty_minute,
            empty_daily_symbols=empty_daily_symbols,
        ),
    }
    _write_backfill_artifact(root=root, payload=payload)
    return payload


def _limited_primary_candidates(
    *, candidates: pd.DataFrame, limit_events: int | None
) -> pd.DataFrame:
    if limit_events is None or candidates.empty:
        return _market_backfill_candidate_frame(candidates)
    eligible_ids = (
        candidates[candidates["eligibility_pass"].fillna(False).astype(bool)]
        .sort_values("event_id")
        .head(max(0, int(limit_events)))["event_id"]
        .astype(str)
        .tolist()
    )
    selected = candidates[candidates["event_id"].astype(str).isin(eligible_ids)].copy()
    return _market_backfill_candidate_frame(selected)


def _candidate_source(value: str) -> str:
    text = str(value or "sec8k_item_events").strip()
    if text not in SEC8K_MARKET_BACKFILL_CANDIDATE_SOURCES:
        raise ValueError(
            "candidate_source must be one of "
            + ", ".join(SEC8K_MARKET_BACKFILL_CANDIDATE_SOURCES)
        )
    return text


def _candidate_source_path(*, root: Path, candidate_source: str) -> Path:
    if candidate_source == "sec_event_semantic_candidates":
        return root / "data" / "curated" / "events" / "sec_event_semantic_candidates" / "data.parquet"
    return root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"


def _filter_backfill_source_candidates(
    *,
    candidates: pd.DataFrame,
    target_items: tuple[str, ...] | None,
    accepted_from: str | None,
    accepted_to: str | None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    frame = candidates.copy()
    targets = set(_normalize_target_items(target_items))
    if targets and "sec_item_number" in frame:
        items = frame["sec_item_number"].map(_normalize_sec_item)
        frame = frame[items.isin(targets)].copy()
    lower = _date_bound(accepted_from)
    upper = _date_bound(accepted_to)
    if lower or upper:
        dates = frame.apply(_candidate_event_date, axis=1)
        mask = pd.Series(True, index=frame.index)
        if lower:
            mask &= dates >= lower
        if upper:
            mask &= dates <= upper
        frame = frame[mask.fillna(False)].copy()
    return frame


def _normalize_target_items(items: tuple[str, ...] | None) -> tuple[str, ...]:
    if not items:
        return ()
    return tuple(dict.fromkeys(item for item in (_normalize_sec_item(value) for value in items) if item))


def _normalize_sec_item(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"([1-9])\.(\d{1,2})", text)
    if not match:
        return text
    return f"{int(match.group(1))}.{int(match.group(2)):02d}"


def _candidate_event_date(row: pd.Series) -> str:
    for column in (
        "accepted_at_utc",
        "accepted_at",
        "acceptanceDateTime",
        "filing_date",
        "filed_date",
        "filingDate",
    ):
        if column not in row:
            continue
        text = _date_text(row.get(column))
        if text:
            return text
    return ""


def _date_bound(value: object) -> str | None:
    text = _date_text(value)
    return text or None


def _date_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat() if value.tzinfo else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        year, month, day = text[:4], text[5:7], text[8:10]
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{year}-{month}-{day}"
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _market_backfill_candidates(
    *,
    primary_candidates: pd.DataFrame,
    include_timestamp_placebo: bool,
) -> pd.DataFrame:
    primary = _market_backfill_candidate_frame(primary_candidates)
    if not include_timestamp_placebo or primary.empty:
        return primary
    placebo = _market_backfill_candidate_frame(_timestamp_placebo_candidates(primary))
    combined = pd.concat([primary, placebo], ignore_index=True)
    return combined.drop_duplicates(subset=["event_id"], keep="last")


def _minute_symbols_by_date(
    *, candidates: pd.DataFrame, config: Form4LabelConfig
) -> dict[str, list[str]]:
    """Return exact symbol batches required for each SEC 8-K entry-minute date."""
    if candidates.empty:
        return {}
    output: dict[str, set[str]] = {}
    for event in candidates.sort_values("event_id").to_dict("records"):
        if not bool(event.get("eligibility_pass")) or not event.get("accepted_at_utc"):
            continue
        symbol = str(event.get("ticker") or "").strip()
        if not symbol:
            continue
        tradable_at = resolve_form4_tradable_at(
            str(event["accepted_at_utc"]),
            latency_minutes=config.latency_minutes,
            exchange=config.exchange,
        )
        output.setdefault(str(tradable_at.date()), set()).add(symbol)
    return {date_value: sorted(symbols) for date_value, symbols in sorted(output.items())}


def _market_backfill_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact candidate fields needed to plan market backfill."""
    if frame.empty:
        return pd.DataFrame(columns=SEC8K_MARKET_BACKFILL_CANDIDATE_COLUMNS)
    selected = frame.reindex(columns=SEC8K_MARKET_BACKFILL_CANDIDATE_COLUMNS).copy()
    selected["event_id"] = selected["event_id"].fillna("").astype(str)
    selected["ticker"] = selected["ticker"].fillna("").astype(str)
    selected["accepted_at_utc"] = selected["accepted_at_utc"].fillna("").astype(str)
    selected["event_type"] = selected["event_type"].fillna("").astype(str)
    selected["eligibility_pass"] = selected["eligibility_pass"].fillna(False).astype(bool)
    return selected


def _fetch_market_frame_with_retries(
    *,
    connector: Sec8KMarketConnector,
    dataset: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    max_attempts: int,
    rate_limit_pause_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Fetch one market-data slice, pausing on local or remote rate-limit pressure."""
    retry_events: list[dict[str, object]] = []
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return connector.fetch(dataset, symbols, start_date, end_date), retry_events
        except PermanentConnectorError as exc:
            retry_events.append(
                {
                    "attempt": attempt,
                    "dataset": dataset,
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol_count": len(symbols),
                    "symbols": symbols,
                    "reason": "permanent_connector_error",
                    "error": str(exc),
                }
            )
            if len(symbols) <= 1:
                return pd.DataFrame(), retry_events
            frames: list[pd.DataFrame] = []
            for symbol in symbols:
                frame, symbol_retry_events = _fetch_market_frame_with_retries(
                    connector=connector,
                    dataset=dataset,
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    max_attempts=1,
                    rate_limit_pause_seconds=rate_limit_pause_seconds,
                    sleep_fn=sleep_fn,
                )
                retry_events.extend(symbol_retry_events)
                if not frame.empty:
                    frames.append(frame)
            return (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            ), retry_events
        except BudgetBlockedConnectorError as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = _budget_block_sleep_seconds(
                exc,
                default_seconds=rate_limit_pause_seconds,
            )
            retry_events.append(
                {
                    "attempt": attempt,
                    "dataset": dataset,
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol_count": len(symbols),
                    "reason": "local_budget_block",
                    "sleep_seconds": sleep_seconds,
                }
            )
            sleep_fn(sleep_seconds)
        except RemoteRateLimitConnectorError as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = float(exc.retry_after_seconds or rate_limit_pause_seconds)
            retry_events.append(
                {
                    "attempt": attempt,
                    "dataset": dataset,
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol_count": len(symbols),
                    "reason": "remote_rate_limit",
                    "sleep_seconds": sleep_seconds,
                }
            )
            sleep_fn(sleep_seconds)
    raise RuntimeError("unreachable SEC 8-K market fetch retry state")


def _budget_block_sleep_seconds(
    exc: BudgetBlockedConnectorError, *, default_seconds: float
) -> float:
    """Return a positive sleep duration from a local budget block decision."""
    next_eligible = exc.decision.next_eligible_at
    if next_eligible is None:
        return float(default_seconds)
    now = datetime.now(timezone.utc)
    if next_eligible.tzinfo is None:
        next_eligible = next_eligible.replace(tzinfo=timezone.utc)
    return max(0.1, (next_eligible - now).total_seconds())


def _append_partitioned_archive_frame(
    *,
    root: Path,
    output_name: str,
    frame: pd.DataFrame,
) -> list[Path]:
    if frame.empty:
        return []
    normalized = normalize_archive_frame(output_name, frame)
    archive_root = Path(root) / "data" / "raw" / output_name
    written: list[Path] = []
    for day_value, day_frame in normalized.groupby("date", dropna=True):
        day_key = pd.Timestamp(day_value).strftime("%Y-%m-%d")
        output = archive_root / f"date={day_key}" / "data.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = normalize_archive_frame(output_name, pd.read_parquet(output))
            combined = pd.concat([existing, day_frame], ignore_index=True)
        else:
            combined = day_frame.copy()
        combined = normalize_archive_frame(output_name, combined).drop_duplicates()
        tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output)
        written.append(output)
    return written


def _rows_by_symbol(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "symbol" not in frame:
        return {}
    counts = frame["symbol"].astype(str).value_counts().to_dict()
    return {str(symbol): int(count) for symbol, count in counts.items()}


def _chunked(items: list[str], *, chunk_size: int) -> list[list[str]]:
    """Split a list into stable, non-empty chunks."""
    normalized_size = max(1, int(chunk_size))
    return [
        items[index : index + normalized_size]
        for index in range(0, len(items), normalized_size)
    ]


def _summarize_paths(paths: list[Path], *, sample_size: int = 5) -> dict[str, object]:
    """Return a compact path summary for operator artifacts."""
    path_text = [str(path) for path in paths]
    return {
        "count": len(path_text),
        "sample": path_text[:sample_size],
        "last": path_text[-1] if path_text else None,
    }


def _coverage_verdict(
    *,
    candidate_symbols: list[str],
    empty_minute: list[dict[str, object]],
    empty_daily_symbols: list[str],
) -> str:
    if not candidate_symbols:
        return "NO_ELIGIBLE_CANDIDATES"
    if empty_minute or empty_daily_symbols:
        return "PARTIAL_COVERAGE"
    return "PASS"


def _write_backfill_artifact(*, root: Path, payload: dict[str, object]) -> None:
    target = Path(root) / "control" / "cluster" / "state" / "research" / "sec8k_market_backfill"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(payload["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
