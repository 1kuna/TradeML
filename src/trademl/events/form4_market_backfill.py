"""Bounded market-data backfill for Form 4 event labels."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Callable, Protocol
import uuid

import pandas as pd

from trademl.connectors.base import (
    BudgetBlockedConnectorError,
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
from trademl.events.form4_event_study import build_form4_control_candidates_from_curated


FORM4_MARKET_BACKFILL_VERSION = "form4_market_backfill_v1"
MARKET_BACKFILL_CANDIDATE_COLUMNS = (
    "event_id",
    "ticker",
    "accepted_at_utc",
    "eligibility_pass",
    "event_type",
    "control_family",
)


class Form4MarketConnector(Protocol):
    """Small connector surface required by Form 4 market-data backfill."""

    vendor_name: str

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch normalized market data."""


def run_form4_market_backfill_from_env(
    *,
    data_root: Path,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    round_trip_cost_bps: float = 50.0,
    limit_events: int | None = None,
    include_controls: bool = True,
    max_fetch_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    daily_symbol_batch_size: int = 100,
) -> dict[str, object]:
    """Backfill Form 4 label market data using the configured Alpaca data connector."""
    connector = build_connector(
        vendor="alpaca",
        env_values=os.environ,
        vendor_limits=default_vendor_limits(),
    )
    if connector is None:
        raise RuntimeError("Alpaca data credentials required for Form 4 market backfill")
    return run_form4_market_backfill(
        data_root=data_root,
        connector=connector,
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
        limit_events=limit_events,
        include_controls=include_controls,
        max_fetch_attempts=max_fetch_attempts,
        rate_limit_pause_seconds=rate_limit_pause_seconds,
        daily_symbol_batch_size=daily_symbol_batch_size,
    )


def run_form4_market_backfill(
    *,
    data_root: Path,
    connector: Form4MarketConnector,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    round_trip_cost_bps: float = 50.0,
    limit_events: int | None = None,
    include_controls: bool = True,
    max_fetch_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    daily_symbol_batch_size: int = 100,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Fetch only market-data slices required by current Form 4 candidate labels."""
    root = Path(data_root).expanduser()
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    candidates_path = (
        root
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing Form 4 candidate event parquet: {candidates_path}")
    candidates = pd.read_parquet(candidates_path)
    primary_candidates = candidates.copy()
    if limit_events is not None:
        eligible_ids = (
            primary_candidates[
                primary_candidates["eligibility_pass"].fillna(False).astype(bool)
            ]
            .sort_values("event_id")
            .head(max(0, int(limit_events)))["event_id"]
            .tolist()
        )
        primary_candidates = primary_candidates[
            (~primary_candidates["eligibility_pass"].fillna(False).astype(bool))
            | primary_candidates["event_id"].isin(eligible_ids)
        ].copy()
    candidates = _market_backfill_candidates(
        root=root,
        primary_candidates=primary_candidates,
        include_controls=include_controls,
    )

    config = Form4LabelConfig(
        horizons=tuple(sorted(set(int(item) for item in horizons))),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    needed = needed_form4_market_selection(candidates=candidates, config=config)
    minute_symbols_by_date = _minute_symbols_by_date(candidates=candidates, config=config)
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
            daily_symbols, chunk_size=max(1, int(daily_symbol_batch_size))
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
            if frame.empty:
                daily_results.append(
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "symbols": symbol_batch,
                        "rows": 0,
                        "written_path_summary": _summarize_paths([]),
                        "status": "empty",
                    }
                )
            else:
                for symbol, group in frame.groupby("symbol"):
                    daily_results.append(
                        {
                            "start_date": start_date,
                            "end_date": end_date,
                            "symbol": str(symbol),
                            "rows": int(len(group)),
                            "written_path_summary": _summarize_paths(paths),
                            "status": "available",
                        }
                    )

    payload: dict[str, object] = {
        "version": FORM4_MARKET_BACKFILL_VERSION,
        "checked_at": checked_at,
        "data_root": str(root),
        "vendor": getattr(connector, "vendor_name", "unknown"),
        "candidate_symbols": candidate_symbols,
        "daily_symbols": daily_symbols,
        "include_controls": include_controls,
        "max_fetch_attempts": max_fetch_attempts,
        "rate_limit_pause_seconds": rate_limit_pause_seconds,
        "daily_symbol_batch_size": daily_symbol_batch_size,
        "daily_fetch_batches": daily_fetch_batches,
        "primary_candidate_count": int(len(primary_candidates)),
        "backfill_candidate_count": int(len(candidates)),
        "control_candidate_count": int(max(0, len(candidates) - len(primary_candidates))),
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
        "empty_minute": [
            {"date": item["date"], "symbol": item["symbol"]}
            for item in minute_results
            if int(item["rows"]) == 0
        ],
        "empty_daily_symbols": [
            item.get("symbol")
            for item in daily_results
            if int(item["rows"]) == 0 and item.get("symbol")
        ],
    }
    _write_backfill_artifact(root=root, payload=payload)
    return payload


def _fetch_market_frame_with_retries(
    *,
    connector: Form4MarketConnector,
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
        except BudgetBlockedConnectorError as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = _budget_block_sleep_seconds(
                exc, default_seconds=rate_limit_pause_seconds
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
    raise RuntimeError("unreachable Form 4 market fetch retry state")


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


def _chunked(items: list[str], *, chunk_size: int) -> list[list[str]]:
    """Split a list into stable, non-empty chunks."""
    normalized_size = max(1, int(chunk_size))
    return [items[index : index + normalized_size] for index in range(0, len(items), normalized_size)]


def _market_backfill_candidates(
    *,
    root: Path,
    primary_candidates: pd.DataFrame,
    include_controls: bool,
) -> pd.DataFrame:
    primary_frame = _market_backfill_candidate_frame(primary_candidates)
    if not include_controls:
        return primary_frame
    control_frames: list[pd.DataFrame] = []
    for family, frame in build_form4_control_candidates_from_curated(root=root).items():
        if frame.empty:
            continue
        control_frames.append(_market_backfill_candidate_frame(frame, control_family=family))
    if not control_frames:
        return primary_frame
    combined = pd.concat([primary_frame, *control_frames], ignore_index=True)
    if "event_id" in combined:
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    return combined


def _minute_symbols_by_date(
    *, candidates: pd.DataFrame, config: Form4LabelConfig
) -> dict[str, list[str]]:
    """Return exact symbol batches required for each Form 4 entry-minute date."""
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
        date_value = str(tradable_at.date())
        output.setdefault(date_value, set()).add(symbol)
    return {date_value: sorted(symbols) for date_value, symbols in sorted(output.items())}


def _market_backfill_candidate_frame(
    frame: pd.DataFrame, *, control_family: str = ""
) -> pd.DataFrame:
    """Return the compact candidate fields needed to plan market backfill."""
    if frame.empty:
        return pd.DataFrame(columns=MARKET_BACKFILL_CANDIDATE_COLUMNS)
    copy = frame.copy()
    if "control_family" not in copy:
        copy["control_family"] = control_family
    selected = copy.reindex(columns=MARKET_BACKFILL_CANDIDATE_COLUMNS).copy()
    selected["event_id"] = selected["event_id"].fillna("").astype(str)
    selected["ticker"] = selected["ticker"].fillna("").astype(str)
    selected["accepted_at_utc"] = selected["accepted_at_utc"].fillna("").astype(str)
    selected["event_type"] = selected["event_type"].fillna("").astype(str)
    selected["control_family"] = selected["control_family"].fillna("").astype(str)
    selected["eligibility_pass"] = selected["eligibility_pass"].fillna(False).astype(bool)
    return selected


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


def _summarize_paths(paths: list[Path], *, sample_size: int = 5) -> dict[str, object]:
    """Return a compact path summary for operator artifacts."""
    path_text = [str(path) for path in paths]
    return {
        "count": len(path_text),
        "sample": path_text[:sample_size],
        "last": path_text[-1] if path_text else None,
    }


def _write_backfill_artifact(*, root: Path, payload: dict[str, object]) -> None:
    target = Path(root) / "control" / "cluster" / "state" / "research" / "form4_market_backfill"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(payload["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
