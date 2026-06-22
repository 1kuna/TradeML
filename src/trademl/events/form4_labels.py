"""PIT tradability and forward-label curation for Form 4 event candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable, TypeAlias

import exchange_calendars as xcals
import pandas as pd


LABEL_SCHEMA_VERSION = "form4_event_labels_v1"
DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_BENCHMARK_SYMBOLS = ("IWM", "SPY")
MinuteLookup: TypeAlias = dict[str, pd.DataFrame]
DailyCloseLookup: TypeAlias = dict[tuple[str, str], float]


@dataclass(slots=True, frozen=True)
class Form4LabelConfig:
    """Configuration for Form 4 event label curation."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    round_trip_cost_bps: float = 50.0
    latency_minutes: int = 5
    exchange: str = "XNYS"
    benchmark_symbols: tuple[str, ...] = DEFAULT_BENCHMARK_SYMBOLS


@dataclass(slots=True, frozen=True)
class MarketFrameLoadResult:
    """Loaded market-data frame with compact source-coverage metadata."""

    frame: pd.DataFrame
    metadata: dict[str, object]


def run_form4_label_curation(
    *,
    data_root: Path,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> dict[str, object]:
    """Build and persist Form 4 event labels from candidate events and bars."""
    config = Form4LabelConfig(
        horizons=tuple(sorted(set(int(item) for item in (horizons or DEFAULT_HORIZONS)))),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    labels, source_metadata = build_form4_event_labels_from_root(
        data_root=Path(data_root),
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
        return_source_metadata=True,
    )
    return write_form4_event_labels(
        root=Path(data_root),
        labels=labels,
        config=config,
        source_metadata=source_metadata,
    )


def build_form4_event_labels_from_root(
    *,
    data_root: Path,
    config: Form4LabelConfig | None = None,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
    return_source_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """Build Form 4 event labels from NAS-style candidate and market-data artifacts."""
    config = config or Form4LabelConfig()
    root = Path(data_root)
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
    labels, source_metadata = build_form4_event_labels_with_market_sources(
        candidates=candidates,
        data_root=root,
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    if not return_source_metadata:
        return labels
    return labels, source_metadata


def build_form4_event_labels_with_market_sources(
    *,
    candidates: pd.DataFrame,
    data_root: Path,
    config: Form4LabelConfig | None = None,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build labels for a provided candidate frame using contracted market sources."""
    config = config or Form4LabelConfig()
    root = Path(data_root)
    minute_frame, daily_frame, source_metadata = load_form4_market_frames(
        candidates=candidates,
        data_root=root,
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    labels = build_form4_event_labels(
        candidates=candidates,
        minute_bars=minute_frame,
        daily_bars=daily_frame,
        config=config,
    )
    return labels, source_metadata


def load_form4_market_frames(
    *,
    candidates: pd.DataFrame,
    data_root: Path,
    config: Form4LabelConfig | None = None,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load only market-data partitions required to label the given Form 4 candidates."""
    config = config or Form4LabelConfig()
    root = Path(data_root)
    needed = needed_form4_market_selection(candidates=candidates, config=config)
    source_contract = _read_source_contract(
        root=root,
        source_contract_path=source_contract_path,
    )
    search_roots = _market_search_roots(root=root, market_data_roots=market_data_roots)
    minute_result = _read_market_frame(
        root=root,
        search_roots=search_roots,
        dataset="equities_minute",
        contract=source_contract,
        patterns=_minute_patterns(),
        date_column_candidates=("timestamp", "vendor_ts", "ts", "datetime", "date"),
        dates=needed["minute_dates"],
        symbols=needed["symbols"],
    )
    daily_result = _read_market_frame(
        root=root,
        search_roots=search_roots,
        dataset="equities_ohlcv_adj",
        contract=source_contract,
        patterns=_daily_patterns(),
        date_column_candidates=("date", "timestamp", "ts", "datetime"),
        dates=needed["daily_dates"],
        symbols=needed["symbols"] | set(config.benchmark_symbols),
    )
    source_metadata = {
        "source_contract_path": str(source_contract["path"]) if source_contract else None,
        "candidate_symbols": sorted(needed["symbols"]),
        "needed_minute_dates": sorted(needed["minute_dates"]),
        "needed_daily_dates": sorted(needed["daily_dates"]),
        "datasets": {
            "equities_minute": minute_result.metadata,
            "equities_ohlcv_adj": daily_result.metadata,
        },
    }
    return minute_result.frame, daily_result.frame, source_metadata


def build_form4_event_labels(
    *,
    candidates: pd.DataFrame,
    minute_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: Form4LabelConfig | None = None,
) -> pd.DataFrame:
    """Build PIT entry/exit labels for Form 4 candidate events."""
    config = config or Form4LabelConfig()
    calendar = xcals.get_calendar(config.exchange)
    minute = _normalize_minute_bars(minute_bars)
    daily = _normalize_daily_bars(daily_bars)
    minute_lookup = _minute_lookup(minute)
    daily_close_lookup = _daily_close_lookup(daily)
    daily_symbols = set(daily["symbol"].unique()) if not daily.empty else set()
    rows: list[dict[str, object]] = []
    for event in candidates.sort_values("event_id").to_dict("records"):
        rows.append(
            _label_one_event(
                event=event,
                minute_lookup=minute_lookup,
                daily_close_lookup=daily_close_lookup,
                daily_symbols=daily_symbols,
                calendar=calendar,
                config=config,
            )
        )
    return pd.DataFrame(rows)


def write_form4_event_labels(
    *,
    root: Path,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write Form 4 event labels and a compact curation report."""
    target = (
        Path(root)
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_labels"
    )
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / "data.parquet"
    labels.to_parquet(data_path, index=False)
    report = summarize_form4_event_labels(
        labels=labels,
        config=config,
        source_metadata=source_metadata,
    )
    report["artifact"] = str(data_path)
    report_target = (
        Path(root)
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_event_labels"
    )
    history = report_target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = report_target / "latest.json"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(report["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"labels_path": str(data_path), "report_path": str(latest), "report": report}


def summarize_form4_event_labels(
    *,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize Form 4 label coverage and blockers."""
    blocker_counts: dict[str, int] = {}
    for blockers in labels.get("label_blockers", pd.Series(dtype=object)).tolist():
        for blocker in _list(blockers):
            blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1
    status_counts = (
        labels["label_status"].value_counts().sort_index().to_dict()
        if "label_status" in labels
        else {}
    )
    return {
        "version": LABEL_SCHEMA_VERSION,
        "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate_count": int(len(labels)),
        "labeled_count": int((labels.get("label_status") == "LABELED").sum())
        if "label_status" in labels
        else 0,
        "blocked_count": int((labels.get("label_status") == "BLOCKED").sum())
        if "label_status" in labels
        else 0,
        "skipped_count": int((labels.get("label_status") == "SKIPPED_INELIGIBLE").sum())
        if "label_status" in labels
        else 0,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "horizons": list(config.horizons),
        "round_trip_cost_bps": config.round_trip_cost_bps,
        "latency_minutes": config.latency_minutes,
        "primary_horizon": 5 if 5 in config.horizons else config.horizons[0],
        "source_metadata": source_metadata or {},
    }


def resolve_form4_tradable_at(
    accepted_at_utc: str | pd.Timestamp,
    *,
    latency_minutes: int = 5,
    exchange: str = "XNYS",
) -> pd.Timestamp:
    """Resolve SEC accepted time to the first tradable minute under MVP rules."""
    calendar = xcals.get_calendar(exchange)
    accepted = _timestamp_utc(accepted_at_utc)
    session = calendar.date_to_session(accepted.date(), direction="next")
    session_open = calendar.session_open(session)
    session_close = calendar.session_close(session)
    latency = pd.Timedelta(minutes=int(latency_minutes))
    if accepted < session_open:
        return session_open + latency
    if accepted >= session_close:
        next_session = calendar.next_session(session)
        return calendar.session_open(next_session) + latency
    decision = _ceil_minute(accepted + latency)
    if decision <= session_close:
        return decision
    next_session = calendar.next_session(session)
    return calendar.session_open(next_session)


def _label_one_event(
    *,
    event: dict[str, object],
    minute_lookup: MinuteLookup,
    daily_close_lookup: DailyCloseLookup,
    daily_symbols: set[str],
    calendar,
    config: Form4LabelConfig,
) -> dict[str, object]:
    row = dict(event)
    row["label_schema_version"] = LABEL_SCHEMA_VERSION
    row["round_trip_cost_bps"] = config.round_trip_cost_bps
    row["label_blockers"] = []
    row["entry_price"] = None
    row["entry_price_source"] = None
    row["benchmark_symbol"] = None
    if not bool(event.get("eligibility_pass")):
        row["label_status"] = "SKIPPED_INELIGIBLE"
        row["label_blockers"] = ["candidate_ineligible"]
        return _ensure_horizon_columns(row, config.horizons)
    accepted_at = event.get("accepted_at_utc")
    if not accepted_at:
        return _blocked(row, config.horizons, "missing_accepted_at")
    source_blockers: list[str] = []
    if not minute_lookup:
        source_blockers.append("missing_minute_source")
    if not daily_close_lookup:
        source_blockers.append("missing_daily_source")
    if source_blockers:
        return _blocked(row, config.horizons, source_blockers)
    ticker = str(event.get("ticker") or "")
    tradable_at = resolve_form4_tradable_at(
        str(accepted_at),
        latency_minutes=config.latency_minutes,
        exchange=config.exchange,
    )
    row["tradable_at_utc"] = tradable_at.isoformat()
    entry = _entry_minute(
        minute_lookup=minute_lookup, symbol=ticker, tradable_at=tradable_at
    )
    if entry is None:
        return _blocked(row, config.horizons, "missing_entry_minute")
    entry_price = float(entry["open"])
    row["entry_price"] = entry_price
    row["entry_price_source"] = "minute_open"
    benchmark_symbol = _select_benchmark(daily_symbols, config.benchmark_symbols)
    row["benchmark_symbol"] = benchmark_symbol
    entry_session = calendar.date_to_session(tradable_at.date(), direction="previous")
    blockers: list[str] = []
    for horizon in config.horizons:
        exit_session = _horizon_session(calendar, entry_session, horizon)
        exit_date = str(exit_session.date())
        stock_close = _daily_close(daily_close_lookup, symbol=ticker, date=exit_date)
        benchmark_entry_close = (
            _daily_close(
                daily_close_lookup, symbol=benchmark_symbol, date=str(entry_session.date())
            )
            if benchmark_symbol
            else None
        )
        benchmark_exit_close = (
            _daily_close(daily_close_lookup, symbol=benchmark_symbol, date=exit_date)
            if benchmark_symbol
            else None
        )
        row[f"exit_date_{horizon}d"] = exit_date
        row[f"exit_close_{horizon}d"] = stock_close
        if stock_close is None:
            blockers.append(f"missing_exit_close_{horizon}d")
            row[f"ret_{horizon}d_net"] = None
            row[f"abret_{horizon}d_net"] = None
            continue
        stock_ret = math.log(float(stock_close) / entry_price)
        net_ret = stock_ret - (config.round_trip_cost_bps / 10_000.0)
        row[f"ret_{horizon}d_net"] = net_ret
        if benchmark_entry_close is None or benchmark_exit_close is None:
            blockers.append(f"missing_benchmark_{horizon}d")
            row[f"abret_{horizon}d_net"] = None
        else:
            benchmark_ret = math.log(float(benchmark_exit_close) / float(benchmark_entry_close))
            row[f"abret_{horizon}d_net"] = net_ret - benchmark_ret
    row["label_blockers"] = sorted(set(blockers))
    row["label_status"] = "LABELED" if not blockers else "BLOCKED"
    return _ensure_horizon_columns(row, config.horizons)


def _blocked(
    row: dict[str, object], horizons: tuple[int, ...], blocker: str | list[str]
) -> dict[str, object]:
    row["label_status"] = "BLOCKED"
    row["label_blockers"] = [blocker] if isinstance(blocker, str) else blocker
    return _ensure_horizon_columns(row, horizons)


def _ensure_horizon_columns(row: dict[str, object], horizons: tuple[int, ...]) -> dict[str, object]:
    for horizon in horizons:
        row.setdefault(f"exit_date_{horizon}d", None)
        row.setdefault(f"exit_close_{horizon}d", None)
        row.setdefault(f"ret_{horizon}d_net", None)
        row.setdefault(f"abret_{horizon}d_net", None)
    return row


def _entry_minute(
    *, minute_lookup: MinuteLookup, symbol: str, tradable_at: pd.Timestamp
) -> pd.Series | None:
    subset = minute_lookup.get(symbol)
    if subset is None or subset.empty:
        return None
    position = subset["timestamp"].searchsorted(tradable_at, side="left")
    if int(position) >= len(subset):
        return None
    return subset.iloc[int(position)]


def _daily_close(
    daily_close_lookup: DailyCloseLookup, *, symbol: str | None, date: str
) -> float | None:
    if not symbol:
        return None
    return daily_close_lookup.get((symbol, date))


def _horizon_session(calendar, entry_session: pd.Timestamp, horizon: int) -> pd.Timestamp:
    session = entry_session
    for _ in range(int(horizon)):
        session = calendar.next_session(session)
    return session


def _select_benchmark(daily_symbols: set[str], symbols: tuple[str, ...]) -> str | None:
    for symbol in symbols:
        if symbol in daily_symbols:
            return symbol
    return None


def _minute_lookup(frame: pd.DataFrame) -> MinuteLookup:
    if frame.empty:
        return {}
    return {
        str(symbol): group.sort_values("timestamp").reset_index(drop=True)
        for symbol, group in frame.groupby("symbol", sort=False)
    }


def _daily_close_lookup(frame: pd.DataFrame) -> DailyCloseLookup:
    if frame.empty:
        return {}
    latest = frame.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    return {
        (str(row["symbol"]), str(row["date"])): float(row["close"])
        for row in latest[["symbol", "date", "close"]].to_dict("records")
    }


def _normalize_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "close"])
    normalized = frame.copy()
    timestamp_col = _first_existing(
        normalized, ("timestamp", "vendor_ts", "ts", "datetime", "date")
    )
    if timestamp_col is None:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "close"])
    normalized["timestamp"] = pd.to_datetime(normalized[timestamp_col], utc=True)
    normalized["symbol"] = normalized["symbol"].astype(str)
    return normalized.dropna(subset=["timestamp", "symbol"]).sort_values(
        ["symbol", "timestamp"]
    )


def _normalize_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    normalized = frame.copy()
    date_col = _first_existing(normalized, ("date", "timestamp", "ts", "datetime"))
    if date_col is None:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    normalized["date"] = pd.to_datetime(normalized[date_col], utc=True).dt.date.astype(str)
    normalized["symbol"] = normalized["symbol"].astype(str)
    return normalized.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])


def _read_market_frame(
    *,
    root: Path,
    search_roots: tuple[Path, ...],
    dataset: str,
    contract: dict[str, object] | None,
    patterns: tuple[str, ...],
    date_column_candidates: tuple[str, ...],
    dates: set[str] | None = None,
    symbols: set[str] | None = None,
) -> MarketFrameLoadResult:
    source_paths = _market_source_paths(
        root=root,
        search_roots=search_roots,
        dataset=dataset,
        contract=contract,
        patterns=patterns,
        dates=dates,
    )
    paths = [source.path for source in source_paths]
    paths = [path for path in dict.fromkeys(paths) if path.is_file()]
    frames: list[pd.DataFrame] = []
    loaded_paths: list[str] = []
    rows_before_filter = 0
    rows_after_filter = 0
    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if frame.empty or "symbol" not in frame or "close" not in frame:
            continue
        if _first_existing(frame, date_column_candidates) is None:
            continue
        rows_before_filter += int(len(frame))
        if symbols and "symbol" in frame:
            frame = frame[frame["symbol"].astype(str).isin(symbols)]
        rows_after_filter += int(len(frame))
        if frame.empty:
            continue
        loaded_paths.append(str(path))
        frames.append(frame)
    if not frames:
        return MarketFrameLoadResult(
            frame=pd.DataFrame(),
            metadata={
                "dataset": dataset,
                "status": "missing_or_empty",
                "searched_paths": [str(source.path) for source in source_paths],
                "existing_files": [str(path) for path in paths],
                "loaded_files": [],
                "rows_before_symbol_filter": rows_before_filter,
                "rows_after_symbol_filter": rows_after_filter,
                "symbols": sorted(symbols or []),
                "dates": sorted(dates or []),
            },
        )
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    coverage = _market_frame_coverage(
        frame=frame,
        date_column_candidates=date_column_candidates,
        requested_symbols=symbols,
        requested_dates=dates,
    )
    return MarketFrameLoadResult(
        frame=frame,
        metadata={
            "dataset": dataset,
            "status": "available",
            "searched_paths": [str(source.path) for source in source_paths],
            "existing_files": [str(path) for path in paths],
            "loaded_files": loaded_paths,
            "file_count": len(loaded_paths),
            "rows_before_symbol_filter": rows_before_filter,
            "rows_after_symbol_filter": rows_after_filter,
            "rows_loaded": int(len(frame)),
            "symbols": sorted(symbols or []),
            "dates": sorted(dates or []),
            **coverage,
        },
    )


@dataclass(slots=True, frozen=True)
class _MarketSourcePath:
    path: Path
    source: str


def needed_form4_market_selection(
    *,
    candidates: pd.DataFrame,
    config: Form4LabelConfig | None = None,
) -> dict[str, set[str]]:
    """Return the symbols and date partitions needed for PIT Form 4 labels."""
    config = config or Form4LabelConfig()
    calendar = xcals.get_calendar(config.exchange)
    symbols: set[str] = set()
    minute_dates: set[str] = set()
    daily_dates: set[str] = set()
    if candidates.empty:
        return {"symbols": symbols, "minute_dates": minute_dates, "daily_dates": daily_dates}
    for event in candidates.sort_values("event_id").to_dict("records"):
        if not bool(event.get("eligibility_pass")) or not event.get("accepted_at_utc"):
            continue
        symbol = str(event.get("ticker") or "").strip()
        if not symbol:
            continue
        symbols.add(symbol)
        tradable_at = resolve_form4_tradable_at(
            str(event["accepted_at_utc"]),
            latency_minutes=config.latency_minutes,
            exchange=config.exchange,
        )
        minute_dates.add(str(tradable_at.date()))
        entry_session = calendar.date_to_session(tradable_at.date(), direction="previous")
        daily_dates.add(str(entry_session.date()))
        for horizon in config.horizons:
            daily_dates.add(str(_horizon_session(calendar, entry_session, horizon).date()))
    return {"symbols": symbols, "minute_dates": minute_dates, "daily_dates": daily_dates}


def _read_source_contract(
    *,
    root: Path,
    source_contract_path: Path | None,
) -> dict[str, object] | None:
    contract_path = source_contract_path or (
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "feature_source_contract"
        / "latest.json"
    )
    if not contract_path.exists():
        return None
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {"path": contract_path, "payload": payload}


def _market_search_roots(
    *,
    root: Path,
    market_data_roots: Iterable[Path] | None,
) -> tuple[Path, ...]:
    roots = [root]
    roots.extend(Path(item).expanduser() for item in (market_data_roots or ()))
    return tuple(Path(item) for item in dict.fromkeys(roots))


def _market_source_paths(
    *,
    root: Path,
    search_roots: tuple[Path, ...],
    dataset: str,
    contract: dict[str, object] | None,
    patterns: tuple[str, ...],
    dates: set[str] | None,
) -> list[_MarketSourcePath]:
    paths: list[_MarketSourcePath] = []
    for dataset_path in _contract_dataset_paths(
        root=root, dataset=dataset, contract=contract
    ):
        paths.extend(
            _partition_paths(dataset_path, dates=dates, source="contract")
        )
    for search_root in search_roots:
        for pattern in patterns:
            pattern_root = _pattern_static_root(search_root, pattern)
            if pattern_root is not None:
                paths.extend(
                    _partition_paths(pattern_root, dates=dates, source="default")
                )
            elif dates:
                continue
            else:
                paths.extend(
                    _MarketSourcePath(path=path, source="glob")
                    for path in sorted(search_root.glob(pattern))
                )
    deduped: dict[Path, _MarketSourcePath] = {}
    for item in paths:
        deduped.setdefault(item.path, item)
    return list(deduped.values())


def _contract_dataset_paths(
    *,
    root: Path,
    dataset: str,
    contract: dict[str, object] | None,
) -> list[Path]:
    if not contract:
        return []
    payload = dict(contract.get("payload") or {})
    datasets = dict(payload.get("datasets") or {})
    candidates = [dataset]
    if dataset == "equities_ohlcv_adj":
        candidates.extend(["equities_eod", "equities_bars"])
    paths: list[Path] = []
    for name in candidates:
        entry = dict(datasets.get(name) or {})
        for raw_path in list(entry.get("paths") or []):
            path = Path(str(raw_path)).expanduser()
            paths.append(path if path.is_absolute() else root / path)
        summary = dict(entry.get("summary") or {})
        for raw_path in list(summary.get("existing_paths") or []):
            path = Path(str(raw_path)).expanduser()
            paths.append(path if path.is_absolute() else root / path)
        for source in list(summary.get("sources") or []):
            if not isinstance(source, dict) or not source.get("exists"):
                continue
            raw_path = source.get("path")
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser()
            paths.append(path if path.is_absolute() else root / path)
    return list(dict.fromkeys(paths))


def _pattern_static_root(root: Path, pattern: str) -> Path | None:
    marker = "**"
    if marker not in pattern:
        return None
    prefix = pattern.split(marker, 1)[0].rstrip("/")
    if not prefix:
        return None
    return root / prefix


def _partition_paths(
    base: Path,
    *,
    dates: set[str] | None,
    source: str,
) -> list[_MarketSourcePath]:
    if base.suffix == ".parquet":
        return [_MarketSourcePath(path=base, source=source)]
    if dates:
        paths: list[_MarketSourcePath] = []
        for date in sorted(dates):
            partition = base / f"date={date}"
            paths.append(_MarketSourcePath(path=partition / "data.parquet", source=source))
            paths.extend(
                _MarketSourcePath(path=path, source=source)
                for path in sorted(partition.glob("*.parquet"))
                if path.name != "data.parquet"
            )
        return paths
    return [
        _MarketSourcePath(path=path, source=source)
        for path in sorted(base.glob("**/*.parquet"))
    ]


def _market_frame_coverage(
    *,
    frame: pd.DataFrame,
    date_column_candidates: tuple[str, ...],
    requested_symbols: set[str] | None,
    requested_dates: set[str] | None,
) -> dict[str, object]:
    symbol_values = (
        set(frame["symbol"].dropna().astype(str).unique().tolist())
        if "symbol" in frame
        else set()
    )
    date_values: set[str] = set()
    date_column = _first_existing(frame, date_column_candidates)
    if date_column:
        timestamps = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        date_values = set(timestamps.dropna().dt.date.astype(str).unique().tolist())
    return {
        "loaded_symbols": sorted(symbol_values),
        "missing_requested_symbols": sorted((requested_symbols or set()) - symbol_values),
        "loaded_dates": sorted(date_values),
        "missing_requested_dates": sorted((requested_dates or set()) - date_values),
    }


def _minute_patterns() -> tuple[str, ...]:
    return (
        "data/curated/equities_minute/**/*.parquet",
        "data/raw/equities_minute/**/*.parquet",
        "data/raw/archive/equities_minute/**/*.parquet",
        "data/archive/equities_minute/**/*.parquet",
    )


def _daily_patterns() -> tuple[str, ...]:
    return (
        "data/curated/equities_eod/**/*.parquet",
        "data/raw/equities_eod/**/*.parquet",
        "data/curated/equities_ohlcv_adj/**/*.parquet",
        "data/raw/equities_ohlcv_adj/**/*.parquet",
        "data/curated/equities_bars/**/*.parquet",
        "data/raw/equities_bars/**/*.parquet",
    )


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _timestamp_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert("UTC")


def _ceil_minute(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.ceil("min")


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if pd.isna(value):
        return []
    if isinstance(value, str) and re.match(r"^\[.*\]$", value):
        return [value]
    return [value]
