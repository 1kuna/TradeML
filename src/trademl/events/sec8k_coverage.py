"""SEC 8-K coverage expansion and semantic labelability orchestration."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Callable, Iterable

import pandas as pd

from trademl.events.form4_fixture_gate import build_sec_form4_connector_from_env
from trademl.events.sec8k import run_sec8k_candidate_curation
from trademl.events.sec8k_ingest import Sec8KIngestClient, run_sec8k_ingest
from trademl.events.sec8k_market_backfill import run_sec8k_market_backfill_from_env
from trademl.events.sec8k_semantic import (
    DEFAULT_MAC_MINI_LMSTUDIO_BASE_URL,
    DEFAULT_SCALED_GATE_FALLBACK_ITEMS,
    DEFAULT_SCALED_GATE_TARGET_ITEMS,
    DEFAULT_SEC_EVENT_MODEL,
    build_sec8k_semantic_labelability_queue,
    read_sec8k_item_event_candidates,
    run_sec_event_semantic_labelability_audit,
    run_sec_event_semantic_scaled_gate,
)


SEC8K_COVERAGE_AUDIT_VERSION = "sec8k_coverage_audit_v1"
SEC8K_COVERAGE_EXPAND_VERSION = "sec8k_coverage_expand_v1"
SEC8K_SEMANTIC_COVERAGE_GATE_VERSION = "sec8k_semantic_coverage_gate_v1"
DEFAULT_COVERAGE_START_DATE = "2024-01-01"
DEFAULT_COVERAGE_END_DATE = "2025-12-31"
DEFAULT_COVERAGE_HORIZONS = (5,)
MARKET_LABEL_BLOCKER_PREFIXES = (
    "missing_entry_minute",
    "missing_exit_close_",
    "missing_benchmark_",
)


def run_sec8k_coverage_audit(
    *,
    data_root: Path,
    start_date: str = DEFAULT_COVERAGE_START_DATE,
    end_date: str = DEFAULT_COVERAGE_END_DATE,
    target_items: Iterable[str] | None = None,
    fallback_target_items: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
) -> dict[str, object]:
    """Audit SEC 8-K source, candidate, labelability, and market coverage."""
    root = Path(data_root).expanduser()
    start = _date_bound(start_date)
    end = _date_bound(end_date)
    selected_targets = _combined_target_items(
        target_items=target_items,
        fallback_target_items=fallback_target_items,
    )
    selected_horizons = tuple(sorted(set(int(item) for item in (horizons or DEFAULT_COVERAGE_HORIZONS))))

    manifest = _read_parquet(_manifest_path(root))
    filing_index = _read_parquet(_filing_index_path(root))
    candidates = read_sec8k_item_event_candidates(_candidate_path(root))
    labelability, labelability_source_metadata = _coverage_labelability_queue(
        root=root,
        target_items=selected_targets,
        start_date=start,
        end_date=end,
        horizons=selected_horizons,
        round_trip_cost_bps=round_trip_cost_bps,
        candidates_exist=_candidate_path(root).exists(),
    )

    monthly = _monthly_coverage(
        root=root,
        months=_month_bounds(start, end),
        manifest=manifest,
        filing_index=filing_index,
        candidates=candidates,
        labelability=labelability,
        target_items=selected_targets,
    )
    monthly_path = _write_monthly_coverage(root=root, monthly=monthly)
    labelability_path = _write_coverage_labelability_queue(root=root, queue=labelability)

    requested_months = monthly["month"].astype(str).tolist() if "month" in monthly else []
    missing_sec_months = monthly.loc[
        (monthly["manifest_count"].fillna(0).astype(int) == 0)
        | (monthly["filing_index_count"].fillna(0).astype(int) == 0),
        "month",
    ].astype(str).tolist()
    raw_archive_gap_months = monthly.loc[
        monthly["raw_archive_gap_count"].fillna(0).astype(int) > 0, "month"
    ].astype(str).tolist()
    candidate_missing_months = monthly.loc[
        monthly["candidate_count"].fillna(0).astype(int) == 0, "month"
    ].astype(str).tolist()
    months_requiring_ingest = sorted(set(missing_sec_months) | set(raw_archive_gap_months))
    blocker_counts = _blocker_counts(
        labelability.get("labelability_blockers", pd.Series(dtype=object)).tolist()
    )
    market_blocker_count = sum(
        count
        for blocker, count in blocker_counts.items()
        if _is_market_label_blocker(blocker)
    )
    artifact_range = _candidate_date_range(candidates)
    payload = _json_safe(
        {
            "version": SEC8K_COVERAGE_AUDIT_VERSION,
            "checked_at": _now_iso(),
            "data_root": str(root),
            "start_date": start,
            "end_date": end,
            "target_items": list(selected_targets),
            "horizons": list(selected_horizons),
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "requested_month_count": len(requested_months),
            "manifest_count": int(len(_date_filtered(manifest, start_date=start, end_date=end))),
            "filing_index_count": int(len(_date_filtered(filing_index, start_date=start, end_date=end))),
            "candidate_count": int(len(_date_filtered(candidates, start_date=start, end_date=end))),
            "target_candidate_count": int(
                len(
                    _target_filtered_candidates(
                        candidates=_date_filtered(candidates, start_date=start, end_date=end),
                        target_items=selected_targets,
                    )
                )
            ),
            "labelability_candidate_count": int(len(labelability)),
            "labelable_count": int(
                (labelability.get("labelability_status") == "LABELABLE").sum()
            )
            if "labelability_status" in labelability
            else 0,
            "labelability_status_counts": _value_counts(labelability, "labelability_status"),
            "labelability_blocker_counts": blocker_counts,
            "market_label_blocker_count": int(market_blocker_count),
            "candidate_artifact_min_date": artifact_range[0],
            "candidate_artifact_max_date": artifact_range[1],
            "candidate_artifact_range_mismatch": _candidate_range_mismatch(
                artifact_range=artifact_range,
                requested_months=requested_months,
                candidate_missing_months=candidate_missing_months,
            ),
            "missing_sec_months": missing_sec_months,
            "raw_archive_gap_months": raw_archive_gap_months,
            "candidate_missing_months": candidate_missing_months,
            "months_requiring_ingest": months_requiring_ingest,
            "raw_archive_gap_count": int(monthly["raw_archive_gap_count"].sum())
            if "raw_archive_gap_count" in monthly
            else 0,
            "monthly_artifact": str(monthly_path),
            "labelability_queue_artifact": str(labelability_path),
            "labelability_source_metadata": labelability_source_metadata,
            "monthly": monthly.to_dict("records"),
            "verdict": _coverage_audit_verdict(
                months_requiring_ingest=months_requiring_ingest,
                candidate_missing_months=candidate_missing_months,
            ),
        }
    )
    return _write_state_packet(root=root, name="sec_8k_coverage", payload=payload)


def run_sec8k_coverage_expand(
    *,
    data_root: Path,
    connector: Sec8KIngestClient | None = None,
    start_date: str = DEFAULT_COVERAGE_START_DATE,
    end_date: str = DEFAULT_COVERAGE_END_DATE,
    target_items: Iterable[str] | None = None,
    fallback_target_items: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    limit_per_month: int | None = None,
    user_agent: str | None = None,
    max_retrieval_attempts: int = 6,
    rate_limit_pause_seconds: float = 60.0,
    use_cache: bool = True,
    rebuild_candidates: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Ingest missing SEC 8-K months and rebuild global item candidates."""
    root = Path(data_root).expanduser()
    before = run_sec8k_coverage_audit(
        data_root=root,
        start_date=start_date,
        end_date=end_date,
        target_items=target_items,
        fallback_target_items=fallback_target_items,
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
    )["packet"]
    planned_months = [str(item) for item in before.get("months_requiring_ingest", [])]
    ingest_client = connector
    if planned_months and ingest_client is None:
        ingest_client = build_sec_form4_connector_from_env(user_agent=user_agent)

    month_results: list[dict[str, object]] = []
    for month in planned_months:
        month_start, month_end = _month_range_within(month, start_date=start_date, end_date=end_date)
        assert ingest_client is not None
        try:
            result = run_sec8k_ingest(
                data_root=root,
                connector=ingest_client,
                start_date=month_start,
                end_date=month_end,
                limit=limit_per_month,
                max_retrieval_attempts=max_retrieval_attempts,
                rate_limit_pause_seconds=rate_limit_pause_seconds,
                use_cache=use_cache,
                sleep_fn=sleep_fn,
            )
            month_results.append(
                {
                    "month": month,
                    "start_date": month_start,
                    "end_date": month_end,
                    "status": "PASS" if result.get("verdict") == "PASS" else "PARTIAL",
                    "manifest_count": result.get("manifest_count", 0),
                    "parsed_count": result.get("parsed_count", 0),
                    "failed_count": result.get("failed_count", 0),
                    "ingest_verdict": result.get("verdict"),
                }
            )
        except Exception as exc:  # pragma: no cover - live operational guard
            month_results.append(
                {
                    "month": month,
                    "start_date": month_start,
                    "end_date": month_end,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidate_rebuild: dict[str, object] | None = None
    if rebuild_candidates:
        try:
            candidate_rebuild = run_sec8k_candidate_curation(data_root=root)
        except FileNotFoundError as exc:
            candidate_rebuild = {"status": "FAILED", "error": str(exc)}
    after = run_sec8k_coverage_audit(
        data_root=root,
        start_date=start_date,
        end_date=end_date,
        target_items=target_items,
        fallback_target_items=fallback_target_items,
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
    )["packet"]
    failed_months = [
        str(item["month"]) for item in month_results if item.get("status") == "FAILED"
    ]
    payload = _json_safe(
        {
            "version": SEC8K_COVERAGE_EXPAND_VERSION,
            "checked_at": _now_iso(),
            "data_root": str(root),
            "start_date": _date_bound(start_date),
            "end_date": _date_bound(end_date),
            "target_items": list(
                _combined_target_items(
                    target_items=target_items,
                    fallback_target_items=fallback_target_items,
                )
            ),
            "planned_months": planned_months,
            "completed_months": [
                str(item["month"]) for item in month_results if item.get("status") != "FAILED"
            ],
            "failed_months": failed_months,
            "month_results": month_results,
            "limit_per_month": limit_per_month,
            "cache_enabled": bool(use_cache),
            "rebuild_candidates": bool(rebuild_candidates),
            "candidate_rebuild": candidate_rebuild,
            "before_audit": before,
            "after_audit": after,
            "verdict": "PASS"
            if not failed_months and not after.get("months_requiring_ingest")
            else "PARTIAL",
        }
    )
    return _write_state_packet(root=root, name="sec_8k_coverage_expand", payload=payload)


def run_sec_event_semantic_coverage_gate(
    *,
    data_root: Path,
    start_date: str = DEFAULT_COVERAGE_START_DATE,
    end_date: str = DEFAULT_COVERAGE_END_DATE,
    target_items: Iterable[str] | None = None,
    fallback_target_items: Iterable[str] | None = None,
    model: str = DEFAULT_SEC_EVENT_MODEL,
    base_url: str = DEFAULT_MAC_MINI_LMSTUDIO_BASE_URL,
    timeout_seconds: float = 300.0,
    response_format_mode: str = "prompt_json",
    batch_size: int = 1,
    max_snippets: int | None = None,
    resume: bool = True,
    primary_horizon: int = 5,
    min_sample: int = 100,
    min_mean_abret: float = 0.005,
    min_control_separation: float = 0.0075,
    max_top5_abs_contribution: float = 0.35,
    round_trip_cost_bps: float = 50.0,
    expand_missing_coverage: bool = True,
    repair_market_coverage: bool = True,
    limit_per_month: int | None = None,
    user_agent: str | None = None,
    max_retrieval_attempts: int = 6,
    sec_rate_limit_pause_seconds: float = 60.0,
    market_max_fetch_attempts: int = 6,
    market_rate_limit_pause_seconds: float = 60.0,
    daily_symbol_batch_size: int = 100,
) -> dict[str, object]:
    """Run coverage expansion, labelability repair, then the semantic scaled gate."""
    root = Path(data_root).expanduser()
    primary_items = _target_items_or_default(target_items, DEFAULT_SCALED_GATE_TARGET_ITEMS)
    fallback_items = _target_items_or_default(
        fallback_target_items,
        DEFAULT_SCALED_GATE_FALLBACK_ITEMS,
    )
    all_items = tuple(dict.fromkeys((*primary_items, *fallback_items)))
    initial_audit = run_sec8k_coverage_audit(
        data_root=root,
        start_date=start_date,
        end_date=end_date,
        target_items=primary_items,
        fallback_target_items=fallback_items,
        horizons=(primary_horizon,),
        round_trip_cost_bps=round_trip_cost_bps,
    )["packet"]
    expansion_payload = None
    if expand_missing_coverage and initial_audit.get("months_requiring_ingest"):
        expansion_payload = run_sec8k_coverage_expand(
            data_root=root,
            start_date=start_date,
            end_date=end_date,
            target_items=primary_items,
            fallback_target_items=fallback_items,
            horizons=(primary_horizon,),
            round_trip_cost_bps=round_trip_cost_bps,
            limit_per_month=limit_per_month,
            user_agent=user_agent,
            max_retrieval_attempts=max_retrieval_attempts,
            rate_limit_pause_seconds=sec_rate_limit_pause_seconds,
            use_cache=True,
            rebuild_candidates=True,
        )["packet"]
    else:
        try:
            run_sec8k_candidate_curation(data_root=root)
        except FileNotFoundError:
            pass

    post_coverage_audit = run_sec8k_coverage_audit(
        data_root=root,
        start_date=start_date,
        end_date=end_date,
        target_items=primary_items,
        fallback_target_items=fallback_items,
        horizons=(primary_horizon,),
        round_trip_cost_bps=round_trip_cost_bps,
    )["packet"]
    primary_labelability = run_sec_event_semantic_labelability_audit(
        data_root=root,
        routing_mode="targeted",
        target_items=primary_items,
        accepted_from=start_date,
        accepted_to=end_date,
        snippet_kind="item_section",
        horizons=(primary_horizon,),
        round_trip_cost_bps=round_trip_cost_bps,
    )["payload"]
    selected_items = primary_items
    selected_labelability = primary_labelability
    fallback_labelability = None
    if int(primary_labelability.get("labelable_count") or 0) < int(min_sample):
        fallback_labelability = run_sec_event_semantic_labelability_audit(
            data_root=root,
            routing_mode="targeted",
            target_items=all_items,
            accepted_from=start_date,
            accepted_to=end_date,
            snippet_kind="item_section",
            horizons=(primary_horizon,),
            round_trip_cost_bps=round_trip_cost_bps,
        )["payload"]
        selected_items = all_items
        selected_labelability = fallback_labelability

    backfill_payload = None
    if (
        repair_market_coverage
        and int(selected_labelability.get("labelable_count") or 0) < int(min_sample)
        and _market_blocker_count(selected_labelability) > 0
    ):
        try:
            backfill_payload = run_sec8k_market_backfill_from_env(
                data_root=root,
                horizons=(primary_horizon,),
                round_trip_cost_bps=round_trip_cost_bps,
                limit_events=None,
                include_timestamp_placebo=True,
                max_fetch_attempts=market_max_fetch_attempts,
                rate_limit_pause_seconds=market_rate_limit_pause_seconds,
                daily_symbol_batch_size=daily_symbol_batch_size,
                candidate_source="sec8k_item_events",
                target_items=selected_items,
                accepted_from=start_date,
                accepted_to=end_date,
            )
        except Exception as exc:  # pragma: no cover - live operational guard
            backfill_payload = {
                "verdict": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        selected_labelability = run_sec_event_semantic_labelability_audit(
            data_root=root,
            routing_mode="targeted",
            target_items=selected_items,
            accepted_from=start_date,
            accepted_to=end_date,
            snippet_kind="item_section",
            horizons=(primary_horizon,),
            round_trip_cost_bps=round_trip_cost_bps,
        )["payload"]

    terminal_without_llm = _pre_llm_terminal_verdict(
        coverage_audit=post_coverage_audit,
        labelability=selected_labelability,
        min_sample=min_sample,
    )
    scaled_gate_payload = None
    final_verdict = terminal_without_llm
    if terminal_without_llm is None:
        scaled_gate_payload = run_sec_event_semantic_scaled_gate(
            data_root=root,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            response_format_mode=response_format_mode,
            batch_size=batch_size,
            target_items=primary_items,
            fallback_target_items=fallback_items,
            years=_years_descending(start_date=start_date, end_date=end_date),
            max_snippets=max_snippets,
            resume=resume,
            primary_horizon=primary_horizon,
            min_sample=min_sample,
            min_mean_abret=min_mean_abret,
            min_control_separation=min_control_separation,
            max_top5_abs_contribution=max_top5_abs_contribution,
        )
        final_verdict = _coverage_gate_scaled_verdict(
            scaled_gate=scaled_gate_payload,
            coverage_audit=post_coverage_audit,
        )

    payload = _json_safe(
        {
            "version": SEC8K_SEMANTIC_COVERAGE_GATE_VERSION,
            "checked_at": _now_iso(),
            "data_root": str(root),
            "start_date": _date_bound(start_date),
            "end_date": _date_bound(end_date),
            "target_items": list(primary_items),
            "fallback_target_items": list(fallback_items),
            "selected_items": list(selected_items),
            "model": model,
            "base_url": base_url,
            "response_format_mode": response_format_mode,
            "batch_size": int(batch_size),
            "resume": bool(resume),
            "max_snippets": max_snippets,
            "min_sample": int(min_sample),
            "initial_audit": initial_audit,
            "expansion": expansion_payload,
            "post_coverage_audit": post_coverage_audit,
            "primary_labelability": primary_labelability,
            "fallback_labelability": fallback_labelability,
            "selected_labelability": selected_labelability,
            "market_backfill": backfill_payload,
            "scaled_gate": scaled_gate_payload,
            "verdict": final_verdict,
        }
    )
    return _write_state_packet(
        root=root,
        name="sec_event_semantic_coverage_gate",
        payload=payload,
    )


def _coverage_labelability_queue(
    *,
    root: Path,
    target_items: tuple[str, ...],
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...],
    round_trip_cost_bps: float,
    candidates_exist: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not candidates_exist:
        return _empty_labelability_frame(), {"datasets": {}, "missing_candidates": True}
    return build_sec8k_semantic_labelability_queue(
        root=root,
        routing_mode="targeted",
        target_items=target_items,
        accepted_from=start_date,
        accepted_to=end_date,
        snippet_kind="item_section",
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
    )


def _monthly_coverage(
    *,
    root: Path,
    months: list[str],
    manifest: pd.DataFrame,
    filing_index: pd.DataFrame,
    candidates: pd.DataFrame,
    labelability: pd.DataFrame,
    target_items: tuple[str, ...],
) -> pd.DataFrame:
    manifest_counts = _month_count_map(manifest)
    filing_counts = _month_count_map(filing_index)
    raw_counts, raw_gap_counts = _raw_archive_month_maps(
        root=root,
        frame=manifest if not manifest.empty else filing_index,
    )
    candidate_counts = _month_count_map(candidates)
    target_counts = _month_count_map(
        _target_filtered_candidates(candidates=candidates, target_items=target_items)
    )
    labelability_counts = _month_count_map(labelability)
    labelable_counts = _month_count_map(
        labelability[labelability.get("labelability_status") == "LABELABLE"].copy()
        if "labelability_status" in labelability
        else pd.DataFrame()
    )
    missing_ticker_counts = _blocker_month_count_map(labelability, blocker="missing_ticker")
    missing_market_counts = _market_blocker_month_count_map(labelability)
    rows = [
        {
            "month": month,
            "manifest_count": manifest_counts.get(month, 0),
            "filing_index_count": filing_counts.get(month, 0),
            "raw_archive_count": raw_counts.get(month, 0),
            "raw_archive_gap_count": raw_gap_counts.get(month, 0),
            "candidate_count": candidate_counts.get(month, 0),
            "target_candidate_count": target_counts.get(month, 0),
            "labelability_candidate_count": labelability_counts.get(month, 0),
            "labelable_count": labelable_counts.get(month, 0),
            "missing_ticker_count": missing_ticker_counts.get(month, 0),
            "missing_market_data_count": missing_market_counts.get(month, 0),
        }
        for month in months
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "month",
            "manifest_count",
            "filing_index_count",
            "raw_archive_count",
            "raw_archive_gap_count",
            "candidate_count",
            "target_candidate_count",
            "labelability_candidate_count",
            "labelable_count",
            "missing_ticker_count",
            "missing_market_data_count",
        ],
    )


def _raw_archive_month_maps(
    *, root: Path, frame: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int]]:
    raw_counts: dict[str, int] = {}
    gap_counts: dict[str, int] = {}
    if frame.empty:
        return raw_counts, gap_counts
    for row in frame.to_dict("records"):
        month = _month_from_row(row)
        if not month:
            continue
        if _raw_archive_exists(root=root, row=row):
            raw_counts[month] = raw_counts.get(month, 0) + 1
        else:
            gap_counts[month] = gap_counts.get(month, 0) + 1
    return raw_counts, gap_counts


def _raw_archive_exists(*, root: Path, row: dict[str, object]) -> bool:
    for column in ("complete_txt_path", "complete_path"):
        path_text = _text(row.get(column))
        if path_text and Path(path_text).expanduser().exists():
            return True
    archive_cik = _text(row.get("archive_cik") or row.get("cik"))
    accession = _text(
        row.get("accession_no_dashes")
        or row.get("accessionNumber")
        or row.get("accession")
    ).replace("-", "")
    if archive_cik and accession:
        path = (
            root
            / "data"
            / "raw"
            / "sec"
            / "archives"
            / f"archive_cik={archive_cik.lstrip('0') or archive_cik}"
            / f"accession={accession}"
            / "complete.txt"
        )
        return path.exists()
    return False


def _target_filtered_candidates(
    *, candidates: pd.DataFrame, target_items: tuple[str, ...]
) -> pd.DataFrame:
    if candidates.empty or "sec_item_number" not in candidates:
        return pd.DataFrame()
    target_set = set(target_items)
    mask = candidates["sec_item_number"].map(_normalize_sec_item).isin(target_set)
    return candidates[mask].copy()


def _date_filtered(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    dates = frame.apply(_event_date_from_series, axis=1)
    mask = (dates >= start_date) & (dates <= end_date)
    return frame[mask.fillna(False)].copy()


def _month_count_map(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame.empty:
        return counts
    for row in frame.to_dict("records"):
        month = _month_from_row(row)
        if month:
            counts[month] = counts.get(month, 0) + 1
    return counts


def _blocker_month_count_map(frame: pd.DataFrame, *, blocker: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame.empty or "labelability_blockers" not in frame:
        return counts
    for row in frame.to_dict("records"):
        blockers = _list(row.get("labelability_blockers"))
        if blocker not in blockers:
            continue
        month = _month_from_row(row)
        if month:
            counts[month] = counts.get(month, 0) + 1
    return counts


def _market_blocker_month_count_map(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame.empty or "labelability_blockers" not in frame:
        return counts
    for row in frame.to_dict("records"):
        blockers = [str(item) for item in _list(row.get("labelability_blockers"))]
        if not any(_is_market_label_blocker(blocker) for blocker in blockers):
            continue
        month = _month_from_row(row)
        if month:
            counts[month] = counts.get(month, 0) + 1
    return counts


def _candidate_date_range(candidates: pd.DataFrame) -> tuple[str | None, str | None]:
    if candidates.empty:
        return None, None
    dates = sorted(date for date in (_event_date_from_dict(row) for row in candidates.to_dict("records")) if date)
    if not dates:
        return None, None
    return dates[0], dates[-1]


def _candidate_range_mismatch(
    *,
    artifact_range: tuple[str | None, str | None],
    requested_months: list[str],
    candidate_missing_months: list[str],
) -> bool:
    if not requested_months:
        return False
    if artifact_range == (None, None):
        return True
    if candidate_missing_months:
        return True
    return False


def _pre_llm_terminal_verdict(
    *,
    coverage_audit: dict[str, object],
    labelability: dict[str, object],
    min_sample: int,
) -> dict[str, object] | None:
    if coverage_audit.get("months_requiring_ingest"):
        return _verdict(
            "BLOCKED_DATA_COVERAGE",
            ["sec_source_coverage_incomplete_after_expand"],
        )
    candidate_count = int(labelability.get("candidate_count") or 0)
    labelable_count = int(labelability.get("labelable_count") or 0)
    if candidate_count < int(min_sample) and not coverage_audit.get("months_requiring_ingest"):
        return _verdict(
            "MORE_DATA_REQUIRED_TRUE_COVERAGE",
            ["insufficient_target_event_frequency_after_full_coverage"],
        )
    if candidate_count >= int(min_sample) and labelable_count < int(min_sample):
        return _verdict(
            "BLOCKED_DATA_COVERAGE",
            ["insufficient_labelable_market_coverage_after_repair"],
        )
    return None


def _coverage_gate_scaled_verdict(
    *, scaled_gate: dict[str, object], coverage_audit: dict[str, object]
) -> dict[str, object]:
    packet = _dict(scaled_gate.get("packet"))
    verdict = dict(_dict(packet.get("verdict")))
    decision = _text(verdict.get("decision"))
    if decision == "MORE_DATA_REQUIRED" and not coverage_audit.get("months_requiring_ingest"):
        verdict["decision"] = "MORE_DATA_REQUIRED_TRUE_COVERAGE"
        failed = [
            str(item)
            for item in _list(verdict.get("failed_gates"))
            if str(item)
        ]
        if "semantic_event_frequency_insufficient_after_full_coverage" not in failed:
            failed.append("semantic_event_frequency_insufficient_after_full_coverage")
        verdict["failed_gates"] = failed
    verdict["paper_live_allowed"] = False
    verdict.setdefault(
        "paper_live_blocker",
        "sec_8k_semantic_coverage_gate_not_live_trading_surface",
    )
    return verdict


def _coverage_audit_verdict(
    *, months_requiring_ingest: list[str], candidate_missing_months: list[str]
) -> str:
    if months_requiring_ingest:
        return "MISSING_SEC_COVERAGE"
    if candidate_missing_months:
        return "CANDIDATE_ARTIFACT_PARTIAL"
    return "PASS"


def _verdict(decision: str, failed_gates: list[str]) -> dict[str, object]:
    return {
        "decision": decision,
        "move_forward": decision == "CONTINUE_SEMANTIC_8K",
        "paper_live_allowed": False,
        "paper_live_blocker": "sec_8k_semantic_coverage_gate_not_live_trading_surface",
        "passing_segments": [],
        "failed_gates": failed_gates,
    }


def _market_blocker_count(labelability: dict[str, object]) -> int:
    return sum(
        int(value)
        for blocker, value in _dict(labelability.get("blocker_counts")).items()
        if _is_market_label_blocker(str(blocker))
    )


def _is_market_label_blocker(blocker: str) -> bool:
    return any(str(blocker).startswith(prefix) for prefix in MARKET_LABEL_BLOCKER_PREFIXES)


def _combined_target_items(
    *,
    target_items: Iterable[str] | None,
    fallback_target_items: Iterable[str] | None,
) -> tuple[str, ...]:
    primary = _target_items_or_default(target_items, DEFAULT_SCALED_GATE_TARGET_ITEMS)
    fallback = _target_items_or_default(
        fallback_target_items,
        DEFAULT_SCALED_GATE_FALLBACK_ITEMS,
    )
    return tuple(dict.fromkeys((*primary, *fallback)))


def _target_items_or_default(
    items: Iterable[str] | None, default: Iterable[str]
) -> tuple[str, ...]:
    selected = tuple(_normalize_sec_item(item) for item in (items or default))
    return tuple(dict.fromkeys(item for item in selected if item))


def _years_descending(*, start_date: str, end_date: str) -> tuple[int, ...]:
    start_year = int(_date_bound(start_date)[:4])
    end_year = int(_date_bound(end_date)[:4])
    return tuple(range(end_year, start_year - 1, -1))


def _month_bounds(start_date: str, end_date: str) -> list[str]:
    start = pd.Period(_date_bound(start_date)[:7], freq="M")
    end = pd.Period(_date_bound(end_date)[:7], freq="M")
    return [str(period) for period in pd.period_range(start=start, end=end, freq="M")]


def _month_range_within(
    month: str, *, start_date: str, end_date: str
) -> tuple[str, str]:
    period = pd.Period(month, freq="M")
    month_start = period.start_time.date().isoformat()
    month_end = period.end_time.date().isoformat()
    return max(month_start, _date_bound(start_date)), min(month_end, _date_bound(end_date))


def _manifest_path(root: Path) -> Path:
    return root / "data" / "curated" / "sec" / "sec8k" / "manifest" / "data.parquet"


def _filing_index_path(root: Path) -> Path:
    return root / "data" / "reference" / "sec_filing_index.parquet"


def _candidate_path(root: Path) -> Path:
    return root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_monthly_coverage(*, root: Path, monthly: pd.DataFrame) -> Path:
    path = root / "data" / "curated" / "events" / "sec_8k_coverage" / "monthly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(path, index=False)
    return path


def _write_coverage_labelability_queue(*, root: Path, queue: pd.DataFrame) -> Path:
    path = root / "data" / "curated" / "events" / "sec_8k_coverage" / "labelability_queue.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_parquet(path, index=False)
    return path


def _empty_labelability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "accession",
            "ticker",
            "accepted_at_utc",
            "filing_date",
            "sec_item_number",
            "labelability_status",
            "labelability_blockers",
        ]
    )


def _write_state_packet(
    *, root: Path, name: str, payload: dict[str, object]
) -> dict[str, object]:
    target = root / "control" / "cluster" / "state" / "research" / name
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{payload['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"packet_path": str(latest), "history_path": str(history_path), "packet": payload}


def _month_from_row(row: dict[str, object]) -> str:
    date_text = _event_date_from_dict(row)
    return date_text[:7] if date_text else ""


def _event_date_from_series(row: pd.Series) -> str:
    return _event_date_from_dict(row.to_dict())


def _event_date_from_dict(row: dict[str, object]) -> str:
    for column in (
        "accepted_at_utc",
        "accepted_at",
        "acceptanceDateTime",
        "filing_date",
        "filed_date",
        "filingDate",
    ):
        value = _date_text(row.get(column))
        if value:
            return value
    return ""


def _date_bound(value: object) -> str:
    date_text = _date_text(value)
    if not date_text:
        raise ValueError(f"date value must be parseable as YYYY-MM-DD: {value}")
    return date_text


def _date_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat() if value.tzinfo else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
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


def _normalize_sec_item(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    match = re.search(r"([1-9])\.(\d{1,2})", text)
    if not match:
        return text
    return f"{int(match.group(1))}.{int(match.group(2)):02d}"


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("").value_counts().sort_index().items()
        if str(key)
    }


def _blocker_counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        for blocker in _list(value):
            text = str(blocker)
            if text:
                counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            converted = value.tolist()
        except Exception:
            converted = value
        if isinstance(converted, list):
            return converted
    if pd.isna(value):
        return []
    return [value]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
