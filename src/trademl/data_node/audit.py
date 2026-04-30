"""Capability audit runner for vendor lanes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from trademl.connectors.base import BaseConnector, ConnectorError, PermanentConnectorError, TemporaryConnectorError
from trademl.data_node.capabilities import VendorCapability, capability_registry, provider_role_matrix

ENTITLEMENT_AUDIT_MARKERS = (
    "403",
    "402",
    "not_entitled",
    "not entitled",
    "not permitted",
    "forbidden",
    "permission to access",
    "subscription",
)


@dataclass(slots=True, frozen=True)
class CapabilityAuditResult:
    """Live audit outcome for a capability."""

    capability_id: str
    vendor: str
    dataset: str
    endpoint: str
    doc_status: str
    live_status: str
    enable_status: str
    checked_at: str
    elapsed_ms: float
    rows: int
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for JSON reports."""
        return {
            "capability_id": self.capability_id,
            "vendor": self.vendor,
            "dataset": self.dataset,
            "endpoint": self.endpoint,
            "doc_status": self.doc_status,
            "live_status": self.live_status,
            "enable_status": self.enable_status,
            "checked_at": self.checked_at,
            "elapsed_ms": self.elapsed_ms,
            "rows": self.rows,
            "message": self.message,
            "lane_status": _lane_status(
                live_status=self.live_status,
                enable_status=self.enable_status,
            ),
        }


def run_capability_audit(
    *,
    connectors: dict[str, BaseConnector],
    output_path: Path | None = None,
    capabilities: list[VendorCapability] | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Run lightweight live canaries for registry capabilities and persist the audit report."""
    anchor = pd.Timestamp(as_of or datetime.now(tz=UTC).date().isoformat()).normalize()
    checked_at = datetime.now(tz=UTC).isoformat()
    results: list[CapabilityAuditResult] = []
    for capability in capabilities or capability_registry():
        connector = connectors.get(capability.vendor)
        if connector is None:
            results.append(
                CapabilityAuditResult(
                    capability_id=capability.capability_id,
                    vendor=capability.vendor,
                    dataset=capability.dataset,
                    endpoint=capability.endpoint,
                    doc_status=capability.doc_status,
                    live_status="live_failed",
                    enable_status="disabled",
                    checked_at=checked_at,
                    elapsed_ms=0.0,
                    rows=0,
                    message="connector unavailable",
                )
            )
            continue
        start_date, end_date, symbols = _sample_request(capability, anchor=anchor)
        started = perf_counter()
        try:
            if hasattr(connector, "fetch_audit_sample"):
                frame = connector.fetch_audit_sample(capability.dataset, symbols, start_date, end_date)  # type: ignore[attr-defined]
            else:
                frame = connector.fetch(capability.dataset, symbols, start_date, end_date)
        except PermanentConnectorError as exc:
            message = str(exc)
            live_status = "entitlement_blocked" if _looks_like_entitlement_failure(message) else "live_failed"
            results.append(
                CapabilityAuditResult(
                    capability_id=capability.capability_id,
                    vendor=capability.vendor,
                    dataset=capability.dataset,
                    endpoint=capability.endpoint,
                    doc_status=capability.doc_status,
                    live_status=live_status,
                    enable_status="disabled",
                    checked_at=checked_at,
                    elapsed_ms=(perf_counter() - started) * 1000,
                    rows=0,
                    message=message,
                )
            )
            continue
        except (TemporaryConnectorError, ConnectorError, Exception) as exc:
            results.append(
                CapabilityAuditResult(
                    capability_id=capability.capability_id,
                    vendor=capability.vendor,
                    dataset=capability.dataset,
                    endpoint=capability.endpoint,
                    doc_status=capability.doc_status,
                    live_status="live_failed",
                    enable_status="disabled",
                    checked_at=checked_at,
                    elapsed_ms=(perf_counter() - started) * 1000,
                    rows=0,
                    message=str(exc),
                )
            )
            continue
        effective_enable = capability.enable_status if capability.doc_status == "doc_verified" else "disabled"
        results.append(
            CapabilityAuditResult(
                capability_id=capability.capability_id,
                vendor=capability.vendor,
                dataset=capability.dataset,
                endpoint=capability.endpoint,
                doc_status=capability.doc_status,
                live_status="live_verified",
                enable_status=effective_enable,
                checked_at=checked_at,
                elapsed_ms=(perf_counter() - started) * 1000,
                rows=int(len(frame)),
                message=None,
            )
        )

    report = {
        "checked_at": checked_at,
        "capabilities": {result.capability_id: result.to_dict() for result in results},
        "provider_roles": provider_role_matrix(connectors=connectors),
        "summary": summarize_audit_results(results),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def summarize_audit_results(results: list[CapabilityAuditResult]) -> dict[str, Any]:
    """Summarize audit results for dashboard consumption."""
    counts = {"live_verified": 0, "live_failed": 0, "entitlement_blocked": 0}
    enabled = {"core": 0, "supplemental": 0, "research_only": 0, "disabled": 0}
    failures: list[str] = []
    for result in results:
        counts[result.live_status] = counts.get(result.live_status, 0) + 1
        enabled[result.enable_status] = enabled.get(result.enable_status, 0) + 1
        if result.live_status == "live_failed":
            failures.append(f"{result.capability_id}: {result.message or 'unknown'}")
    return {"live_status": counts, "enable_status": enabled, "failures": failures[:25]}


def _lane_status(*, live_status: str, enable_status: str) -> str:
    """Return the coarse entitlement status used by scheduler/operator surfaces."""
    if live_status == "entitlement_blocked":
        return "ENTITLEMENT_BLOCKED"
    if live_status == "live_failed":
        return "AUDIT_FAILED"
    if enable_status == "disabled":
        return "DISABLED"
    return "ENABLED"


def _looks_like_entitlement_failure(message: str) -> bool:
    """Return whether an audit failure means the key is not entitled to a lane."""
    text = str(message or "").lower()
    return any(marker in text for marker in ENTITLEMENT_AUDIT_MARKERS)


def _sample_request(capability: VendorCapability, *, anchor: pd.Timestamp) -> tuple[str, str, list[str]]:
    """Return a tiny live canary request for a capability."""
    end_date = anchor.strftime("%Y-%m-%d")
    recent_start = (anchor - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    annual_start = (anchor - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    if capability.dataset == "equities_eod":
        return recent_start, end_date, ["AAPL"]
    if capability.dataset == "equities_minute":
        return (anchor - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), end_date, ["AAPL"]
    if capability.dataset in {"stock_trades", "stock_quotes", "stock_bars_boats", "stock_bars_otc"}:
        return (anchor - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), end_date, ["AAPL"]
    if capability.dataset == "stock_snapshots":
        return end_date, end_date, ["AAPL"]
    if capability.dataset in {"crypto_bars", "crypto_trades", "crypto_quotes"}:
        return (anchor - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), end_date, ["BTC/USD"]
    if capability.dataset == "crypto_snapshots":
        return end_date, end_date, ["BTC/USD"]
    if capability.dataset == "option_chain_reference":
        return end_date, end_date, ["SPY"]
    if capability.dataset == "option_snapshots":
        return end_date, end_date, ["SPY260116C00500000"]
    if capability.dataset == "option_bars":
        return (anchor - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), end_date, ["SPY260116C00500000"]
    if capability.dataset in {"corp_actions", "reference_dividends", "reference_splits", "dividends", "splits"}:
        return annual_start, end_date, ["AAPL"]
    if capability.dataset in {
        "financial_statements",
        "company_profile",
        "price_target",
        "insider_transactions",
        "news",
        "company_news",
        "news_sentiment",
        "stock_news",
        "press_releases",
    }:
        return annual_start, end_date, ["AAPL"]
    if capability.dataset == "filing_index":
        return annual_start, end_date, ["320193"]
    if capability.dataset == "companyfacts":
        return annual_start, end_date, ["320193"]
    if capability.dataset in {"macros_treasury", "vintagedates"}:
        return annual_start, end_date, ["DGS10"]
    return annual_start, end_date, []
