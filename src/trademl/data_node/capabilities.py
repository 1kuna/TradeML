"""Capability registry for vendor dataset lanes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json


TASK_KINDS = {"FORWARD", "BACKFILL", "REFERENCE", "EVENT", "MACRO", "RESEARCH_ONLY"}
DOC_STATUSES = {"doc_verified", "doc_unverified"}
LIVE_STATUSES = {"live_verified", "live_failed", "entitlement_blocked"}
ENABLE_STATUSES = {"core", "supplemental", "research_only", "disabled"}
QC_ROLES = {"disabled", "primary", "independent", "supplemental"}
SATURATION_POLICIES = {"canonical_first", "canonical_only", "reference_only", "events_only", "macro_only", "balanced", "research_only"}


@dataclass(slots=True, frozen=True)
class VendorCapability:
    """Definition for a single vendor dataset lane."""

    capability_id: str
    vendor: str
    dataset: str
    endpoint: str
    auth_mode: str
    batching_mode: str
    pagination_mode: str
    task_kind: str
    tier: str
    enable_status: str
    doc_status: str
    live_status: str
    expected_history_years: int
    required_fields: tuple[str, ...]
    priority: int
    lane_width: int
    output_name: str | None = None
    planner_group: str = "supplemental_research_backlog"
    max_symbols_per_run: int = 0
    explode_symbols: bool = True
    rotation_key: str | None = None
    preferred_batch_size: int = 1
    qc_role: str = "disabled"
    doc_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the capability for dashboard and audit reports."""
        payload = asdict(self)
        payload["required_fields"] = list(self.required_fields)
        payload["doc_urls"] = list(self.doc_urls)
        return payload


@dataclass(slots=True, frozen=True)
class VendorProfile:
    """Docs-backed role and saturation guidance for a vendor."""

    vendor: str
    primary_use: str
    best_for: tuple[str, ...]
    avoid_for: tuple[str, ...]
    saturation_policy: str
    qc_policy: str
    documented_extra_lanes: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile for audit reports and dashboard use."""
        payload = asdict(self)
        payload["best_for"] = list(self.best_for)
        payload["avoid_for"] = list(self.avoid_for)
        payload["documented_extra_lanes"] = list(self.documented_extra_lanes)
        return payload


def capability_registry() -> list[VendorCapability]:
    """Return the canonical capability registry."""
    return sorted(_CAPABILITIES, key=lambda item: (item.priority, item.capability_id))


def capability_map() -> dict[str, VendorCapability]:
    """Return the registry indexed by capability identifier."""
    return {capability.capability_id: capability for capability in capability_registry()}


def effective_capabilities(
    *,
    connectors: dict[str, object] | None = None,
    audit_state: dict[str, Any] | None = None,
    include_research: bool = False,
) -> list[VendorCapability]:
    """Return capabilities enabled after applying any audit overrides."""
    available_vendors = set(connectors or {})
    effective: list[VendorCapability] = []
    for capability in capability_registry():
        if available_vendors and capability.vendor not in available_vendors:
            continue
        status = effective_enable_status(capability, audit_state=audit_state)
        if status == "disabled":
            continue
        if status == "research_only" and not include_research:
            continue
        effective.append(capability)
    return effective


def effective_enable_status(capability: VendorCapability, *, audit_state: dict[str, Any] | None = None) -> str:
    """Return the enabled state after applying any persisted audit result."""
    record = (audit_state or {}).get("capabilities", {}).get(capability.capability_id, {})
    doc_status = str(record.get("doc_status", capability.doc_status))
    live_status = str(record.get("live_status", capability.live_status))
    enable_status = str(record.get("enable_status", capability.enable_status))
    if doc_status not in DOC_STATUSES or live_status not in LIVE_STATUSES or enable_status not in ENABLE_STATUSES:
        return "disabled"
    if doc_status != "doc_verified":
        return "disabled"
    if live_status not in {"live_verified", "entitlement_blocked"}:
        return "disabled"
    return enable_status


def backfill_capabilities(
    *,
    dataset: str,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
) -> list[VendorCapability]:
    """Return enabled backfill lanes for a dataset ordered by priority."""
    return [
        capability
        for capability in effective_capabilities(connectors=connectors, audit_state=audit_state)
        if capability.task_kind == "BACKFILL" and capability.dataset == dataset
    ]


def forward_capabilities(
    *,
    dataset: str,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
) -> list[VendorCapability]:
    """Return enabled forward lanes for a dataset ordered by priority."""
    return [
        capability
        for capability in effective_capabilities(connectors=connectors, audit_state=audit_state)
        if capability.task_kind == "FORWARD" and capability.dataset == dataset
    ]


def auxiliary_capabilities(
    *,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
    include_research: bool = False,
) -> list[VendorCapability]:
    """Return enabled non-canonical collection lanes."""
    return [
        capability
        for capability in effective_capabilities(
            connectors=connectors,
            audit_state=audit_state,
            include_research=include_research,
        )
        if capability.task_kind in {"REFERENCE", "EVENT", "MACRO", "RESEARCH_ONLY"}
    ]


def build_reference_jobs(
    *,
    connectors: dict[str, object],
    symbols: list[str],
    audit_state: dict[str, Any] | None = None,
    include_research: bool = False,
) -> list[dict[str, object]]:
    """Materialize enabled auxiliary capabilities into collection jobs."""
    jobs: list[dict[str, object]] = []
    for capability in auxiliary_capabilities(
        connectors=connectors,
        audit_state=audit_state,
        include_research=include_research,
    ):
        if capability.task_kind not in {"REFERENCE", "EVENT"}:
            continue
        job: dict[str, object] = {
            "source": capability.vendor,
            "dataset": capability.dataset,
            "symbols": list(symbols) if capability.batching_mode != "global" else [],
            "output_name": capability.output_name or capability.dataset,
            "tier": capability.tier,
            "capability_id": capability.capability_id,
            "planner_group": capability.planner_group,
            "explode_symbols": capability.explode_symbols,
        }
        if capability.max_symbols_per_run:
            job["max_symbols_per_run"] = capability.max_symbols_per_run
        if capability.rotation_key:
            job["rotation_key"] = capability.rotation_key
        jobs.append(job)
    return jobs


def default_macro_series() -> list[str]:
    """Return the default PIT macro pack."""
    return ["DGS10", "DGS2", "DFF", "DTB3", "CPIAUCSL", "UNRATE", "VIXCLS"]


def canonical_qc_capabilities(
    *,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
) -> list[VendorCapability]:
    """Return the enabled vendor lanes suitable for cross-vendor bar QC."""
    candidates = [
        capability
        for capability in effective_capabilities(connectors=connectors, audit_state=audit_state)
        if capability.dataset == "equities_eod" and capability.task_kind in {"FORWARD", "BACKFILL"} and capability.qc_role != "disabled"
    ]
    role_order = {"independent": 0, "supplemental": 1, "primary": 2}
    ordered = sorted(
        candidates,
        key=lambda capability: (role_order.get(capability.qc_role, 99), capability.priority, capability.vendor),
    )
    deduped: list[VendorCapability] = []
    seen_vendors: set[str] = set()
    for capability in ordered:
        if capability.vendor in seen_vendors:
            continue
        deduped.append(capability)
        seen_vendors.add(capability.vendor)
    return deduped


def vendor_profile(vendor: str) -> VendorProfile | None:
    """Return the docs-backed vendor role profile."""
    return _VENDOR_PROFILES.get(vendor)


def provider_role_matrix(
    *,
    connectors: dict[str, object] | None = None,
    audit_state: dict[str, Any] | None = None,
    include_disabled: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate vendor roles, enabled lanes, and saturation guidance."""
    available_vendors = set(connectors or {})
    rows: list[dict[str, Any]] = []
    for vendor, profile in sorted(_VENDOR_PROFILES.items()):
        if available_vendors and vendor not in available_vendors:
            continue
        enabled_capabilities = [
            capability
            for capability in capability_registry()
            if capability.vendor == vendor
            and (include_disabled or effective_enable_status(capability, audit_state=audit_state) != "disabled")
        ]
        enabled_lanes = [
            capability.capability_id
            for capability in enabled_capabilities
            if effective_enable_status(capability, audit_state=audit_state) != "disabled"
        ]
        enabled_datasets = sorted(
            {
                capability.dataset
                for capability in enabled_capabilities
                if effective_enable_status(capability, audit_state=audit_state) != "disabled"
            }
        )
        enabled_task_kinds = sorted(
            {
                capability.task_kind
                for capability in enabled_capabilities
                if effective_enable_status(capability, audit_state=audit_state) != "disabled"
            }
        )
        doc_urls = sorted({url for capability in enabled_capabilities for url in capability.doc_urls})
        row = profile.to_dict()
        row.update(
            {
                "enabled_lanes": enabled_lanes,
                "enabled_datasets": enabled_datasets,
                "enabled_task_kinds": enabled_task_kinds,
                "doc_urls": doc_urls,
            }
        )
        rows.append(row)
    return rows


def load_audit_state(path: Path) -> dict[str, Any]:
    """Load the persisted audit state if present."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


_CAPABILITIES: tuple[VendorCapability, ...] = (
    VendorCapability(
        capability_id="alpaca.equities_eod.forward",
        vendor="alpaca",
        dataset="equities_eod",
        endpoint="/v2/stocks/bars",
        auth_mode="headers",
        batching_mode="multi_symbol",
        pagination_mode="token",
        task_kind="FORWARD",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=10,
        lane_width=2,
        planner_group="canonical_bars_backlog",
        preferred_batch_size=100,
        qc_role="primary",
        doc_urls=(
            "https://docs.alpaca.markets/reference/stockbars",
            "https://docs.alpaca.markets/reference/get-v2-assets",
        ),
    ),
    VendorCapability(
        capability_id="tiingo.equities_eod.forward",
        vendor="tiingo",
        dataset="equities_eod",
        endpoint="/tiingo/daily/{ticker}/prices",
        auth_mode="token_header",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="FORWARD",
        tier="A",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=30,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=15,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        qc_role="supplemental",
        doc_urls=("https://www.tiingo.com/kb/article/where-to-find-your-tiingo-api-token/",),
    ),
    VendorCapability(
        capability_id="twelve_data.equities_eod.forward",
        vendor="twelve_data",
        dataset="equities_eod",
        endpoint="/time_series",
        auth_mode="apikey_query",
        batching_mode="multi_symbol",
        pagination_mode="none",
        task_kind="FORWARD",
        tier="A",
        enable_status="disabled",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=16,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        preferred_batch_size=8,
        qc_role="disabled",
        doc_urls=(
            "https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data",
            "https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",
        ),
    ),
    VendorCapability(
        capability_id="massive.equities_eod.forward",
        vendor="massive",
        dataset="equities_eod",
        endpoint="/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="FORWARD",
        tier="A",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=17,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        qc_role="independent",
        doc_urls=("https://massive.com/docs/rest/stocks/corporate-actions/dividends",),
    ),
    VendorCapability(
        capability_id="finnhub.equities_eod.forward",
        vendor="finnhub",
        dataset="equities_eod",
        endpoint="/api/v1/stock/candle",
        auth_mode="token_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="FORWARD",
        tier="A",
        enable_status="disabled",
        doc_status="doc_verified",
        live_status="entitlement_blocked",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=18,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        doc_urls=("https://finnhub.io/docs/api",),
    ),
    VendorCapability(
        capability_id="tiingo.equities_eod.backfill",
        vendor="tiingo",
        dataset="equities_eod",
        endpoint="/tiingo/daily/{ticker}/prices",
        auth_mode="token_header",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="BACKFILL",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=30,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=20,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        qc_role="supplemental",
        doc_urls=("https://www.tiingo.com/kb/article/where-to-find-your-tiingo-api-token/",),
    ),
    VendorCapability(
        capability_id="alpaca.equities_eod.backfill",
        vendor="alpaca",
        dataset="equities_eod",
        endpoint="/v2/stocks/bars",
        auth_mode="headers",
        batching_mode="multi_symbol",
        pagination_mode="token",
        task_kind="BACKFILL",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=30,
        lane_width=2,
        planner_group="canonical_bars_backlog",
        preferred_batch_size=100,
        qc_role="primary",
        doc_urls=("https://docs.alpaca.markets/reference/stockbars",),
    ),
    VendorCapability(
        capability_id="twelve_data.equities_eod.backfill",
        vendor="twelve_data",
        dataset="equities_eod",
        endpoint="/time_series",
        auth_mode="apikey_query",
        batching_mode="multi_symbol",
        pagination_mode="none",
        task_kind="BACKFILL",
        tier="A",
        enable_status="disabled",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=40,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        preferred_batch_size=8,
        qc_role="disabled",
        doc_urls=(
            "https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data",
            "https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",
        ),
    ),
    VendorCapability(
        capability_id="massive.equities_eod.backfill",
        vendor="massive",
        dataset="equities_eod",
        endpoint="/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="BACKFILL",
        tier="A",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=50,
        lane_width=1,
        planner_group="canonical_bars_backlog",
        qc_role="independent",
        doc_urls=("https://massive.com/docs/rest/stocks/corporate-actions/dividends",),
    ),
    VendorCapability(
        capability_id="finnhub.equities_eod.backfill",
        vendor="finnhub",
        dataset="equities_eod",
        endpoint="/api/v1/stock/candle",
        auth_mode="token_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="BACKFILL",
        tier="A",
        enable_status="disabled",
        doc_status="doc_verified",
        live_status="entitlement_blocked",
        expected_history_years=5,
        required_fields=("date", "symbol", "open", "high", "low", "close", "volume"),
        priority=60,
        lane_width=2,
        planner_group="canonical_bars_backlog",
        doc_urls=("https://finnhub.io/docs/api",),
    ),
    VendorCapability(
        capability_id="alpaca.assets.reference",
        vendor="alpaca",
        dataset="assets",
        endpoint="/v2/assets",
        auth_mode="headers",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("symbol", "exchange", "status", "tradable"),
        priority=70,
        lane_width=1,
        output_name="alpaca_assets",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://docs.alpaca.markets/reference/get-v2-assets",),
    ),
    VendorCapability(
        capability_id="alpha_vantage.listings.reference",
        vendor="alpha_vantage",
        dataset="listings",
        endpoint="/query?function=LISTING_STATUS",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("symbol", "name", "exchange", "ipoDate"),
        priority=80,
        lane_width=1,
        output_name="listings",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://www.alphavantage.co/documentation/",),
    ),
    VendorCapability(
        capability_id="massive.reference_tickers.reference",
        vendor="massive",
        dataset="reference_tickers",
        endpoint="/v3/reference/tickers",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="cursor",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("ticker", "name", "active"),
        priority=90,
        lane_width=1,
        output_name="universe",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://massive.com/docs/rest/stocks/reference/tickers/all-tickers",),
    ),
    VendorCapability(
        capability_id="twelve_data.stocks.reference",
        vendor="twelve_data",
        dataset="stocks",
        endpoint="/stocks",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("symbol", "exchange", "type"),
        priority=100,
        lane_width=1,
        output_name="twelve_data_stocks",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://support.twelvedata.com/en/articles/5620513-how-to-find-all-available-symbols-at-twelve-data",),
    ),
    VendorCapability(
        capability_id="fmp.delistings.reference",
        vendor="fmp",
        dataset="delistings",
        endpoint="/stable/delisted-companies",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("symbol", "companyName", "delistedDate"),
        priority=110,
        lane_width=1,
        output_name="delistings",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://site.financialmodelingprep.com/developer/docs/stable/delisted-companies",),
    ),
    VendorCapability(
        capability_id="fmp.symbol_changes.reference",
        vendor="fmp",
        dataset="symbol_changes",
        endpoint="/stable/symbol-change",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="disabled",
        doc_status="doc_verified",
        live_status="entitlement_blocked",
        expected_history_years=0,
        required_fields=("oldSymbol", "newSymbol", "date"),
        priority=120,
        lane_width=1,
        output_name="symbol_changes",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://site.financialmodelingprep.com/developer/docs/stable/symbol-changes",),
    ),
    VendorCapability(
        capability_id="alpaca.corp_actions.reference",
        vendor="alpaca",
        dataset="corp_actions",
        endpoint="/v2/stocks/{symbol}/corporate_actions/announcements",
        auth_mode="headers",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=10,
        required_fields=("symbol", "event_type", "ex_date"),
        priority=125,
        lane_width=1,
        output_name="corp_actions",
        planner_group="reference_events_backlog",
        max_symbols_per_run=25,
        rotation_key="alpaca:corp_actions",
        doc_urls=("https://docs.alpaca.markets/reference/corporateactions-1",),
    ),
    VendorCapability(
        capability_id="alpha_vantage.corp_actions.reference",
        vendor="alpha_vantage",
        dataset="corp_actions",
        endpoint="/query?function=DIVIDENDS|SPLITS",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("symbol", "event_type", "ex_date"),
        priority=130,
        lane_width=1,
        output_name="corp_actions",
        planner_group="reference_events_backlog",
        max_symbols_per_run=10,
        rotation_key="alpha_vantage:corp_actions",
        doc_urls=("https://www.alphavantage.co/documentation/",),
    ),
    VendorCapability(
        capability_id="massive.reference_dividends.reference",
        vendor="massive",
        dataset="reference_dividends",
        endpoint="/v3/reference/dividends",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="cursor",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("symbol", "cash_amount", "ex_dividend_date"),
        priority=140,
        lane_width=1,
        output_name="dividends",
        planner_group="reference_events_backlog",
        max_symbols_per_run=10,
        rotation_key="massive:reference_dividends",
        doc_urls=("https://massive.com/docs/rest/stocks/corporate-actions/dividends",),
    ),
    VendorCapability(
        capability_id="massive.reference_splits.reference",
        vendor="massive",
        dataset="reference_splits",
        endpoint="/v3/reference/splits",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="cursor",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("symbol", "execution_date"),
        priority=150,
        lane_width=1,
        output_name="splits",
        planner_group="reference_events_backlog",
        max_symbols_per_run=10,
        rotation_key="massive:reference_splits",
        doc_urls=("https://massive.com/docs/rest/stocks/corporate-actions/splits",),
    ),
    VendorCapability(
        capability_id="twelve_data.dividends.reference",
        vendor="twelve_data",
        dataset="dividends",
        endpoint="/dividends",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("symbol", "event_type", "ex_date"),
        priority=160,
        lane_width=1,
        output_name="corp_actions",
        planner_group="reference_events_backlog",
        max_symbols_per_run=50,
        rotation_key="twelve_data:dividends",
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
    VendorCapability(
        capability_id="twelve_data.splits.reference",
        vendor="twelve_data",
        dataset="splits",
        endpoint="/splits",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=20,
        required_fields=("symbol", "event_type", "ex_date"),
        priority=170,
        lane_width=1,
        output_name="corp_actions",
        planner_group="reference_events_backlog",
        max_symbols_per_run=50,
        rotation_key="twelve_data:splits",
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
    VendorCapability(
        capability_id="sec_edgar.company_tickers.reference",
        vendor="sec_edgar",
        dataset="company_tickers",
        endpoint="/files/company_tickers.json",
        auth_mode="user_agent",
        batching_mode="global",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("ticker", "cik_str"),
        priority=180,
        lane_width=1,
        output_name="sec_company_tickers",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
    ),
    VendorCapability(
        capability_id="sec_edgar.filing_index.event",
        vendor="sec_edgar",
        dataset="filing_index",
        endpoint="/submissions/CIK{cik}.json",
        auth_mode="user_agent",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="EVENT",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=15,
        required_fields=("cik", "form", "filingDate"),
        priority=190,
        lane_width=2,
        output_name="sec_filings",
        planner_group="reference_events_backlog",
        max_symbols_per_run=50,
        rotation_key="sec_edgar:filing_index",
        doc_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
    ),
    VendorCapability(
        capability_id="fred.macros_treasury.macro",
        vendor="fred",
        dataset="macros_treasury",
        endpoint="/fred/series/observations",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="MACRO",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=30,
        required_fields=("series_id", "observation_date", "value"),
        priority=200,
        lane_width=2,
        output_name="macros_fred",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://fred.stlouisfed.org/docs/api/fred/series_observations.html",),
    ),
    VendorCapability(
        capability_id="fred.vintagedates.macro",
        vendor="fred",
        dataset="vintagedates",
        endpoint="/fred/series/vintagedates",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="MACRO",
        tier="A",
        enable_status="core",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=30,
        required_fields=("series_id", "vintage_date"),
        priority=210,
        lane_width=2,
        output_name="fred_vintagedates",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",),
    ),
    VendorCapability(
        capability_id="finnhub.earnings_calendar.event",
        vendor="finnhub",
        dataset="earnings_calendar",
        endpoint="/api/v1/calendar/earnings",
        auth_mode="token_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="EVENT",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date",),
        priority=220,
        lane_width=1,
        output_name="earnings_calendar",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://finnhub.io/docs/api",),
    ),
    VendorCapability(
        capability_id="fmp.earnings_calendar.event",
        vendor="fmp",
        dataset="earnings_calendar",
        endpoint="/stable/earnings-calendar",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="EVENT",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date",),
        priority=230,
        lane_width=1,
        output_name="earnings_calendar_fmp",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://site.financialmodelingprep.com/developer/docs/stable/earnings-calendar",),
    ),
    VendorCapability(
        capability_id="twelve_data.earnings_calendar.event",
        vendor="twelve_data",
        dataset="earnings_calendar",
        endpoint="/earnings",
        auth_mode="apikey_query",
        batching_mode="global",
        pagination_mode="none",
        task_kind="EVENT",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=5,
        required_fields=("date",),
        priority=240,
        lane_width=1,
        output_name="earnings_calendar_twelve_data",
        planner_group="reference_events_backlog",
        explode_symbols=False,
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
    VendorCapability(
        capability_id="finnhub.company_profile.reference",
        vendor="finnhub",
        dataset="company_profile",
        endpoint="/api/v1/stock/profile2",
        auth_mode="token_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=0,
        required_fields=("ticker", "name"),
        priority=250,
        lane_width=1,
        output_name="company_profiles",
        planner_group="reference_events_backlog",
        max_symbols_per_run=50,
        rotation_key="finnhub:company_profile",
        doc_urls=("https://finnhub.io/docs/api",),
    ),
    VendorCapability(
        capability_id="twelve_data.financial_statements.reference",
        vendor="twelve_data",
        dataset="financial_statements",
        endpoint="/income_statement|/balance_sheet|/cash_flow",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=10,
        required_fields=("symbol", "statement_type", "date"),
        priority=260,
        lane_width=1,
        output_name="financial_statements_twelve_data",
        planner_group="reference_events_backlog",
        max_symbols_per_run=20,
        rotation_key="twelve_data:financial_statements",
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
    VendorCapability(
        capability_id="sec_edgar.companyfacts.reference",
        vendor="sec_edgar",
        dataset="companyfacts",
        endpoint="/api/xbrl/companyfacts/CIK{cik}.json",
        auth_mode="user_agent",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="REFERENCE",
        tier="B",
        enable_status="supplemental",
        doc_status="doc_verified",
        live_status="live_verified",
        expected_history_years=15,
        required_fields=("cik", "facts"),
        priority=270,
        lane_width=1,
        output_name="sec_companyfacts",
        planner_group="reference_events_backlog",
        max_symbols_per_run=50,
        rotation_key="sec_edgar:companyfacts",
        doc_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
    ),
    VendorCapability(
        capability_id="twelve_data.price_target.research",
        vendor="twelve_data",
        dataset="price_target",
        endpoint="/price_target",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="RESEARCH_ONLY",
        tier="C",
        enable_status="research_only",
        doc_status="doc_verified",
        live_status="entitlement_blocked",
        expected_history_years=0,
        required_fields=("symbol",),
        priority=280,
        lane_width=1,
        output_name="price_targets_twelve_data",
        planner_group="supplemental_research_backlog",
        max_symbols_per_run=20,
        rotation_key="twelve_data:price_target",
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
    VendorCapability(
        capability_id="twelve_data.insider_transactions.research",
        vendor="twelve_data",
        dataset="insider_transactions",
        endpoint="/insider_transactions",
        auth_mode="apikey_query",
        batching_mode="single_symbol",
        pagination_mode="none",
        task_kind="RESEARCH_ONLY",
        tier="C",
        enable_status="research_only",
        doc_status="doc_verified",
        live_status="entitlement_blocked",
        expected_history_years=0,
        required_fields=("symbol",),
        priority=290,
        lane_width=1,
        output_name="insider_transactions_twelve_data",
        planner_group="supplemental_research_backlog",
        max_symbols_per_run=20,
        rotation_key="twelve_data:insider_transactions",
        doc_urls=("https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request",),
    ),
)


_VENDOR_PROFILES: dict[str, VendorProfile] = {
    "alpaca": VendorProfile(
        vendor="alpaca",
        primary_use="same-day and recent-history canonical bars",
        best_for=("multi-symbol daily bars", "asset universe snapshots", "recent corporate actions"),
        avoid_for=("deep-history saturation beyond current entitlement window",),
        saturation_policy="canonical_first",
        qc_policy="primary source only; do not use as the backup comparator",
        documented_extra_lanes=("trading API metadata",),
        notes="Use Alpaca to cover the fast moving edge and multi-symbol batching efficiently.",
    ),
    "tiingo": VendorProfile(
        vendor="tiingo",
        primary_use="deep-history daily bars",
        best_for=("long-tail EOD history",),
        avoid_for=("broad reference fanout", "price QC when minute/day budget is tight"),
        saturation_policy="canonical_only",
        qc_policy="optional tertiary bar QC only",
        documented_extra_lanes=("fundamentals", "supported tickers"),
        notes="Treat Tiingo as the history specialist and keep it focused on bar backfill.",
    ),
    "twelve_data": VendorProfile(
        vendor="twelve_data",
        primary_use="broad fundamentals, reference, and event support",
        best_for=("corp actions", "earnings", "financial statements", "reference coverage"),
        avoid_for=("canonical bars", "bar QC", "unbounded low-value research sweeps on free credits"),
        saturation_policy="reference_only",
        qc_policy="disabled for bars",
        documented_extra_lanes=("price targets", "insider transactions"),
        notes="Keep Twelve Data focused on reference/event/fundamental coverage; do not spend its free-plan credits on canonical OHLCV.",
    ),
    "massive": VendorProfile(
        vendor="massive",
        primary_use="independent bar QC and reference validation",
        best_for=("independent OHLCV cross-checks", "ticker reference", "splits", "dividends"),
        avoid_for=("primary large-scale backfill under the free minute cap",),
        saturation_policy="canonical_first",
        qc_policy="preferred independent QC vendor",
        documented_extra_lanes=("ticker details", "additional reference endpoints"),
        notes="Keep Massive focused on validation and targeted overflow rather than bulk collection.",
    ),
    "finnhub": VendorProfile(
        vendor="finnhub",
        primary_use="event and company reference support",
        best_for=("earnings calendar", "company profiles"),
        avoid_for=("canonical bars", "bar QC"),
        saturation_policy="reference_only",
        qc_policy="disabled for bars until candle entitlement is re-verified",
        documented_extra_lanes=("peers", "recommendation trends", "company news", "financials-reported"),
        notes="Finnhub is useful for event/reference context, not current canonical OHLCV.",
    ),
    "alpha_vantage": VendorProfile(
        vendor="alpha_vantage",
        primary_use="low-rate listing status and supplemental corp actions",
        best_for=("listing status", "low-frequency corp actions"),
        avoid_for=("high-volume workflows",),
        saturation_policy="reference_only",
        qc_policy="disabled for bars",
        documented_extra_lanes=("additional fundamentals functions"),
        notes="Use Alpha Vantage as a corroborating low-rate reference lane.",
    ),
    "fred": VendorProfile(
        vendor="fred",
        primary_use="macro observations and vintages",
        best_for=("macro pack", "ALFRED vintages"),
        avoid_for=("equity/reference tasks",),
        saturation_policy="macro_only",
        qc_policy="not applicable",
        documented_extra_lanes=("broader macro series packs"),
        notes="FRED/ALFRED should always keep the macro lane full because it does not compete with bar vendors.",
    ),
    "fmp": VendorProfile(
        vendor="fmp",
        primary_use="delistings and supplemental event coverage",
        best_for=("delistings", "earnings calendar"),
        avoid_for=("entitlement-gated symbol changes unless re-verified",),
        saturation_policy="reference_only",
        qc_policy="disabled for bars",
        documented_extra_lanes=("historical price EOD light", "symbol changes"),
        notes="Use only the stable free endpoints unless a fresh audit restores broader access.",
    ),
    "sec_edgar": VendorProfile(
        vendor="sec_edgar",
        primary_use="authoritative filing timeline and company linkage",
        best_for=("company tickers", "filing index", "companyfacts"),
        avoid_for=("bar/reference work outside SEC filings",),
        saturation_policy="events_only",
        qc_policy="not applicable",
        documented_extra_lanes=("bulk submissions archives", "bulk companyfacts archives"),
        notes="SEC is the authoritative PIT event lane and should keep moving independently of bar collection.",
    ),
}
