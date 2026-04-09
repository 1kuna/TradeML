from __future__ import annotations

from trademl.data_node.provider_contracts import dataset_contract, default_vendor_limits, provider_contract


def test_provider_contract_exposes_canonical_contracts() -> None:
    alpaca = provider_contract("alpaca")
    tiingo_equities = dataset_contract("tiingo", "equities_eod")
    massive_equities = dataset_contract("massive", "equities_eod")
    massive_reference = dataset_contract("massive", "reference_tickers")

    assert alpaca is not None
    assert alpaca.rpm == 190
    assert dataset_contract("alpaca", "equities_eod").retry_after_header == "X-RateLimit-Reset"
    assert tiingo_equities is not None
    assert tiingo_equities.max_batch_symbols == 1
    assert tiingo_equities.critical_path_allowed is True
    assert massive_equities is not None
    assert massive_equities.critical_path_allowed is False
    assert massive_equities.pagination_mode == "next_url"
    assert massive_reference is not None
    assert massive_reference.pagination_mode == "next_url"
    assert dataset_contract("fmp", "delistings").pagination_mode == "page"
    assert dataset_contract("fmp", "delistings").pagination_limit == 100


def test_default_vendor_limits_are_derived_from_provider_contracts() -> None:
    limits = default_vendor_limits()

    assert limits["alpaca"] == {"rpm": 190, "daily_cap": 273600}
    assert limits["twelve_data"] == {"rpm": 7, "daily_cap": 760}
