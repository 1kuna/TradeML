from __future__ import annotations

from trademl.data_node.provider_contracts import (
    dataset_contract,
    default_vendor_limits,
    provider_contract,
)


def test_provider_contract_exposes_canonical_contracts() -> None:
    alpaca = provider_contract("alpaca")
    tiingo_equities = dataset_contract("tiingo", "equities_eod")
    massive_equities = dataset_contract("massive", "equities_eod")
    massive_reference = dataset_contract("massive", "reference_tickers")

    assert alpaca is not None
    assert alpaca.rpm == 200
    assert (
        dataset_contract("alpaca", "equities_eod").retry_after_header
        == "X-RateLimit-Reset"
    )
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
    assert dataset_contract("alpaca", "equities_minute").pagination_mode == "page_token"
    assert dataset_contract("alpaca", "equities_minute").pagination_limit == 10000
    assert dataset_contract("alpaca", "equities_minute").critical_path_allowed is False
    assert dataset_contract("massive", "equities_minute").pagination_mode == "next_url"
    assert dataset_contract("massive", "equities_minute").critical_path_allowed is False
    assert dataset_contract("alpaca", "news").pagination_mode == "page_token"
    assert dataset_contract("alpaca", "stock_trades").pagination_limit == 10000
    assert dataset_contract("alpaca", "stock_quotes").max_batch_symbols == 100
    assert dataset_contract("alpaca", "stock_snapshots").endpoint_key == "stock_snapshots"
    assert dataset_contract("alpaca", "stock_bars_boats").pagination_mode == "page_token"
    assert dataset_contract("alpaca", "crypto_bars").pagination_limit == 10000
    assert dataset_contract("alpaca", "crypto_quotes").docs_urls
    assert dataset_contract("alpaca", "crypto_snapshots").max_batch_symbols == 100
    assert dataset_contract("alpaca", "crypto_websocket").max_batch_symbols == 30
    assert dataset_contract("alpaca", "option_chain_reference").pagination_limit == 1000
    assert "not trade-approved" in dataset_contract("alpaca", "option_chain_reference").notes
    assert dataset_contract("alpaca", "option_bars").pagination_limit == 10000
    assert dataset_contract("tiingo", "news").max_batch_symbols == 50
    assert "permission to access the news api" in dataset_contract(
        "tiingo", "news"
    ).entitlement_failure_markers
    assert dataset_contract("alpha_vantage", "news_sentiment").max_batch_symbols == 10
    assert dataset_contract("fmp", "stock_news").pagination_mode == "page"
    assert dataset_contract("finnhub", "company_news").max_batch_symbols == 1


def test_default_vendor_limits_are_derived_from_provider_contracts() -> None:
    limits = default_vendor_limits()

    assert limits["alpaca"] == {"rpm": 200, "daily_cap": 288000}
    assert limits["twelve_data"] == {"rpm": 8, "daily_cap": 800}
    assert limits["massive"] == {"rpm": 5, "daily_cap": 7200}
    assert limits["sec_edgar"] == {"rpm": 600, "daily_cap": 864000}
