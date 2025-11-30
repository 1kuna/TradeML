"""
Unit tests for fundamentals curated layer.

Tests the logic in data_layer/curated/fundamentals.py:
- Income statement normalization
- Balance sheet normalization
- Cash flow normalization
- Derived ratio computation
- Point-in-time discipline
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


class TestHelperFunctions:
    """Test helper functions."""

    def test_safe_float_valid(self):
        """Verify valid values are converted to float."""
        from data_layer.curated.fundamentals import _safe_float

        assert _safe_float(100) == 100.0
        assert _safe_float("123.45") == 123.45
        assert _safe_float(0) == 0.0

    def test_safe_float_invalid(self):
        """Verify invalid values return None."""
        from data_layer.curated.fundamentals import _safe_float

        assert _safe_float(None) is None
        assert _safe_float(np.nan) is None
        assert _safe_float("invalid") is None

    def test_compute_ratio_valid(self):
        """Verify ratio computation with valid inputs."""
        from data_layer.curated.fundamentals import _compute_ratio

        assert _compute_ratio(100, 200) == 0.5
        assert _compute_ratio(50, 25) == 2.0

    def test_compute_ratio_zero_denominator(self):
        """Verify zero denominator returns None."""
        from data_layer.curated.fundamentals import _compute_ratio

        assert _compute_ratio(100, 0) is None

    def test_compute_ratio_none_inputs(self):
        """Verify None inputs return None."""
        from data_layer.curated.fundamentals import _compute_ratio

        assert _compute_ratio(None, 100) is None
        assert _compute_ratio(100, None) is None
        assert _compute_ratio(None, None) is None


class TestIncomeNormalization:
    """Test income statement normalization."""

    def test_normalize_income_basic(self):
        """Verify basic income statement normalization."""
        from data_layer.curated.fundamentals import _normalize_fmp_income

        df = pd.DataFrame([{
            "date": "2024-01-15",
            "fillingDate": "2024-02-01",
            "period": "Q4",
            "calendarYear": 2024,
            "revenue": 100_000_000,
            "grossProfit": 40_000_000,
            "operatingIncome": 25_000_000,
            "netIncome": 20_000_000,
            "ebitda": 30_000_000,
            "eps": 1.25,
            "epsdiluted": 1.20,
        }])

        result = _normalize_fmp_income(df, "AAPL")

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["filing_date"] == date(2024, 2, 1)
        assert result.iloc[0]["period_end"] == date(2024, 1, 15)
        assert result.iloc[0]["period_type"] == "quarter"
        assert result.iloc[0]["fiscal_quarter"] == 4
        assert result.iloc[0]["revenue"] == 100_000_000

    def test_normalize_income_annual(self):
        """Verify annual period detection."""
        from data_layer.curated.fundamentals import _normalize_fmp_income

        df = pd.DataFrame([{
            "date": "2023-12-31",
            "fillingDate": "2024-02-15",
            "period": "FY",
            "calendarYear": 2023,
            "revenue": 400_000_000,
        }])

        result = _normalize_fmp_income(df, "MSFT")

        assert result.iloc[0]["period_type"] == "annual"
        assert result.iloc[0]["fiscal_quarter"] is None

    def test_normalize_income_empty(self):
        """Verify empty DataFrame handling."""
        from data_layer.curated.fundamentals import _normalize_fmp_income

        result = _normalize_fmp_income(pd.DataFrame(), "AAPL")
        assert result.empty


class TestBalanceNormalization:
    """Test balance sheet normalization."""

    def test_normalize_balance_basic(self):
        """Verify basic balance sheet normalization."""
        from data_layer.curated.fundamentals import _normalize_fmp_balance

        df = pd.DataFrame([{
            "date": "2024-01-15",
            "fillingDate": "2024-02-01",
            "totalAssets": 500_000_000,
            "totalLiabilities": 200_000_000,
            "totalStockholdersEquity": 300_000_000,
            "cashAndCashEquivalents": 50_000_000,
            "totalDebt": 100_000_000,
            "totalCurrentAssets": 150_000_000,
            "totalCurrentLiabilities": 80_000_000,
        }])

        result = _normalize_fmp_balance(df, "AAPL")

        assert len(result) == 1
        assert result.iloc[0]["total_assets"] == 500_000_000
        assert result.iloc[0]["total_equity"] == 300_000_000
        assert result.iloc[0]["working_capital"] == 70_000_000  # 150M - 80M


class TestCashFlowNormalization:
    """Test cash flow statement normalization."""

    def test_normalize_cashflow_basic(self):
        """Verify basic cash flow normalization."""
        from data_layer.curated.fundamentals import _normalize_fmp_cashflow

        df = pd.DataFrame([{
            "date": "2024-01-15",
            "fillingDate": "2024-02-01",
            "operatingCashFlow": 30_000_000,
            "capitalExpenditure": -5_000_000,
            "freeCashFlow": 25_000_000,
            "dividendsPaid": -2_000_000,
        }])

        result = _normalize_fmp_cashflow(df, "AAPL")

        assert len(result) == 1
        assert result.iloc[0]["operating_cash_flow"] == 30_000_000
        assert result.iloc[0]["free_cash_flow"] == 25_000_000


class TestDerivedRatios:
    """Test derived ratio computation."""

    def test_margin_ratios(self):
        """Verify margin ratios are computed correctly."""
        from data_layer.curated.fundamentals import _compute_derived_ratios

        df = pd.DataFrame([{
            "symbol": "AAPL",
            "period_type": "annual",
            "revenue": 100_000_000,
            "gross_profit": 40_000_000,
            "operating_income": 25_000_000,
            "net_income": 20_000_000,
        }])

        result = _compute_derived_ratios(df)

        assert np.isclose(result.iloc[0]["gross_margin"], 0.40)
        assert np.isclose(result.iloc[0]["operating_margin"], 0.25)
        assert np.isclose(result.iloc[0]["net_margin"], 0.20)

    def test_return_ratios_annual(self):
        """Verify return ratios (ROE, ROA) for annual data."""
        from data_layer.curated.fundamentals import _compute_derived_ratios

        df = pd.DataFrame([{
            "symbol": "AAPL",
            "period_type": "annual",
            "net_income": 20_000_000,
            "total_equity": 100_000_000,
            "total_assets": 200_000_000,
        }])

        result = _compute_derived_ratios(df)

        assert np.isclose(result.iloc[0]["roe"], 0.20)  # 20M / 100M
        assert np.isclose(result.iloc[0]["roa"], 0.10)  # 20M / 200M

    def test_return_ratios_quarterly_annualized(self):
        """Verify quarterly return ratios are annualized (4x)."""
        from data_layer.curated.fundamentals import _compute_derived_ratios

        df = pd.DataFrame([{
            "symbol": "AAPL",
            "period_type": "quarter",
            "net_income": 5_000_000,  # Quarterly income
            "total_equity": 100_000_000,
            "total_assets": 200_000_000,
        }])

        result = _compute_derived_ratios(df)

        # Annualized: 5M * 4 = 20M
        assert np.isclose(result.iloc[0]["roe"], 0.20)  # (5M * 4) / 100M
        assert np.isclose(result.iloc[0]["roa"], 0.10)  # (5M * 4) / 200M

    def test_leverage_ratios(self):
        """Verify leverage ratios (D/E, current ratio)."""
        from data_layer.curated.fundamentals import _compute_derived_ratios

        df = pd.DataFrame([{
            "symbol": "AAPL",
            "period_type": "annual",
            "total_debt": 50_000_000,
            "total_equity": 100_000_000,
            "current_assets": 60_000_000,
            "current_liabilities": 40_000_000,
        }])

        result = _compute_derived_ratios(df)

        assert np.isclose(result.iloc[0]["debt_to_equity"], 0.50)
        assert np.isclose(result.iloc[0]["current_ratio"], 1.50)

    def test_ratio_with_missing_data(self):
        """Verify ratios handle missing data gracefully."""
        from data_layer.curated.fundamentals import _compute_derived_ratios

        df = pd.DataFrame([{
            "symbol": "AAPL",
            "period_type": "annual",
            "revenue": None,
            "gross_profit": 40_000_000,
        }])

        result = _compute_derived_ratios(df)

        # gross_margin should be None since revenue is None
        assert result.iloc[0]["gross_margin"] is None


class TestPointInTime:
    """Test point-in-time discipline."""

    @pytest.fixture
    def sample_fundamentals(self, tmp_path):
        """Create sample fundamentals data."""
        # Create two filings for AAPL
        data = pd.DataFrame([
            {
                "symbol": "AAPL",
                "filing_date": date(2024, 2, 1),  # Q4 2023 filed Feb 1
                "period_end": date(2023, 12, 31),
                "period_type": "quarter",
                "revenue": 100_000_000,
            },
            {
                "symbol": "AAPL",
                "filing_date": date(2024, 5, 1),  # Q1 2024 filed May 1
                "period_end": date(2024, 3, 31),
                "period_type": "quarter",
                "revenue": 110_000_000,
            },
        ])

        # Write to tmp_path
        sym_dir = tmp_path / "symbol=AAPL"
        sym_dir.mkdir(parents=True)
        data.to_parquet(sym_dir / "data.parquet", index=False)

        return tmp_path

    def test_get_pit_fundamentals_before_filing(self, sample_fundamentals):
        """Verify PIT returns only filings known as of date."""
        from data_layer.curated.fundamentals import get_pit_fundamentals

        # As of March 1, 2024, only Q4 2023 filing is known
        result = get_pit_fundamentals(
            symbols=["AAPL"],
            as_of_date=date(2024, 3, 1),
            base_dir=str(sample_fundamentals),
        )

        assert len(result) == 1
        assert result.iloc[0]["filing_date"] == date(2024, 2, 1)
        assert result.iloc[0]["revenue"] == 100_000_000

    def test_get_pit_fundamentals_after_both_filings(self, sample_fundamentals):
        """Verify PIT returns most recent filing when both known."""
        from data_layer.curated.fundamentals import get_pit_fundamentals

        # As of June 1, 2024, both filings are known
        result = get_pit_fundamentals(
            symbols=["AAPL"],
            as_of_date=date(2024, 6, 1),
            base_dir=str(sample_fundamentals),
        )

        assert len(result) == 1
        assert result.iloc[0]["filing_date"] == date(2024, 5, 1)
        assert result.iloc[0]["revenue"] == 110_000_000

    def test_get_pit_fundamentals_before_any_filing(self, sample_fundamentals):
        """Verify PIT returns empty when no filings exist yet."""
        from data_layer.curated.fundamentals import get_pit_fundamentals

        # As of Jan 1, 2024, no filings exist yet
        result = get_pit_fundamentals(
            symbols=["AAPL"],
            as_of_date=date(2024, 1, 1),
            base_dir=str(sample_fundamentals),
        )

        assert result.empty


class TestPanelLoading:
    """Test panel loading functionality."""

    @pytest.fixture
    def sample_panel(self, tmp_path):
        """Create sample fundamentals for multiple symbols."""
        for sym, revenue in [("AAPL", 100_000_000), ("MSFT", 150_000_000)]:
            data = pd.DataFrame([{
                "symbol": sym,
                "filing_date": date(2024, 2, 1),
                "period_end": date(2023, 12, 31),
                "period_type": "quarter",
                "revenue": revenue,
            }])
            sym_dir = tmp_path / f"symbol={sym}"
            sym_dir.mkdir(parents=True)
            data.to_parquet(sym_dir / "data.parquet", index=False)

        return tmp_path

    def test_load_multiple_symbols(self, sample_panel):
        """Verify loading multiple symbols into panel."""
        from data_layer.curated.fundamentals import load_fundamentals_panel

        result = load_fundamentals_panel(
            symbols=["AAPL", "MSFT"],
            base_dir=str(sample_panel),
        )

        assert len(result) == 2
        assert set(result["symbol"]) == {"AAPL", "MSFT"}

    def test_load_missing_symbol_graceful(self, sample_panel):
        """Verify missing symbols don't cause errors."""
        from data_layer.curated.fundamentals import load_fundamentals_panel

        result = load_fundamentals_panel(
            symbols=["AAPL", "UNKNOWN"],
            base_dir=str(sample_panel),
        )

        # Should only have AAPL
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"

    def test_load_empty_directory(self, tmp_path):
        """Verify empty directory returns empty DataFrame."""
        from data_layer.curated.fundamentals import load_fundamentals_panel

        result = load_fundamentals_panel(
            symbols=["AAPL"],
            base_dir=str(tmp_path),
        )

        assert result.empty
