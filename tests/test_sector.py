"""Tests for the Sector Analyzer module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ainalyst.sector import SectorAnalyzer, SectorSummary


class TestSectorAnalyzer:
    @patch("ainalyst.sector.fetch_tickers")
    @patch("ainalyst.sector.build_snapshots")
    def test_analyse_with_peers(self, mock_build, mock_fetch):
        from ainalyst.fundamentals import FundamentalSnapshot

        snaps = [
            FundamentalSnapshot(
                ticker="A.AX", company_name="A", sector="Materials", currency="AUD",
                total_revenue=1000, cost_of_revenue=400, gross_profit=600,
                operating_income=200, ebitda=250, net_income=150,
                interest_expense=10, tax_provision=50,
                total_assets=5000, total_liabilities=2000, total_debt=1000,
                cash_and_equivalents=500, total_equity=3000,
                operating_cashflow=300, capital_expenditure=80, free_cashflow=220,
                depreciation=50, shares_outstanding=100, current_price=10.0,
                market_cap=1000,
            ),
            FundamentalSnapshot(
                ticker="B.AX", company_name="B", sector="Materials", currency="AUD",
                total_revenue=2000, cost_of_revenue=800, gross_profit=1200,
                operating_income=400, ebitda=500, net_income=300,
                interest_expense=20, tax_provision=100,
                total_assets=10000, total_liabilities=4000, total_debt=2000,
                cash_and_equivalents=1000, total_equity=6000,
                operating_cashflow=600, capital_expenditure=160, free_cashflow=440,
                depreciation=100, shares_outstanding=200, current_price=20.0,
                market_cap=4000,
            ),
        ]
        mock_build.return_value = snaps

        result = SectorAnalyzer().analyse_sector("Materials",
                                                  tickers=["A", "B"])
        assert result.sector == "Materials"
        assert result.count == 2
        assert result.median_pe is not None
        assert result.median_gross_margin is not None

    def test_analyse_all_sectors(self):
        analyzer = SectorAnalyzer()
        with patch.object(analyzer, "analyse_sector") as mock_analyse:
            mock_analyse.return_value = SectorSummary(
                sector="Test", companies=pd.DataFrame(), count=0,
                median_pe=None, mean_pe=None,
                median_ev_ebitda=None, mean_ev_ebitda=None,
                median_ev_sales=None, mean_ev_sales=None,
                median_gross_margin=None, median_operating_margin=None,
                median_net_margin=None, median_roe=None,
            )
            results = analyzer.analyse_all_sectors()
            assert len(results) > 0

    @patch("ainalyst.sector.load_directory")
    @patch("ainalyst.sector.fetch_tickers")
    @patch("ainalyst.sector.build_snapshots")
    def test_analyse_with_directory_fallback(self, mock_build, mock_fetch, mock_load):
        from ainalyst.fundamentals import FundamentalSnapshot
        mock_load.return_value = pd.DataFrame({
            "ticker": ["CSTM.AX"],
            "name": ["Custom"],
            "industry": ["Custom Sector"],
            "listing_date": [pd.Timestamp("2020-01-01")],
            "market_cap": [500_000_000],
        })
        mock_build.return_value = [
            FundamentalSnapshot(
                ticker="CSTM.AX", company_name="Custom", sector="Custom",
                currency="AUD",
                total_revenue=500, cost_of_revenue=200, gross_profit=300,
                operating_income=100, ebitda=125, net_income=75,
                interest_expense=5, tax_provision=25,
                total_assets=2500, total_liabilities=1000, total_debt=500,
                cash_and_equivalents=250, total_equity=1500,
                operating_cashflow=150, capital_expenditure=40, free_cashflow=110,
                depreciation=25, shares_outstanding=50, current_price=10.0,
                market_cap=500,
            ),
        ]

        result = SectorAnalyzer().analyse_sector("Custom Sector",
                                                  use_directory=True)
        assert result.count == 1

    @patch("ainalyst.sector.load_directory")
    def test_analyse_nonexistent_sector_directory(self, mock_load):
        mock_load.return_value = pd.DataFrame({
            "ticker": pd.Series([], dtype="str"),
            "name": pd.Series([], dtype="str"),
            "industry": pd.Series([], dtype="str"),
            "listing_date": pd.Series([], dtype="datetime64[ns]"),
            "market_cap": pd.Series([], dtype="float64"),
        })
        result = SectorAnalyzer().analyse_sector("NothingHere",
                                                  use_directory=True)
        assert result.count == 0

    def test_sector_comparison_table(self):
        analyzer = SectorAnalyzer()
        summaries = [
            SectorSummary(
                sector="A", companies=pd.DataFrame(), count=1,
                median_pe=10.0, mean_pe=10.0,
                median_ev_ebitda=5.0, mean_ev_ebitda=5.0,
                median_ev_sales=1.0, mean_ev_sales=1.0,
                median_gross_margin=0.4, median_operating_margin=0.15,
                median_net_margin=0.10, median_roe=0.12,
            ),
        ]
        df = analyzer.sector_comparison_table(summaries)
        assert "A" in df.index
        assert df.loc["A", "count"] == 1
