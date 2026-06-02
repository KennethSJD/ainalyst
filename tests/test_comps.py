"""Tests for the Trading Comparables model."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ainalyst.fundamentals import FundamentalSnapshot
from ainalyst.valuation.comps import CompsModel, CompsResult


def _make_snap(**overrides) -> FundamentalSnapshot:
    defaults = dict(
        ticker="TGT.AX",
        company_name="Target Corp",
        sector="Materials",
        currency="AUD",
        total_revenue=1_000_000_000,
        cost_of_revenue=600_000_000,
        gross_profit=400_000_000,
        operating_income=150_000_000,
        ebitda=200_000_000,
        net_income=100_000_000,
        interest_expense=20_000_000,
        tax_provision=50_000_000,
        total_assets=2_000_000_000,
        total_liabilities=800_000_000,
        total_debt=500_000_000,
        cash_and_equivalents=200_000_000,
        total_equity=1_200_000_000,
        operating_cashflow=180_000_000,
        capital_expenditure=50_000_000,
        free_cashflow=130_000_000,
        depreciation=50_000_000,
        shares_outstanding=500_000_000,
        current_price=5.00,
        market_cap=2_500_000_000,
    )
    defaults.update(overrides)
    return FundamentalSnapshot(**defaults)


def _make_peer(ticker: str, market_cap: float, pe: float, ev_ebitda: float,
               ev_sales: float, **kw) -> FundamentalSnapshot:
    return _make_snap(
        ticker=ticker,
        company_name=f"{ticker} Ltd",
        market_cap=market_cap,
        current_price=market_cap / 500_000_000,
        total_revenue=kw.get("total_revenue", 800_000_000),
        ebitda=kw.get("ebitda", 160_000_000),
        net_income=kw.get("net_income", 80_000_000),
        total_debt=kw.get("total_debt", 300_000_000),
        cash_and_equivalents=kw.get("cash_and_equivalents", 100_000_000),
    )


class TestCompsModel:
    @patch("ainalyst.valuation.comps.fetch_tickers")
    @patch("ainalyst.valuation.comps.build_snapshots")
    def test_basic_run_with_peers(self, mock_build, mock_fetch):
        target = _make_snap()
        peers = [
            _make_peer("PEER.AX", 2_000_000_000, 12.0, 8.0, 2.0),
            _make_peer("BUDDY.AX", 3_000_000_000, 15.0, 10.0, 2.5),
        ]
        mock_build.return_value = peers

        result = CompsModel().run(target, peer_tickers=["PEER", "BUDDY"])
        assert result.peer_count == 2
        assert result.median_pe is not None
        assert result.composite_value is not None
        assert result.composite_value > 0

    @patch("ainalyst.valuation.comps.fetch_tickers")
    @patch("ainalyst.valuation.comps.build_snapshots")
    def test_market_cap_filter_includes_similar(self, mock_build, mock_fetch):
        target = _make_snap(market_cap=2_500_000_000)
        peers = [
            _make_peer("SIMILAR.AX", 3_000_000_000, 12.0, 8.0, 2.0),
            _make_peer("TINY.AX", 50_000_000, 5.0, 3.0, 0.5),
        ]
        mock_build.return_value = peers

        result = CompsModel().run(target, peer_tickers=["SIMILAR", "TINY"],
                                  market_cap_filter=0.5)
        # TINY (50M) is outside ±50% of 2.5B (1.25B-3.75B)
        assert result.peer_count == 1

    @patch("ainalyst.valuation.comps.fetch_tickers")
    @patch("ainalyst.valuation.comps.build_snapshots")
    def test_empty_peers_still_returns_result(self, mock_build, mock_fetch):
        target = _make_snap()
        mock_build.return_value = []

        result = CompsModel().run(target, peer_tickers=[])
        assert result.peer_count == 0
        assert result.composite_value is None

    def test_comps_result_signal(self):
        result = CompsResult(
            ticker="T.AX",
            company_name="T",
            peer_table=pd.DataFrame(),
            peer_count=3,
            median_pe=10.0,
            median_ev_ebitda=7.0,
            median_ev_sales=1.5,
            mean_pe=11.0,
            mean_ev_ebitda=7.5,
            mean_ev_sales=1.6,
            implied_pe_value=15.0,
            implied_ev_ebitda_value=14.0,
            implied_ev_sales_value=13.0,
            composite_value=14.0,
            current_price=10.0,
            margin_of_safety_pct=0.40,
        )
        assert result.valuation_signal == "UNDERVALUED"
