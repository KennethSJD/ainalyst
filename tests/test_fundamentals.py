"""Tests for fundamentals parsing helpers."""

import numpy as np
import pandas as pd
import pytest

from ainalyst.fundamentals import _get, _get_any, FundamentalSnapshot


class TestGetHelpers:
    def test_get_existing_label(self) -> None:
        df = pd.DataFrame({"2023": [100, 200]}, index=["Revenue", "EBITDA"])
        assert _get(df, "Revenue", 0) == 100.0

    def test_get_missing_label(self) -> None:
        df = pd.DataFrame({"2023": [100]}, index=["Revenue"])
        assert _get(df, "EBITDA", 0) == 0.0

    def test_get_nan_returns_zero(self) -> None:
        df = pd.DataFrame({"2023": [np.nan]}, index=["Revenue"])
        assert _get(df, "Revenue", 0) == 0.0

    def test_get_empty_df(self) -> None:
        assert _get(pd.DataFrame(), "Revenue", 0) == 0.0

    def test_get_any_first_hit(self) -> None:
        df = pd.DataFrame({"2023": [0, 42]}, index=["TotalRevenue", "Revenue"])
        assert _get_any(df, ["TotalRevenue", "Revenue"]) == 42.0  # TotalRevenue is 0 → skip

    def test_get_any_all_miss(self) -> None:
        df = pd.DataFrame({"2023": [10]}, index=["Other"])
        assert _get_any(df, ["Revenue", "TotalRevenue"]) == 0.0


class TestFundamentalSnapshot:
    def _make(self, **kw) -> FundamentalSnapshot:
        defaults = dict(
            ticker="X.AX", company_name="X", sector="Tech", currency="AUD",
            total_revenue=1000, cost_of_revenue=400, gross_profit=600,
            operating_income=200, ebitda=250, net_income=150,
            interest_expense=10, tax_provision=50,
            total_assets=5000, total_liabilities=2000, total_debt=1000,
            cash_and_equivalents=500, total_equity=3000,
            operating_cashflow=300, capital_expenditure=80, free_cashflow=220,
            depreciation=50, shares_outstanding=100, current_price=10.0,
            market_cap=1000,
        )
        defaults.update(kw)
        return FundamentalSnapshot(**defaults)

    def test_enterprise_value(self) -> None:
        s = self._make(market_cap=1000, total_debt=200, cash_and_equivalents=50)
        assert s.enterprise_value == 1150  # 1000 + 200 - 50

    def test_pe_ratio(self) -> None:
        s = self._make(net_income=100, shares_outstanding=50, current_price=20)
        # EPS = 2, P/E = 10
        assert s.pe_ratio == pytest.approx(10.0)

    def test_pe_negative_income(self) -> None:
        s = self._make(net_income=-100)
        assert s.pe_ratio is None

    def test_ev_ebitda(self) -> None:
        s = self._make(market_cap=1000, total_debt=200, cash_and_equivalents=50, ebitda=100)
        # EV = 1150, EV/EBITDA = 11.5
        assert s.ev_ebitda == pytest.approx(11.5)

    def test_ev_ebitda_negative(self) -> None:
        s = self._make(ebitda=-10)
        assert s.ev_ebitda is None

    def test_margins(self) -> None:
        s = self._make(total_revenue=1000, gross_profit=600, operating_income=200, net_income=100)
        assert s.gross_margin == pytest.approx(0.6)
        assert s.operating_margin == pytest.approx(0.2)
        assert s.net_margin == pytest.approx(0.1)

    def test_to_dict_has_derived_keys(self) -> None:
        d = self._make().to_dict()
        for key in ("enterprise_value", "pe_ratio", "ev_ebitda", "ev_sales",
                     "gross_margin", "operating_margin", "net_margin", "roe"):
            assert key in d
