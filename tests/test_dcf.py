"""Tests for the DCF model using synthetic data (no network calls)."""

import pytest

from ainalyst.config import DCFAssumptions
from ainalyst.fundamentals import FundamentalSnapshot
from ainalyst.valuation.dcf import DCFModel


def _make_snap(**overrides) -> FundamentalSnapshot:
    """Build a minimal synthetic snapshot for testing."""
    defaults = dict(
        ticker="TST.AX",
        company_name="Test Corp",
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


class TestDCFModel:
    def test_basic_run(self) -> None:
        snap = _make_snap()
        result = DCFModel().run(snap)

        assert result.ticker == "TST.AX"
        assert result.enterprise_value > 0
        assert result.equity_value > 0
        assert result.intrinsic_per_share > 0
        assert len(result.projected_fcf) == 5
        assert len(result.pv_fcfs) == 5
        assert result.terminal_value > 0

    def test_sensitivity_shape(self) -> None:
        snap = _make_snap()
        result = DCFModel().run(snap)
        # Default: 5 WACC rows × 5 TGR columns
        assert result.sensitivity.shape == (5, 5)

    def test_custom_assumptions(self) -> None:
        snap = _make_snap()
        a = DCFAssumptions(
            projection_years=3,
            revenue_growth_rates=[0.10, 0.08, 0.06],
            wacc=0.12,
            terminal_growth_rate=0.03,
            target_operating_margin=0.20,
        )
        result = DCFModel(a).run(snap)
        assert len(result.projected_fcf) == 3
        assert result.assumptions["wacc"] == 0.12

    def test_negative_revenue_raises(self) -> None:
        snap = _make_snap(total_revenue=-100)
        with pytest.raises(ValueError, match="non-positive revenue"):
            DCFModel().run(snap)

    def test_margin_of_safety_computed(self) -> None:
        snap = _make_snap()
        result = DCFModel().run(snap)
        assert result.margin_of_safety_pct is not None

    def test_valuation_signal(self) -> None:
        snap = _make_snap()
        result = DCFModel().run(snap)
        assert result.valuation_signal in ("UNDERVALUED", "OVERVALUED", "FAIRLY VALUED")
