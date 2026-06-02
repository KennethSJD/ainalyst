"""Tests for the EV ↔ Equity bridge."""

import pytest

from ainalyst.valuation.bridge import ev_to_equity, equity_per_share, margin_of_safety


class TestEvToEquity:
    def test_basic_conversion(self) -> None:
        # EV=1B, Debt=300M, Cash=100M → Equity = 1B - 300M + 100M = 800M
        assert ev_to_equity(1_000_000_000, 300_000_000, 100_000_000) == 800_000_000

    def test_no_debt_no_cash(self) -> None:
        assert ev_to_equity(500, 0, 0) == 500

    def test_more_cash_than_debt(self) -> None:
        # EV=100, Debt=10, Cash=50 → 140
        assert ev_to_equity(100, 10, 50) == 140

    def test_negative_ev(self) -> None:
        # Should still work mathematically
        result = ev_to_equity(-100, 50, 20)
        assert result == -130  # -100 - 50 + 20


class TestEquityPerShare:
    def test_basic(self) -> None:
        assert equity_per_share(1_000_000, 100_000) == 10.0

    def test_zero_shares_raises(self) -> None:
        with pytest.raises(ValueError, match="diluted_shares must be > 0"):
            equity_per_share(1_000, 0)

    def test_negative_shares_raises(self) -> None:
        with pytest.raises(ValueError):
            equity_per_share(1_000, -5)


class TestMarginOfSafety:
    def test_undervalued(self) -> None:
        # Intrinsic $12, Price $10 → +20%
        mos = margin_of_safety(12.0, 10.0)
        assert abs(mos - 0.20) < 1e-9

    def test_overvalued(self) -> None:
        # Intrinsic $8, Price $10 → -20%
        mos = margin_of_safety(8.0, 10.0)
        assert abs(mos - (-0.20)) < 1e-9

    def test_fair_value(self) -> None:
        mos = margin_of_safety(10.0, 10.0)
        assert mos == 0.0

    def test_zero_price(self) -> None:
        assert margin_of_safety(10.0, 0.0) == 0.0
