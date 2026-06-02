"""Discounted Cash Flow (DCF) model with sensitivity matrix output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.config import DCFAssumptions
from ainalyst.fundamentals import FundamentalSnapshot
from ainalyst.valuation.bridge import ev_to_equity, equity_per_share, margin_of_safety

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DCFResult:
    """Immutable output of a DCF run."""

    ticker: str
    company_name: str

    # Core outputs
    enterprise_value: float
    equity_value: float
    intrinsic_per_share: float
    current_price: float | None
    margin_of_safety_pct: float | None

    # Projection detail
    projected_fcf: list[float]
    discount_factors: list[float]
    pv_fcfs: list[float]
    terminal_value: float
    pv_terminal: float

    # Sensitivity matrix: rows=WACC, cols=TGR, cells=intrinsic/share
    sensitivity: pd.DataFrame

    # Assumptions used
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def valuation_signal(self) -> str:
        if self.margin_of_safety_pct is None:
            return "N/A"
        if self.margin_of_safety_pct > 0.15:
            return "UNDERVALUED"
        elif self.margin_of_safety_pct < -0.15:
            return "OVERVALUED"
        return "FAIRLY VALUED"


# ──────────────────────────────────────────────────────────────────────
# DCF engine
# ──────────────────────────────────────────────────────────────────────

class DCFModel:
    """Build an unlevered FCFF-based DCF for an ASX company."""

    def __init__(self, assumptions: DCFAssumptions | None = None) -> None:
        self.a = assumptions or DCFAssumptions()

    # ── public entry point ──────────────────────────────────────────

    def run(self, snap: FundamentalSnapshot) -> DCFResult:
        """Execute the DCF and return a :class:`DCFResult`."""
        base_revenue = snap.total_revenue
        if base_revenue <= 0:
            raise ValueError(
                f"Cannot run DCF: {snap.ticker} has non-positive revenue ({base_revenue})"
            )
        if snap.shares_outstanding <= 0:
            raise ValueError(
                f"Cannot run DCF: {snap.ticker} has zero shares outstanding"
            )

        # ── 1. Project Free Cash Flows ──────────────────────────────
        growth_rates = self._pad_growth_rates()
        projected_fcf: list[float] = []
        revenue = base_revenue

        for yr in range(self.a.projection_years):
            revenue *= 1 + growth_rates[yr]
            ebit = revenue * self.a.target_operating_margin
            nopat = ebit * (1 - self.a.tax_rate)
            capex = revenue * self.a.capex_pct_revenue
            delta_nwc = revenue * self.a.delta_nwc_pct_revenue
            # Use actual D&A from financials where available, else ratio to revenue
            if snap.depreciation > 0:
                d_and_a = snap.depreciation
            elif snap.total_revenue > 0:
                d_and_a = revenue * (snap.depreciation / snap.total_revenue) if snap.depreciation > 0 else capex * 0.8
            else:
                d_and_a = capex * 0.8
            fcf = nopat + d_and_a - capex - delta_nwc
            projected_fcf.append(fcf)

        # ── 2. Discount factors & PV of projection period ───────────
        discount_factors = [1 / (1 + self.a.wacc) ** (yr + 1) for yr in range(self.a.projection_years)]
        pv_fcfs = [f * d for f, d in zip(projected_fcf, discount_factors)]

        # ── 3. Terminal value (Gordon Growth) ───────────────────────
        terminal_fcf = projected_fcf[-1] * (1 + self.a.terminal_growth_rate)
        terminal_value = terminal_fcf / (self.a.wacc - self.a.terminal_growth_rate)
        pv_terminal = terminal_value * discount_factors[-1]

        # ── 4. Enterprise → Equity → Per Share ─────────────────────
        ev = sum(pv_fcfs) + pv_terminal
        eq = ev_to_equity(ev, snap.total_debt, snap.cash_and_equivalents)
        shares = snap.shares_outstanding if snap.shares_outstanding > 0 else 1.0
        ips = equity_per_share(eq, shares)

        mos: float | None = None
        if snap.current_price and snap.current_price > 0:
            mos = margin_of_safety(ips, snap.current_price)

        # ── 5. Sensitivity matrix ──────────────────────────────────
        sensitivity = self._build_sensitivity(
            projected_fcf=projected_fcf,
            total_debt=snap.total_debt,
            cash=snap.cash_and_equivalents,
            shares=shares,
        )

        return DCFResult(
            ticker=snap.ticker,
            company_name=snap.company_name,
            enterprise_value=ev,
            equity_value=eq,
            intrinsic_per_share=ips,
            current_price=snap.current_price,
            margin_of_safety_pct=mos,
            projected_fcf=projected_fcf,
            discount_factors=discount_factors,
            pv_fcfs=pv_fcfs,
            terminal_value=terminal_value,
            pv_terminal=pv_terminal,
            sensitivity=sensitivity,
            assumptions=self._assumptions_dict(),
        )

    # ── internals ───────────────────────────────────────────────────

    def _pad_growth_rates(self) -> list[float]:
        """Ensure growth rates list matches projection_years length."""
        rates = list(self.a.revenue_growth_rates)
        while len(rates) < self.a.projection_years:
            rates.append(rates[-1] if rates else 0.04)
        return rates[: self.a.projection_years]

    def _build_sensitivity(
        self,
        projected_fcf: list[float],
        total_debt: float,
        cash: float,
        shares: float,
    ) -> pd.DataFrame:
        """WACC × Terminal-Growth-Rate sensitivity matrix."""
        rows: list[dict[str, float]] = []
        for wacc in self.a.wacc_range:
            row: dict[str, float] = {}
            for tgr in self.a.tgr_range:
                if wacc <= tgr:
                    row[f"{tgr:.1%}"] = np.nan
                    continue
                dfs = [1 / (1 + wacc) ** (yr + 1) for yr in range(self.a.projection_years)]
                pv = sum(f * d for f, d in zip(projected_fcf, dfs))
                tv_fcf = projected_fcf[-1] * (1 + tgr)
                tv = tv_fcf / (wacc - tgr)
                pv_tv = tv * dfs[-1]
                ev = pv + pv_tv
                eq = ev_to_equity(ev, total_debt, cash)
                ips = eq / shares if shares > 0 else 0.0
                row[f"{tgr:.1%}"] = round(ips, 2)
            rows.append(row)

        idx = [f"{w:.1%}" for w in self.a.wacc_range]
        return pd.DataFrame(rows, index=idx)

    def _assumptions_dict(self) -> dict[str, Any]:
        return {
            "projection_years": self.a.projection_years,
            "revenue_growth_rates": self.a.revenue_growth_rates,
            "target_operating_margin": self.a.target_operating_margin,
            "tax_rate": self.a.tax_rate,
            "capex_pct_revenue": self.a.capex_pct_revenue,
            "delta_nwc_pct_revenue": self.a.delta_nwc_pct_revenue,
            "wacc": self.a.wacc,
            "terminal_growth_rate": self.a.terminal_growth_rate,
        }
