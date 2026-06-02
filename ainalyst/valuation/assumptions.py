"""
Empirically-derived DCF assumptions from balance sheet + market data.

Computes:
  - WACC    via CAPM-based cost of equity + after-tax cost of debt
  - TGR     via sustainable growth rate (ROE × retention) capped at GDP ceiling
  - Margins from actual financials

Produces a computation log traceable back to source data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ainalyst.config import DCFAssumptions, normalise_ticker
from ainalyst.fundamentals import FundamentalSnapshot

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configurable market parameters
# ──────────────────────────────────────────────────────────────────────

# AU 10-year government bond yield (RBA / Bloomberg, ~May 2026)
DEFAULT_RISK_FREE_RATE: float = 0.043

# Long-term Australian equity risk premium (Damodaran / Aswath)
DEFAULT_EQUITY_RISK_PREMIUM: float = 0.055

# Minimum cost of equity — can't be below risk-free rate
MIN_COST_OF_EQUITY_PREMIUM: float = 0.02

# AU corporate tax rate
DEFAULT_TAX_RATE: float = 0.30

# Long-term GDP growth ceiling for TGR (AU + global weighted)
DEFAULT_GDP_CEILING: float = 0.035

# If beta is unavailable or NaN, use sector default
DEFAULT_BETA_HEALTHCARE: float = 0.65


@dataclass(slots=True)
class AssumptionSource:
    """Traceable provenance for a single assumption value."""

    key: str
    value: float
    formula: str
    inputs: dict[str, float]
    note: str = ""


@dataclass(slots=True)
class DerivedAssumptions:
    """Container holding all derived assumptions + full computation log."""

    wacc: float
    terminal_growth_rate: float
    target_operating_margin: float
    # Supporting metrics
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_weight: float
    debt_weight: float
    sustainable_growth_rate: float
    # Provenance
    sources: list[AssumptionSource] = field(default_factory=list)

    def to_dcf_assumptions(self) -> DCFAssumptions:
        """Convert to a DCFAssumptions object for the DCF engine."""
        return DCFAssumptions(
            wacc=round(self.wacc, 4),
            terminal_growth_rate=round(self.terminal_growth_rate, 4),
            target_operating_margin=round(self.target_operating_margin, 4),
        )

    def log_report(self) -> str:
        """Human-readable computation log."""
        def _fmt_val(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        lines = [
            "=" * 65,
            "DERIVED DCF ASSUMPTIONS — Computation Log",
            "=" * 65,
        ]
        for src in self.sources:
            lines.append(f"\n  [{src.key}] = {_fmt_val(src.value)}  ({src.value*100:.2f}%)" if isinstance(src.value, float) else f"\n  [{src.key}] = {src.value}")
            lines.append(f"    Formula:  {src.formula}")
            inputs_fmt = ", ".join(f"{k}={_fmt_val(v)}" for k, v in src.inputs.items())
            lines.append(f"    Inputs:   {inputs_fmt}")
            if src.note:
                lines.append(f"    Note:     {src.note}")
        lines.append(f"\n  ── Result ──")
        lines.append(f"  WACC                = {self.wacc:.4f}  ({self.wacc*100:.2f}%)")
        lines.append(f"  Terminal Growth     = {self.terminal_growth_rate:.4f}  ({self.terminal_growth_rate*100:.2f}%)")
        lines.append(f"  Target Op. Margin   = {self.target_operating_margin:.4f}  ({self.target_operating_margin*100:.2f}%)")
        lines.append(f"  ── Detail ──")
        lines.append(f"  Cost of Equity      = {self.cost_of_equity:.4f}  ({self.cost_of_equity*100:.2f}%)")
        lines.append(f"  Cost of Debt (pre)  = {self.cost_of_debt_pretax:.4f}  ({self.cost_of_debt_pretax*100:.2f}%)")
        lines.append(f"  Cost of Debt (post) = {self.cost_of_debt_aftertax:.4f}  ({self.cost_of_debt_aftertax*100:.2f}%)")
        lines.append(f"  Equity Weight       = {self.equity_weight:.4f}  ({self.equity_weight*100:.1f}%)")
        lines.append(f"  Debt Weight         = {self.debt_weight:.4f}  ({self.debt_weight*100:.1f}%)")
        lines.append(f"  Sustainable Growth  = {self.sustainable_growth_rate:.4f}  ({self.sustainable_growth_rate*100:.2f}%)")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for persistence."""
        return {
            "wacc": self.wacc,
            "terminal_growth_rate": self.terminal_growth_rate,
            "target_operating_margin": self.target_operating_margin,
            "cost_of_equity": self.cost_of_equity,
            "cost_of_debt_pretax": self.cost_of_debt_pretax,
            "cost_of_debt_aftertax": self.cost_of_debt_aftertax,
            "equity_weight": self.equity_weight,
            "debt_weight": self.debt_weight,
            "sustainable_growth_rate": self.sustainable_growth_rate,
            "computation_log": self.log_report(),
            "sources": [
                {
                    "key": s.key,
                    "value": s.value,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "note": s.note,
                }
                for s in self.sources
            ],
        }


def derive_assumptions(
    snap: FundamentalSnapshot,
    info: dict[str, Any] | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    tax_rate: float = DEFAULT_TAX_RATE,
    gdp_ceiling: float = DEFAULT_GDP_CEILING,
) -> DerivedAssumptions:
    """Compute empirically-grounded DCF assumptions from a single company snapshot.

    Parameters
    ----------
    snap : FundamentalSnapshot
        The company's current financial snapshot.
    info : dict | None
        yfinance .info dict for beta, payoutRatio, returnOnEquity, etc.
        If None, derives what it can from snap alone.
    risk_free_rate : float
        Government bond yield (default 4.3% for AU 10yr).
    equity_risk_premium : float
        Long-term equity risk premium (default 5.5% for AU).
    tax_rate : float
        Corporate tax rate (default 30%).
    gdp_ceiling : float
        Maximum terminal growth rate (default 3.5% long-term GDP).

    Returns
    -------
    DerivedAssumptions
        Container with all derived values + traceable computation log.
    """
    sources: list[AssumptionSource] = []
    info = info or {}

    # ── 1. Cost of Debt ──────────────────────────────────────────
    if snap.interest_expense > 0 and snap.total_debt > 0:
        cost_debt_pre = snap.interest_expense / snap.total_debt
    else:
        # Fallback: use 4% as rough corporate borrowing rate
        cost_debt_pre = 0.04

    cost_debt_post = cost_debt_pre * (1 - tax_rate)
    sources.append(AssumptionSource(
        key="cost_of_debt_pretax",
        value=cost_debt_pre,
        formula="interest_expense / total_debt",
        inputs={
            "interest_expense": snap.interest_expense / 1e6,
            "total_debt": snap.total_debt / 1e6,
        },
        note=f"Interest expense ${snap.interest_expense/1e6:.0f}M / Debt ${snap.total_debt/1e6:.0f}M"
        if snap.interest_expense > 0 else "Fallback: assumed 4% corporate borrowing rate",
    ))
    sources.append(AssumptionSource(
        key="cost_of_debt_aftertax",
        value=cost_debt_post,
        formula="cost_of_debt_pre × (1 − tax_rate)",
        inputs={"cost_of_debt_pre": cost_debt_pre, "tax_rate": tax_rate},
    ))

    # ── 2. Cost of Equity (CAPM) ─────────────────────────────────
    beta = info.get("beta")
    beta_note = ""
    if beta is None or not isinstance(beta, (int, float)):
        sector = snap.sector.lower() if snap.sector else ""
        if "health" in sector or "bio" in sector:
            beta = DEFAULT_BETA_HEALTHCARE
        else:
            beta = 1.0
        beta_note = f"Beta unavailable from yfinance; using sector default {beta}"
    elif beta <= 0:
        # Negative beta — stock moves opposite to market (common in energy/gold)
        # Floor at 0.3 to avoid absurdly low cost of equity
        beta_note = f"Negative beta ({beta}) from yfinance — likely noise or short window; floored at 0.3 for CAPM"
        beta = max(beta, 0.3)

    sources.append(AssumptionSource(
        key="beta",
        value=beta,
        formula="yfinance .info['beta']" if not beta_note else "yfinance .info['beta'] → adjusted",
        inputs={},
        note=beta_note or "5-year monthly beta vs ASX200 from Yahoo Finance",
    ))

    cost_equity = risk_free_rate + beta * equity_risk_premium
    # Floor: cost of equity must be at least risk-free + minimum premium
    min_coe = risk_free_rate + MIN_COST_OF_EQUITY_PREMIUM
    cost_equity_raw = cost_equity
    if cost_equity < min_coe:
        cost_equity = min_coe
    sources.append(AssumptionSource(
        key="cost_of_equity",
        value=cost_equity,
        formula="Rf + β × ERP",
        inputs={
            "risk_free_rate": risk_free_rate,
            "beta": beta,
            "equity_risk_premium": equity_risk_premium,
        },
        note=f"Rf={risk_free_rate:.1%} (AU 10yr), ERP={equity_risk_premium:.1%} (Damodaran)"
        + (f" — floored from {cost_equity_raw:.1%} to {min_coe:.1%}" if cost_equity_raw < min_coe else ""),
    ))

    # ── 3. Capital Structure Weights ─────────────────────────────
    market_cap = info.get("marketCap") or snap.market_cap or 0
    total_debt = snap.total_debt

    if market_cap > 0 and total_debt > 0:
        equity_weight = market_cap / (market_cap + total_debt)
        debt_weight = total_debt / (market_cap + total_debt)
    elif market_cap > 0:
        equity_weight, debt_weight = 1.0, 0.0
    else:
        equity_weight, debt_weight = 0.7, 0.3  # rough fallback

    sources.append(AssumptionSource(
        key="equity_weight",
        value=equity_weight,
        formula="market_cap / (market_cap + total_debt)",
        inputs={
            "market_cap": market_cap / 1e9,
            "total_debt": total_debt / 1e9,
        },
        note=f"MCap ${market_cap/1e9:.1f}B, Debt ${total_debt/1e9:.1f}B",
    ))
    sources.append(AssumptionSource(
        key="debt_weight",
        value=debt_weight,
        formula="total_debt / (market_cap + total_debt)",
        inputs={
            "market_cap": market_cap / 1e9,
            "total_debt": total_debt / 1e9,
        },
    ))

    # ── 4. WACC ──────────────────────────────────────────────────
    wacc = equity_weight * cost_equity + debt_weight * cost_debt_post
    sources.append(AssumptionSource(
        key="wacc",
        value=wacc,
        formula="EquityWeight × CostEquity + DebtWeight × CostDebtAfterTax",
        inputs={
            "equity_weight": equity_weight,
            "cost_of_equity": cost_equity,
            "debt_weight": debt_weight,
            "cost_of_debt_aftertax": cost_debt_post,
        },
    ))

    # ── 5. Sustainable Growth Rate ───────────────────────────────
    roe = info.get("returnOnEquity") or snap.roe or 0.10
    payout = info.get("payoutRatio") or 0.40
    if not isinstance(payout, (int, float)) or payout < 0:
        payout = 0.40
    retention = 1.0 - min(payout, 1.0)
    sgr = roe * retention

    sources.append(AssumptionSource(
        key="return_on_equity",
        value=roe,
        formula="yfinance .info['returnOnEquity'] or net_income/total_equity",
        inputs={"net_income": snap.net_income / 1e6, "total_equity": snap.total_equity / 1e6}
        if snap.net_income > 0 else {},
    ))
    sources.append(AssumptionSource(
        key="payout_ratio",
        value=payout,
        formula="yfinance .info['payoutRatio']",
        inputs={},
    ))
    sources.append(AssumptionSource(
        key="sustainable_growth_rate",
        value=sgr,
        formula="ROE × (1 − payout_ratio)",
        inputs={"roe": roe, "payout_ratio": payout},
        note="Gordon Growth sustainable rate: how fast equity can grow from retained earnings",
    ))

    # ── 6. Terminal Growth Rate (capped) ─────────────────────────
    tgr = min(sgr, gdp_ceiling)
    if tgr < 0:
        tgr = 0.02  # floor
    sources.append(AssumptionSource(
        key="terminal_growth_rate",
        value=tgr,
        formula="min(sustainable_growth_rate, gdp_ceiling)",
        inputs={
            "sustainable_growth_rate": sgr,
            "gdp_ceiling": gdp_ceiling,
        },
        note=f"SGR capped at long-term GDP ceiling {gdp_ceiling:.1%}"
        if sgr > gdp_ceiling else "SGR below GDP ceiling — no cap applied",
    ))

    # ── 7. Target Operating Margin ───────────────────────────────
    # Use actual operating margin from the snapshot (closest to truth)
    op_margin = info.get("operatingMargins") or snap.operating_margin or 0.15
    if op_margin <= 0:
        op_margin = 0.15
    sources.append(AssumptionSource(
        key="target_operating_margin",
        value=op_margin,
        formula="yfinance operatingMargins or EBIT/revenue",
        inputs={
            "operating_income": snap.operating_income / 1e6,
            "total_revenue": snap.total_revenue / 1e6,
        },
        note=f"Actual trailing operating margin = {op_margin:.1%}",
    ))

    return DerivedAssumptions(
        wacc=wacc,
        terminal_growth_rate=tgr,
        target_operating_margin=op_margin,
        cost_of_equity=cost_equity,
        cost_of_debt_pretax=cost_debt_pre,
        cost_of_debt_aftertax=cost_debt_post,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        sustainable_growth_rate=sgr,
        sources=sources,
    )
