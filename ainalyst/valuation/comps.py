"""Trading Comparables (Comps) model — peer-multiple implied valuation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.acquisition import ASXTicker, fetch_tickers
from ainalyst.config import normalise_ticker, peers_for_sector
from ainalyst.fundamentals import FundamentalSnapshot, build_snapshot, build_snapshots
from ainalyst.valuation.bridge import ev_to_equity, equity_per_share, margin_of_safety

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class CompsResult:
    """Output of a comps-based valuation."""

    ticker: str
    company_name: str

    # Peer data
    peer_table: pd.DataFrame          # rows = peers, cols = multiples + financials
    peer_count: int

    # Medians / means of peer multiples
    median_pe: float | None
    median_ev_ebitda: float | None
    median_ev_sales: float | None
    mean_pe: float | None
    mean_ev_ebitda: float | None
    mean_ev_sales: float | None

    # Implied valuation for the *target* using each multiple
    implied_pe_value: float | None
    implied_ev_ebitda_value: float | None
    implied_ev_sales_value: float | None

    # Best-estimate composite (average of available implied values)
    composite_value: float | None
    current_price: float | None
    margin_of_safety_pct: float | None

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
# Comps engine
# ──────────────────────────────────────────────────────────────────────

class CompsModel:
    """Compute implied equity value of a target from peer multiples."""

    def run(
        self,
        target: FundamentalSnapshot,
        peer_tickers: list[str] | None = None,
        market_cap_filter: float | None = None,
    ) -> CompsResult:
        """Run the comps analysis.

        Parameters
        ----------
        target : FundamentalSnapshot
            The company being valued.
        peer_tickers : list[str] | None
            Explicit peer symbols.  Falls back to sector defaults.
        market_cap_filter : float | None
            If set, only keep peers within ±50%% of the target's market cap.
            Value represents the tolerance (default 0.5 = ±50%%).
        """
        # Resolve peers
        if peer_tickers is None:
            try:
                peer_tickers = peers_for_sector(target.sector)
            except KeyError:
                log.warning("No default peers for sector '%s'; using empty list", target.sector)
                peer_tickers = []

        # Remove the target itself from the peer list
        target_norm = normalise_ticker(target.ticker)
        peer_tickers = [p for p in peer_tickers if normalise_ticker(p) != target_norm]

        # Build peer snapshots
        peer_objs = fetch_tickers(peer_tickers)
        peer_snaps = build_snapshots(peer_objs)

        # Apply market cap filtering
        if market_cap_filter is not None and target.market_cap:
            lo = target.market_cap * (1 - market_cap_filter)
            hi = target.market_cap * (1 + market_cap_filter)
            before = len(peer_snaps)
            peer_snaps = [
                s for s in peer_snaps
                if s.market_cap and lo <= s.market_cap <= hi
            ]
            log.info(
                "Market-cap filter (±%.0f%%): kept %d/%d peers",
                market_cap_filter * 100,
                len(peer_snaps),
                before,
            )

        # Assemble peer table
        rows: list[dict[str, Any]] = []
        for s in peer_snaps:
            rows.append({
                "Ticker": s.ticker,
                "Company": s.company_name,
                "Revenue": s.total_revenue,
                "EBITDA": s.ebitda,
                "Net Income": s.net_income,
                "Market Cap": s.market_cap,
                "EV": s.enterprise_value,
                "P/E": s.pe_ratio,
                "EV/EBITDA": s.ev_ebitda,
                "EV/Sales": s.ev_sales,
            })
        peer_table = pd.DataFrame(rows)

        # Compute medians & means (ignoring NaN / None)
        med_pe = _safe_median(peer_table, "P/E")
        med_ev_ebitda = _safe_median(peer_table, "EV/EBITDA")
        med_ev_sales = _safe_median(peer_table, "EV/Sales")
        mean_pe = _safe_mean(peer_table, "P/E")
        mean_ev_ebitda = _safe_mean(peer_table, "EV/EBITDA")
        mean_ev_sales = _safe_mean(peer_table, "EV/Sales")

        shares = target.shares_outstanding if target.shares_outstanding > 0 else 1.0

        # Implied values per share for target
        implied_pe = _implied_pe(med_pe, target, shares)
        implied_ev_ebitda = _implied_ev_multiple(med_ev_ebitda, target.ebitda, target, shares)
        implied_ev_sales = _implied_ev_multiple(med_ev_sales, target.total_revenue, target, shares)

        # Composite
        implied_values = [v for v in (implied_pe, implied_ev_ebitda, implied_ev_sales) if v is not None]
        composite = float(np.mean(implied_values)) if implied_values else None

        mos: float | None = None
        if composite is not None and target.current_price and target.current_price > 0:
            mos = margin_of_safety(composite, target.current_price)

        return CompsResult(
            ticker=target.ticker,
            company_name=target.company_name,
            peer_table=peer_table,
            peer_count=len(peer_snaps),
            median_pe=med_pe,
            median_ev_ebitda=med_ev_ebitda,
            median_ev_sales=med_ev_sales,
            mean_pe=mean_pe,
            mean_ev_ebitda=mean_ev_ebitda,
            mean_ev_sales=mean_ev_sales,
            implied_pe_value=implied_pe,
            implied_ev_ebitda_value=implied_ev_ebitda,
            implied_ev_sales_value=implied_ev_sales,
            composite_value=composite,
            current_price=target.current_price,
            margin_of_safety_pct=mos,
            assumptions={"peer_tickers": peer_tickers},
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_median(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    # Filter out extreme outliers (> 3σ from median)
    if s.empty:
        return None
    med = float(s.median())
    std = float(s.std()) if len(s) > 1 else 0.0
    if std > 0:
        s = s[(s - med).abs() <= 3 * std]
    return float(s.median()) if not s.empty else None


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    med = float(s.median())
    std = float(s.std()) if len(s) > 1 else 0.0
    if std > 0:
        s = s[(s - med).abs() <= 3 * std]
    return float(s.mean()) if not s.empty else None


def _implied_pe(
    median_pe: float | None,
    target: FundamentalSnapshot,
    shares: float,
) -> float | None:
    """P/E implied value: Median P/E × Target EPS."""
    if median_pe is None or target.net_income <= 0 or shares <= 0:
        return None
    eps = target.net_income / shares
    return median_pe * eps


def _implied_ev_multiple(
    median_multiple: float | None,
    target_metric: float,
    target: FundamentalSnapshot,
    shares: float,
) -> float | None:
    """EV-multiple implied per-share value via the bridge."""
    if median_multiple is None or target_metric <= 0 or shares <= 0:
        return None
    implied_ev = median_multiple * target_metric
    implied_eq = ev_to_equity(implied_ev, target.total_debt, target.cash_and_equivalents)
    return equity_per_share(implied_eq, shares)
