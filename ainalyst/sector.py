"""Aggregate Sector Analyzer — compute sector-level summary statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.acquisition import fetch_tickers
from ainalyst.config import peers_for_sector, SECTOR_PEERS
from ainalyst.fundamentals import FundamentalSnapshot, build_snapshots

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SectorSummary:
    """Aggregated metrics for a single sector."""

    sector: str
    companies: pd.DataFrame          # per-company detail
    count: int

    # Aggregate multiples
    median_pe: float | None
    mean_pe: float | None
    median_ev_ebitda: float | None
    mean_ev_ebitda: float | None
    median_ev_sales: float | None
    mean_ev_sales: float | None

    # Aggregate margins
    median_gross_margin: float | None
    median_operating_margin: float | None
    median_net_margin: float | None
    median_roe: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "count": self.count,
            "median_pe": self.median_pe,
            "mean_pe": self.mean_pe,
            "median_ev_ebitda": self.median_ev_ebitda,
            "mean_ev_ebitda": self.mean_ev_ebitda,
            "median_ev_sales": self.median_ev_sales,
            "mean_ev_sales": self.mean_ev_sales,
            "median_gross_margin": self.median_gross_margin,
            "median_operating_margin": self.median_operating_margin,
            "median_net_margin": self.median_net_margin,
            "median_roe": self.median_roe,
        }


# ──────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────

class SectorAnalyzer:
    """Compute aggregate multiples and margins for ASX sectors."""

    def analyse_sector(
        self,
        sector: str,
        tickers: list[str] | None = None,
    ) -> SectorSummary:
        """Analyse a single sector.

        Parameters
        ----------
        sector : str
            GICS sector name (must match keys in ``SECTOR_PEERS``).
        tickers : list[str] | None
            Override list of tickers; defaults to ``SECTOR_PEERS[sector]``.
        """
        if tickers is None:
            tickers = peers_for_sector(sector)

        objs = fetch_tickers(tickers)
        snaps = build_snapshots(objs)
        if not snaps:
            log.warning("No valid snapshots for sector '%s'", sector)
            return _empty_summary(sector)

        df = pd.DataFrame([s.to_dict() for s in snaps])

        return SectorSummary(
            sector=sector,
            companies=df,
            count=len(snaps),
            median_pe=_med(df, "pe_ratio"),
            mean_pe=_mn(df, "pe_ratio"),
            median_ev_ebitda=_med(df, "ev_ebitda"),
            mean_ev_ebitda=_mn(df, "ev_ebitda"),
            median_ev_sales=_med(df, "ev_sales"),
            mean_ev_sales=_mn(df, "ev_sales"),
            median_gross_margin=_med(df, "gross_margin"),
            median_operating_margin=_med(df, "operating_margin"),
            median_net_margin=_med(df, "net_margin"),
            median_roe=_med(df, "roe"),
        )

    def analyse_all_sectors(self) -> list[SectorSummary]:
        """Run :meth:`analyse_sector` for every sector in :data:`SECTOR_PEERS`."""
        return [self.analyse_sector(s) for s in SECTOR_PEERS]

    def sector_comparison_table(self, summaries: list[SectorSummary] | None = None) -> pd.DataFrame:
        """Return a comparison DataFrame across multiple sectors."""
        if summaries is None:
            summaries = self.analyse_all_sectors()
        return pd.DataFrame([s.to_dict() for s in summaries]).set_index("sector")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _med(df: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.median()) if not s.empty else None


def _mn(df: pd.DataFrame, col: str) -> float | None:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else None


def _empty_summary(sector: str) -> SectorSummary:
    return SectorSummary(
        sector=sector,
        companies=pd.DataFrame(),
        count=0,
        median_pe=None, mean_pe=None,
        median_ev_ebitda=None, mean_ev_ebitda=None,
        median_ev_sales=None, mean_ev_sales=None,
        median_gross_margin=None, median_operating_margin=None,
        median_net_margin=None, median_roe=None,
    )
