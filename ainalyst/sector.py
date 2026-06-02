"""Aggregate Sector Analyzer — compute sector-level summary statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.acquisition import fetch_tickers
from ainalyst.config import peers_for_sector, SECTOR_PEERS
from ainalyst.directory import load_directory
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
        use_directory: bool = False,
        max_companies: int = 20,
    ) -> SectorSummary:
        """Analyse a single sector.

        Parameters
        ----------
        sector : str
            GICS sector name (must match keys in ``SECTOR_PEERS`` or
            a GICS industry group name from the ASX directory).
        tickers : list[str] | None
            Override list of tickers; defaults to ``SECTOR_PEERS[sector]``
            unless ``use_directory=True``.
        use_directory : bool
            If True, pull all tickers in this industry group from the full
            ASX directory instead of the hardcoded ``SECTOR_PEERS`` list.
        max_companies : int
            Cap on number of company snapshots to fetch (default 20).
        """
        if tickers is None:
            if use_directory:
                tickers = _tickers_from_directory(sector, max_companies)
                if not tickers:
                    log.warning(
                        "No tickers for sector '%s' in directory, falling back to peers",
                        sector,
                    )
                    try:
                        tickers = peers_for_sector(sector)
                    except KeyError:
                        log.warning("No default peers for sector '%s'", sector)
                        return _empty_summary(sector)
            else:
                try:
                    tickers = peers_for_sector(sector)
                except KeyError:
                    log.warning(
                        "Sector '%s' not in SECTOR_PEERS, trying directory",
                        sector,
                    )
                    tickers = _tickers_from_directory(sector, max_companies)
                    if not tickers:
                        return _empty_summary(sector)

        objs = fetch_tickers(tickers[:max_companies])
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


def _tickers_from_directory(sector: str, max_tickers: int = 20) -> list[str]:
    """Find tickers in a given GICS industry group from the ASX directory."""
    try:
        df = load_directory()
    except Exception as exc:
        log.warning("Failed to load directory: %s", exc)
        return []
    # Case-insensitive partial match on industry name
    mask = df["industry"].str.contains(sector, case=False, na=False)
    matched = df[mask].sort_values("market_cap", ascending=False)
    tickers = [
        t.replace(".AX", "")
        for t in matched["ticker"].head(max_tickers)
    ]
    log.info(
        "Directory lookup: '%s' → %d tickers (top %d by market cap)",
        sector,
        len(matched),
        len(tickers),
    )
    return tickers


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
