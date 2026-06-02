"""Central configuration, sector mappings, and default assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ──────────────────────────────────────────────────────────────────────
# ASX ticker handling
# ──────────────────────────────────────────────────────────────────────

ASX_SUFFIX: Final[str] = ".AX"


def normalise_ticker(ticker: str) -> str:
    """Ensure *ticker* carries the `.AX` suffix for Yahoo Finance lookups."""
    t = ticker.strip().upper()
    if not t.endswith(ASX_SUFFIX):
        t += ASX_SUFFIX
    return t


# ──────────────────────────────────────────────────────────────────────
# GICS Sector → Representative ASX Tickers (default peer sets)
# ──────────────────────────────────────────────────────────────────────

SECTOR_PEERS: dict[str, list[str]] = {
    "Materials": [
        "BHP", "RIO", "FMG", "S32", "MIN", "IGO", "ILU", "OZL", "SFR", "AWC",
    ],
    "Basic Materials": [
        "BHP", "RIO", "FMG", "S32", "MIN", "IGO", "ILU", "OZL", "SFR", "AWC",
    ],
    "Financials": [
        "CBA", "NAB", "WBC", "ANZ", "MQG", "SUN", "IAG", "QBE", "MPL", "BEN",
    ],
    "Health Care": [
        "CSL", "COH", "RMD", "SHL", "FPH", "PME", "PRN", "NAN", "TLX", "IMU",
    ],
    "Information Technology": [
        "XRO", "WTC", "CPU", "ALU", "MP1", "TNE", "DTC", "NXT", "PME", "TYR",
    ],
    "Energy": [
        "WDS", "STO", "ORG", "WHC", "NHC", "BPT", "KAR", "STX", "VEA", "COE",
    ],
    "Consumer Discretionary": [
        "WES", "HVN", "JBH", "SUL", "LOV", "PMV", "ADH", "BRG", "NCK", "BBN",
    ],
    "Consumer Staples": [
        "WOW", "COL", "TWE", "A2M", "ING", "CGC", "BAP", "BGA", "GNC", "SHV",
    ],
    "Industrials": [
        "TCL", "SYD", "BXB", "AZJ", "DOW", "SEK", "REH", "NWH", "MND", "SVW",
    ],
    "Real Estate": [
        "GMG", "SCG", "VCX", "MGR", "GPT", "CHC", "BWP", "CQR", "NSR", "ABP",
    ],
    "Communication Services": [
        "TLS", "TPG", "REA", "CAR", "NWS", "SWM", "OML", "SGR", "SKT", "UNI",
    ],
    "Utilities": [
        "APA", "AGL", "ORG", "AST", "SKI", "DBI", "MCY", "INF", "CEN", "GNX",
    ],
}


def peers_for_sector(sector: str) -> list[str]:
    """Return the default peer ticker list for *sector* (case-insensitive match)."""
    key = next((k for k in SECTOR_PEERS if k.lower() == sector.lower()), None)
    if key is None:
        raise KeyError(f"Unknown sector '{sector}'. Known: {list(SECTOR_PEERS)}")
    return SECTOR_PEERS[key]


# ──────────────────────────────────────────────────────────────────────
# DCF default assumptions
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DCFAssumptions:
    """Immutable container for DCF model inputs."""

    projection_years: int = 5
    revenue_growth_rates: list[float] = field(
        default_factory=lambda: [0.08, 0.07, 0.06, 0.05, 0.04]
    )
    target_operating_margin: float = 0.15
    tax_rate: float = 0.30            # Australian corporate tax rate
    capex_pct_revenue: float = 0.05
    delta_nwc_pct_revenue: float = 0.02
    wacc: float = 0.10
    terminal_growth_rate: float = 0.025
    # Sensitivity sweep ranges
    wacc_range: list[float] = field(
        default_factory=lambda: [0.08, 0.09, 0.10, 0.11, 0.12]
    )
    tgr_range: list[float] = field(
        default_factory=lambda: [0.015, 0.020, 0.025, 0.030, 0.035]
    )


# ──────────────────────────────────────────────────────────────────────
# Comps defaults
# ──────────────────────────────────────────────────────────────────────

COMPS_MULTIPLES: Final[list[str]] = ["P/E", "EV/EBITDA", "EV/Sales"]

# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

REPORT_TITLE_DEFAULT: Final[str] = "Ainalyst Equity Research Report"
