#!/usr/bin/env python3
"""Example: run a full valuation analysis on an ASX company.

Usage:
    python examples/run_analysis.py CSL
    python examples/run_analysis.py BHP --wacc 0.09 --peers RIO FMG S32 MIN
"""

from __future__ import annotations

import argparse
import logging
import sys

from ainalyst.acquisition import ASXTicker
from ainalyst.config import DCFAssumptions
from ainalyst.fundamentals import build_snapshot
from ainalyst.valuation.dcf import DCFModel
from ainalyst.valuation.comps import CompsModel
from ainalyst.sector import SectorAnalyzer
from ainalyst.report import generate_full_report

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ainalyst valuation")
    parser.add_argument("ticker", help="ASX ticker symbol")
    parser.add_argument("--wacc", type=float, default=0.10)
    parser.add_argument("--tgr", type=float, default=0.025)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--peers", nargs="*", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    # 1. Acquire data
    ticker = ASXTicker(args.ticker)
    log.info("Analysing %s (%s)…", ticker.company_name, ticker.symbol)

    # 2. Parse fundamentals
    snap = build_snapshot(ticker)
    log.info("Revenue: $%,.0f | EBITDA: $%,.0f | Price: $%.2f",
             snap.total_revenue, snap.ebitda, snap.current_price or 0)

    # 3. DCF
    assumptions = DCFAssumptions(
        wacc=args.wacc,
        terminal_growth_rate=args.tgr,
        target_operating_margin=args.margin,
    )
    dcf_result = None
    try:
        dcf_result = DCFModel(assumptions).run(snap)
        log.info("DCF intrinsic value: $%.2f/share (%s)",
                 dcf_result.intrinsic_per_share, dcf_result.valuation_signal)
    except ValueError as exc:
        log.warning("DCF skipped: %s", exc)

    # 4. Comps
    comps_result = None
    try:
        comps_result = CompsModel().run(snap, peer_tickers=args.peers)
        log.info("Comps composite: $%.2f/share (%s)",
                 comps_result.composite_value or 0, comps_result.valuation_signal)
    except Exception as exc:
        log.warning("Comps skipped: %s", exc)

    # 5. Sector context
    sector_summary = None
    try:
        sector_summary = SectorAnalyzer().analyse_sector(snap.sector)
    except Exception:
        pass

    # 6. Generate report
    output = args.output or f"reports/{args.ticker.upper()}_report.html"
    path = generate_full_report(snap, dcf_result, comps_result, sector_summary, output)
    log.info("Report → %s", path.resolve())


if __name__ == "__main__":
    main()
