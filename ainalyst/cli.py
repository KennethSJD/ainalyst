"""CLI entry point for ainalyst."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ainalyst.acquisition import ASXTicker
from ainalyst.config import DCFAssumptions
from ainalyst.fundamentals import build_snapshot
from ainalyst.valuation.dcf import DCFModel
from ainalyst.valuation.comps import CompsModel
from ainalyst.sector import SectorAnalyzer
from ainalyst.report import generate_full_report

log = logging.getLogger("ainalyst")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ainalyst",
        description="ASX company valuation engine — DCF, Trading Comps, Sector Analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyse ─────────────────────────────────────────────────────
    p_analyse = sub.add_parser("analyse", help="Full valuation report for a single ticker")
    p_analyse.add_argument("ticker", help="ASX ticker (e.g. CSL, BHP)")
    p_analyse.add_argument("-o", "--output", default=None, help="Output HTML path")
    p_analyse.add_argument("--wacc", type=float, default=0.10, help="WACC (default 0.10)")
    p_analyse.add_argument("--tgr", type=float, default=0.025, help="Terminal growth rate")
    p_analyse.add_argument("--margin", type=float, default=0.15, help="Target operating margin")
    p_analyse.add_argument("--peers", nargs="*", default=None, help="Peer tickers for comps")
    p_analyse.add_argument("--no-comps", action="store_true", help="Skip comps analysis")
    p_analyse.add_argument("--no-dcf", action="store_true", help="Skip DCF analysis")

    # ── sector ──────────────────────────────────────────────────────
    p_sector = sub.add_parser("sector", help="Aggregate sector analysis")
    p_sector.add_argument("sector", help="GICS sector name (e.g. 'Materials')")
    p_sector.add_argument("-o", "--output", default=None, help="Output HTML path")

    # ── common ──────────────────────────────────────────────────────
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    try:
        if args.command == "analyse":
            return _cmd_analyse(args)
        elif args.command == "sector":
            return _cmd_sector(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as exc:
        log.error("Fatal: %s", exc, exc_info=args.verbose)
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────────
# Command implementations
# ──────────────────────────────────────────────────────────────────────

def _cmd_analyse(args: argparse.Namespace) -> int:
    ticker_obj = ASXTicker(args.ticker)
    log.info("Fetching data for %s …", ticker_obj.symbol)

    snap = build_snapshot(ticker_obj)
    log.info(
        "%s  Revenue=%s  EBITDA=%s  Price=%s",
        snap.ticker,
        f"${snap.total_revenue/1e6:,.0f}M" if snap.total_revenue else "N/A",
        f"${snap.ebitda/1e6:,.0f}M" if snap.ebitda else "N/A",
        f"${snap.current_price:,.2f}" if snap.current_price else "N/A",
    )

    dcf_result = None
    if not args.no_dcf:
        assumptions = DCFAssumptions(
            wacc=args.wacc,
            terminal_growth_rate=args.tgr,
            target_operating_margin=args.margin,
        )
        try:
            dcf_result = DCFModel(assumptions).run(snap)
            log.info(
                "DCF → Intrinsic $%.2f  (%s)",
                dcf_result.intrinsic_per_share,
                dcf_result.valuation_signal,
            )
        except ValueError as exc:
            log.warning("DCF skipped: %s", exc)

    comps_result = None
    if not args.no_comps:
        try:
            comps_result = CompsModel().run(snap, peer_tickers=args.peers)
            log.info(
                "Comps → Composite $%.2f  (%s)",
                comps_result.composite_value or 0,
                comps_result.valuation_signal,
            )
        except Exception as exc:
            log.warning("Comps skipped: %s", exc)

    # Sector context (best-effort)
    sector_summary = None
    try:
        sector_summary = SectorAnalyzer().analyse_sector(snap.sector)
    except Exception as exc:
        log.debug("Sector analysis skipped: %s", exc)

    output = args.output or f"reports/{args.ticker.upper()}_report.html"
    path = generate_full_report(
        snap=snap,
        dcf=dcf_result,
        comps=comps_result,
        sector=sector_summary,
        output_path=output,
    )
    log.info("Report saved → %s", path.resolve())
    return 0


def _cmd_sector(args: argparse.Namespace) -> int:
    analyzer = SectorAnalyzer()
    summary = analyzer.analyse_sector(args.sector)

    log.info(
        "Sector '%s': %d companies — Median P/E=%s  EV/EBITDA=%s",
        summary.sector,
        summary.count,
        f"{summary.median_pe:.1f}" if summary.median_pe else "N/A",
        f"{summary.median_ev_ebitda:.1f}" if summary.median_ev_ebitda else "N/A",
    )

    if args.output:
        from ainalyst.report import ReportBuilder
        rb = ReportBuilder()
        rb._parts.append(f"<h1>Sector Report: {summary.sector}</h1>")
        rb.add_sector_summary(summary)
        rb.save(args.output, title=f"Ainalyst — {summary.sector} Sector")
        log.info("Report saved → %s", Path(args.output).resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
