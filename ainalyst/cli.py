"""CLI entry point for ainalyst."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from ainalyst.acquisition import ASXTicker
from ainalyst.config import DCFAssumptions, load_custom_assumptions, save_custom_assumptions
from ainalyst.fundamentals import build_snapshot, build_historical_snapshots
from ainalyst.report import generate_full_report, ReportBuilder
from ainalyst.valuation.dcf import DCFModel
from ainalyst.valuation.comps import CompsModel
from ainalyst.valuation.assumptions import derive_assumptions, DerivedAssumptions
from ainalyst.sector import SectorAnalyzer
from ainalyst.ipo import (
    build_ipo_report,
    generate_ipo_report,
    add_to_watchlist,
    remove_from_watchlist,
)

log = logging.getLogger("ainalyst")


def _default_report_path(ticker: str, prefix: str = "report", ext: str = ".html") -> str:
    """Build dated report path: reports/{TICKER}/{TICKER}_{prefix}_{YYYY-MM-DD}.ext"""
    today = date.today().isoformat()
    t = ticker.upper().replace(".AX", "")
    return f"reports/{t}/{t}_{prefix}_{today}{ext}"


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
    p_analyse.add_argument("--wacc", type=float, default=None, help="WACC (default 0.10)")
    p_analyse.add_argument("--tgr", type=float, default=None, help="Terminal growth rate (default 0.025)")
    p_analyse.add_argument("--margin", type=float, default=None, help="Target operating margin (default 0.15)")
    p_analyse.add_argument("--peers", nargs="*", default=None, help="Peer tickers for comps")
    p_analyse.add_argument("--no-comps", action="store_true", help="Skip comps analysis")
    p_analyse.add_argument("--no-dcf", action="store_true", help="Skip DCF analysis")
    p_analyse.add_argument(
        "--mc-filter", type=float, default=None,
        help="Market-cap filter for peers (±50%% default)",
    )
    p_analyse.add_argument(
        "--csv", default=None,
        help="Also write CSV summary to this path",
    )
    p_analyse.add_argument(
        "--history", action="store_true",
        help="Include multi-year financial trend table",
    )
    p_analyse.add_argument(
        "--save", action="store_true",
        help="Persist these DCF assumptions as defaults for this ticker",
    )
    p_analyse.add_argument(
        "--derive", action="store_true",
        help="Compute WACC/TGR/margin from balance sheet + CAPM instead of defaults",
    )

    # ── sector ──────────────────────────────────────────────────────
    p_sector = sub.add_parser("sector", help="Aggregate sector analysis")
    p_sector.add_argument("sector", help="GICS sector name (e.g. 'Materials')")
    p_sector.add_argument("-o", "--output", default=None, help="Output HTML path")

    # ── ipo ─────────────────────────────────────────────────────────
    p_ipo = sub.add_parser("ipo", help="Track recent & upcoming ASX IPOs")
    p_ipo.add_argument(
        "tickers", nargs="*", default=None,
        help="Tickers to scan (default: all recent IPOs from ASX directory)",
    )
    p_ipo.add_argument(
        "--years", type=float, default=2.0,
        help="Look back N years for IPOs (default 2)",
    )
    p_ipo.add_argument(
        "--watchlist", default=None,
        help="Path to YAML watchlist (default: ipo_watchlist.yaml)",
    )
    p_ipo.add_argument(
        "--max", type=int, default=None,
        help="Cap number of detected IPOs enriched (quick runs)",
    )
    p_ipo.add_argument(
        "--refresh", action="store_true",
        help="Force fresh ASX directory fetch (bypass cache)",
    )
    p_ipo.add_argument("-o", "--output", default=None, help="Output HTML path")
    p_ipo.add_argument(
        "--csv", default=None,
        help="Also write CSV to this path",
    )

    # ── ipo add ─────────────────────────────────────────────────────
    p_ipo_add = sub.add_parser("ipo-add", help="Add/update an IPO watchlist entry")
    p_ipo_add.add_argument("--ticker", required=True, help="ASX ticker")
    p_ipo_add.add_argument("--name", default="Unknown", help="Company name")
    p_ipo_add.add_argument("--sector", default="Unknown", help="GICS sector")
    p_ipo_add.add_argument("--date", default=None, help="Expected listing date (YYYY-MM-DD)")
    p_ipo_add.add_argument("--price", type=float, default=None, help="Expected IPO price")
    p_ipo_add.add_argument(
        "--status", default="upcoming",
        choices=["upcoming", "rumoured", "listed", "withdrawn"],
    )
    p_ipo_add.add_argument("--notes", default="", help="Notes / sources")
    p_ipo_add.add_argument("--watchlist", default=None, help="Watchlist path")

    # ── ipo remove ──────────────────────────────────────────────────
    p_ipo_rm = sub.add_parser("ipo-remove", help="Remove an IPO watchlist entry")
    p_ipo_rm.add_argument("ticker", help="Ticker to remove")
    p_ipo_rm.add_argument("--watchlist", default=None, help="Watchlist path")

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
        elif args.command == "ipo":
            return _cmd_ipo(args)
        elif args.command == "ipo-add":
            return _cmd_ipo_add(args)
        elif args.command == "ipo-remove":
            return _cmd_ipo_remove(args)
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
    # Build snapshot once — reuse for both assumption derivation and analysis
    ticker_obj = ASXTicker(args.ticker)
    log.info("Fetching data for %s …", ticker_obj.symbol)
    snap = build_snapshot(ticker_obj)

    # Resolve DCF assumptions (explicit flags > --derive > saved > defaults)
    saved = load_custom_assumptions(args.ticker)
    wacc = args.wacc
    tgr = args.tgr
    margin = args.margin
    derived: DerivedAssumptions | None = None

    if args.derive:
        derived = derive_assumptions(snap, info=ticker_obj.info)
        log.info("\n%s", derived.log_report())
        if wacc is None:
            wacc = derived.wacc
        if tgr is None:
            tgr = derived.terminal_growth_rate
        if margin is None:
            margin = derived.target_operating_margin

    # Fall back to saved assumptions for any flag still None (only if not deriving)
    if saved and not args.derive:
        if wacc is None:
            wacc = saved.get("wacc")
        if tgr is None:
            tgr = saved.get("tgr")
        if margin is None:
            margin = saved.get("margin")
        if wacc is not None and tgr is not None and margin is not None:
            if "computation_log" in saved:
                log.info("Loaded saved assumptions for %s (computed %s):\n%s",
                         args.ticker, saved.get("computed_at", "previously"),
                         saved["computation_log"])
            else:
                log.info("Loaded saved assumptions for %s: wacc=%.1f%% tgr=%.1f%% margin=%.0f%%",
                         args.ticker, wacc * 100, tgr * 100, margin * 100)

    if wacc is None:
        wacc = 0.10
    if tgr is None:
        tgr = 0.025
    if margin is None:
        margin = 0.15
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
            wacc=wacc,
            terminal_growth_rate=tgr,
            target_operating_margin=margin,
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
            comps_result = CompsModel().run(
                snap, peer_tickers=args.peers,
                market_cap_filter=args.mc_filter or None,
            )
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

    output = args.output or _default_report_path(args.ticker, prefix="report")
    csv_path = args.csv or _default_report_path(args.ticker, prefix="report", ext=".csv")

    # Build historical snapshots if requested
    historical = None
    if args.history:
        try:
            historical = build_historical_snapshots(ticker_obj)
            log.info("Loaded %d historical periods", len(historical))
        except Exception as exc:
            log.debug("Historical snapshot skipped: %s", exc)

    path = generate_full_report(
        snap=snap,
        dcf=dcf_result,
        comps=comps_result,
        sector=sector_summary,
        output_path=output,
        historical=historical,
        output_csv=csv_path,
    )
    log.info("Report saved → %s", path.resolve())

    # Persist assumptions if --save requested
    if args.save and dcf_result:
        saved_data: dict[str, Any] = {
            "wacc": wacc,
            "tgr": tgr,
            "margin": margin,
        }
        if derived is not None:
            saved_data.update(derived.to_dict())
            from datetime import datetime, timezone
            saved_data["computed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        saved_path = save_custom_assumptions(args.ticker, saved_data)
        log.info("Assumptions saved for %s → %s", args.ticker, saved_path)

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


def _cmd_ipo(args: argparse.Namespace) -> int:
    def _progress(current: int, total: int, ticker: str) -> None:
        log.info("Scanning %s (%d/%d) …", ticker, current, total)

    report = build_ipo_report(
        tickers=args.tickers or None,
        lookback_years=args.years,
        watchlist_path=args.watchlist,
        progress_callback=_progress,
        max_tickers=args.max,
        refresh=args.refresh,
    )

    listed = report.listed_entries()
    upcoming = report.upcoming_entries()

    log.info(
        "Found %d recent IPOs, %d upcoming/rumoured",
        len(listed),
        len(upcoming),
    )

    if report.median_total_return is not None:
        log.info(
            "Median return: %+.1f%%  Mean: %+.1f%%  Positive: %.0f%%",
            report.median_total_return * 100,
            (report.mean_total_return or 0) * 100,
            (report.pct_positive or 0) * 100,
        )

    if report.best_performer:
        bp = report.best_performer
        wp = report.worst_performer
        log.info(
            "Best: %s (%+.1f%%)  Worst: %s (%+.1f%%)",
            bp.ticker,
            (bp.total_return or 0) * 100,
            wp.ticker if wp else "—",
            (wp.total_return or 0) * 100 if wp else 0,
        )

    for e in listed:
        log.info(
            "  %s  %-30s  Listed %s  IPO $%.2f → $%.2f  %+.1f%%",
            e.ticker,
            e.company_name[:30],
            e.listing_date.strftime("%Y-%m-%d"),
            e.ipo_price or 0,
            e.current_price or 0,
            (e.total_return or 0) * 100,
        )

    output = args.output or "reports/ipo_report.html"
    path = generate_ipo_report(report, output_path=output, output_csv=args.csv)
    log.info("Report saved → %s", path.resolve())
    return 0


def _cmd_ipo_add(args: argparse.Namespace) -> int:
    entry = {
        "ticker": args.ticker.upper(),
        "company_name": args.name,
        "sector": args.sector,
        "expected_date": args.date,
        "ipo_price": args.price,
        "status": args.status,
        "notes": args.notes,
    }
    add_to_watchlist(entry, path=args.watchlist)
    log.info("Added %s (%s) to watchlist", args.ticker, args.status)
    return 0


def _cmd_ipo_remove(args: argparse.Namespace) -> int:
    removed = remove_from_watchlist(args.ticker, path=args.watchlist)
    if removed:
        log.info("Removed %s from watchlist", args.ticker)
    else:
        log.warning("Ticker %s not found in watchlist", args.ticker)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
