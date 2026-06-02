"""IPO Tracker — discover recent ASX IPOs and measure post-listing performance."""

from __future__ import annotations

import csv
import html
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from ainalyst.config import normalise_ticker
from ainalyst.directory import load_directory, recent_ipos_from_directory

log = logging.getLogger(__name__)

_RATE_LIMIT = float(os.environ.get("AINALYST_RATE_LIMIT", 0.15))

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

PERFORMANCE_WINDOWS: dict[str, int] = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "2y": 504,
}
"""Named performance windows → approximate trading days."""


@dataclass(slots=True)
class IPOEntry:
    """A single IPO, either detected from market data or from the watchlist."""

    ticker: str
    company_name: str
    sector: str
    listing_date: datetime
    ipo_price: float | None  # first-day close if detected, stated price if watchlist
    current_price: float | None
    market_cap: float | None
    shares_outstanding: float | None

    # Performance returns keyed by window name (e.g. "1d", "1m", "1y")
    returns: dict[str, float | None] = field(default_factory=dict)

    # Dividend return = cumulative dividends / ipo_price (only for listed entries)
    dividend_return: float | None = None

    # Watchlist-only fields
    status: str = "listed"  # "listed" | "upcoming" | "rumoured" | "withdrawn"
    notes: str = ""

    @property
    def total_return(self) -> float | None:
        """Total return = price return + cumulative dividend yield."""
        if self.ipo_price and self.current_price and self.ipo_price > 0:
            div_yield = self.dividend_return or 0.0
            return (self.current_price / self.ipo_price) - 1.0 + div_yield
        return None

    @property
    def days_since_listing(self) -> int:
        """Calendar days since listing date."""
        return (datetime.now(timezone.utc) - self.listing_date).days

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "listing_date": self.listing_date.strftime("%Y-%m-%d"),
            "ipo_price": self.ipo_price,
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "total_return": self.total_return,
            "dividend_return": self.dividend_return,
            "days_since_listing": self.days_since_listing,
            "status": self.status,
            "notes": self.notes,
        }
        for k, v in self.returns.items():
            d[f"return_{k}"] = v
        return d


@dataclass(slots=True)
class IPOReport:
    """Aggregated IPO tracker results."""

    entries: list[IPOEntry]
    scan_date: datetime
    lookback_years: float
    total_detected: int
    # Aggregate stats
    median_total_return: float | None = None
    mean_total_return: float | None = None
    pct_positive: float | None = None
    best_performer: IPOEntry | None = None
    worst_performer: IPOEntry | None = None

    def listed_entries(self) -> list[IPOEntry]:
        return [e for e in self.entries if e.status == "listed"]

    def upcoming_entries(self) -> list[IPOEntry]:
        return [e for e in self.entries if e.status in ("upcoming", "rumoured")]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.entries])

    def to_csv(self) -> str:
        """Return CSV string of all entries."""
        if not self.entries:
            return ""
        keys = list(self.entries[0].to_dict().keys())
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        for e in self.entries:
            writer.writerow(e.to_dict())
        return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Watchlist I/O
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_WATCHLIST = Path(__file__).parent.parent / "ipo_watchlist.yaml"


def load_watchlist(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the YAML watchlist of upcoming / manually-tracked IPOs."""
    p = Path(path) if path else _DEFAULT_WATCHLIST
    if not p.exists():
        log.debug("No watchlist at %s", p)
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "ipos" not in data:
        log.warning("Watchlist at %s has unexpected format", p)
        return []
    # `ipos:` with no entries (all commented out) parses to None.
    return data["ipos"] or []


def save_watchlist(entries: list[dict[str, Any]], path: Path | str | None = None) -> None:
    """Write the watchlist back to YAML."""
    p = Path(path) if path else _DEFAULT_WATCHLIST
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump({"ipos": entries}, f, default_flow_style=False, sort_keys=False)
    log.info("Watchlist saved → %s (%d entries)", p, len(entries))


def add_to_watchlist(
    entry: dict[str, Any],
    path: Path | str | None = None,
) -> None:
    """Add or update an IPO entry in the watchlist YAML."""
    existing = load_watchlist(path)
    ticker = entry.get("ticker", "").upper()
    # Replace existing entry with same ticker, or append
    replaced = False
    for i, item in enumerate(existing):
        if item.get("ticker", "").upper() == ticker:
            existing[i] = entry
            replaced = True
            break
    if not replaced:
        existing.append(entry)
    save_watchlist(existing, path)


def remove_from_watchlist(
    ticker: str,
    path: Path | str | None = None,
) -> bool:
    """Remove an IPO entry by ticker. Returns True if removed."""
    existing = load_watchlist(path)
    target = ticker.upper()
    new_list = [item for item in existing if item.get("ticker", "").upper() != target]
    if len(new_list) == len(existing):
        log.warning("Ticker %s not found in watchlist", ticker)
        return False
    save_watchlist(new_list, path)
    log.info("Removed %s from watchlist", ticker)
    return True


def _watchlist_to_entries(raw: list[dict[str, Any]]) -> list[IPOEntry]:
    """Convert raw watchlist dicts to IPOEntry objects."""
    entries: list[IPOEntry] = []
    for item in raw:
        try:
            listing_date_str = item.get("listing_date", item.get("expected_date", ""))
            if listing_date_str:
                ld = datetime.strptime(str(listing_date_str), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            else:
                ld = datetime.now(timezone.utc)

            entries.append(
                IPOEntry(
                    ticker=item.get("ticker", "TBD"),
                    company_name=item.get("company_name", "Unknown"),
                    sector=item.get("sector", "Unknown"),
                    listing_date=ld,
                    ipo_price=item.get("ipo_price"),
                    current_price=None,
                    market_cap=None,
                    shares_outstanding=None,
                    status=item.get("status", "upcoming"),
                    notes=item.get("notes", ""),
                )
            )
        except Exception as exc:
            log.warning("Skipping watchlist entry %s: %s", item, exc)
    return entries


# ──────────────────────────────────────────────────────────────────────
# Performance computation
# ──────────────────────────────────────────────────────────────────────


def _compute_returns(
    hist: pd.DataFrame,
    ipo_price: float,
) -> dict[str, float | None]:
    """Compute returns over standard windows from IPO price."""
    if hist.empty or ipo_price <= 0:
        return {}
    returns: dict[str, float | None] = {}
    total_days = len(hist)
    for name, days in PERFORMANCE_WINDOWS.items():
        if days < total_days:
            price_at_window = float(hist["Close"].iloc[days])
            returns[name] = (price_at_window / ipo_price) - 1.0
        else:
            returns[name] = None
    return returns


def _enrich_performance(entry: IPOEntry) -> IPOEntry:
    """Fetch price history + dividends for *entry*.

    Populates ipo_price, current_price, returns, dividend_return.
    """
    time.sleep(_RATE_LIMIT)
    try:
        yft = yf.Ticker(normalise_ticker(entry.ticker))
        hist = yft.history(period="max")
    except Exception as exc:
        log.debug("History fetch failed for %s: %s", entry.ticker, exc)
        return entry
    if hist.empty:
        return entry

    entry.ipo_price = float(hist["Close"].iloc[0])
    entry.current_price = float(hist["Close"].iloc[-1])
    entry.returns = _compute_returns(hist, entry.ipo_price)

    # Cumulative dividends
    try:
        divs = yft.dividends
        if not divs.empty and entry.ipo_price and entry.ipo_price > 0:
            entry.dividend_return = float(divs.sum()) / entry.ipo_price
    except Exception as exc:
        log.debug("Dividend fetch failed for %s: %s", entry.ticker, exc)

    return entry


# ──────────────────────────────────────────────────────────────────────
# Detection: authoritative ASX directory → enrich with yfinance
# ──────────────────────────────────────────────────────────────────────


def detect_via_directory(
    lookback_years: float = 2.0,
    refresh: bool = False,
    max_tickers: int | None = None,
    progress_callback: Any = None,
) -> list[IPOEntry]:
    """Detect recent IPOs from the full ASX directory (authoritative source).

    Listing dates come directly from the directory; yfinance is only called
    to compute post-listing price performance for the recent subset.
    """
    recent = recent_ipos_from_directory(lookback_years=lookback_years)
    if max_tickers is not None:
        recent = recent.head(max_tickers)
    total = len(recent)
    entries: list[IPOEntry] = []

    for i, row in enumerate(recent.itertuples(index=False)):
        code = row.ticker.replace(".AX", "")
        if progress_callback:
            progress_callback(i + 1, total, row.ticker)
        entry = IPOEntry(
            ticker=code,
            company_name=row.name,
            sector=row.industry,
            listing_date=row.listing_date.to_pydatetime(),
            ipo_price=None,
            current_price=None,
            market_cap=float(row.market_cap) if pd.notna(row.market_cap) else None,
            shares_outstanding=None,
            status="listed",
        )
        _enrich_performance(entry)
        entries.append(entry)

    entries.sort(key=lambda e: e.listing_date, reverse=True)
    return entries


def detect_explicit_tickers(
    tickers: list[str],
    lookback_years: float = 2.0,
    progress_callback: Any = None,
) -> list[IPOEntry]:
    """Check specific tickers for IPO status using the ASX directory."""
    df = load_directory()
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(lookback_years * 365.25))

    target_set = {normalise_ticker(t) for t in tickers}
    mask = df["ticker"].isin(target_set) & (df["listing_date"] >= cutoff)
    recent = df[mask].sort_values("listing_date", ascending=False)

    total = len(recent)
    entries: list[IPOEntry] = []
    for i, row in enumerate(recent.itertuples(index=False)):
        code = row.ticker.replace(".AX", "")
        if progress_callback:
            progress_callback(i + 1, total, row.ticker)
        entry = IPOEntry(
            ticker=code,
            company_name=row.name,
            sector=row.industry,
            listing_date=row.listing_date.to_pydatetime(),
            ipo_price=None,
            current_price=None,
            market_cap=float(row.market_cap) if pd.notna(row.market_cap) else None,
            shares_outstanding=None,
            status="listed",
        )
        _enrich_performance(entry)
        entries.append(entry)

    return entries


# ──────────────────────────────────────────────────────────────────────
# High-level: build full IPO report
# ──────────────────────────────────────────────────────────────────────


def build_ipo_report(
    tickers: list[str] | None = None,
    lookback_years: float = 2.0,
    watchlist_path: Path | str | None = None,
    progress_callback: Any = None,
    refresh: bool = False,
    max_tickers: int | None = None,
) -> IPOReport:
    """Build a complete IPO tracker report.

    Parameters
    ----------
    tickers : list[str] | None
        Explicit tickers to scan. If given, filters directory to those tickers.
        If None, detects all recent IPOs from the full ASX directory.
    lookback_years : float
        How far back to look for IPOs (default 2).
    watchlist_path : Path | str | None
        Path to YAML watchlist; ``None`` → default location.
    progress_callback : callable | None
        Optional progress indicator fn(current, total, ticker).
    refresh : bool
        Force a fresh ASX directory fetch (skip cache).
    max_tickers : int | None
        Cap the number of detected IPOs enriched (useful for quick runs).
    """
    # 1. Detect recent IPOs from directory
    if tickers is not None:
        detected = detect_explicit_tickers(
            tickers,
            lookback_years=lookback_years,
            progress_callback=progress_callback,
        )
    else:
        detected = detect_via_directory(
            lookback_years=lookback_years,
            refresh=refresh,
            max_tickers=max_tickers,
            progress_callback=progress_callback,
        )
    detected_syms = {e.ticker for e in detected}

    # 2. Merge watchlist entries
    watchlist_raw = load_watchlist(watchlist_path)
    watchlist_entries = _watchlist_to_entries(watchlist_raw)

    # Enrich listed watchlist entries with live data if not already detected
    for we in watchlist_entries:
        if we.ticker in detected_syms:
            continue
        if we.status == "listed":
            try:
                time.sleep(_RATE_LIMIT)
                symbol = normalise_ticker(we.ticker)
                yft = yf.Ticker(symbol)
                info = yft.info or {}
                we.current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )
                we.market_cap = info.get("marketCap")
                hist = yft.history(period="max")
                if not hist.empty:
                    we.ipo_price = we.ipo_price or float(hist["Close"].iloc[0])
                    we.returns = _compute_returns(hist, we.ipo_price or 0.0)
                try:
                    divs = yft.dividends
                    if not divs.empty and we.ipo_price and we.ipo_price > 0:
                        we.dividend_return = float(divs.sum()) / we.ipo_price
                except Exception:
                    pass
            except Exception as exc:
                log.debug("Watchlist enrich failed for %s: %s", we.ticker, exc)

    all_entries = detected + watchlist_entries

    # 3. Compute aggregate stats over listed entries
    listed = [e for e in all_entries if e.status == "listed" and e.total_return is not None]
    total_returns = [e.total_return for e in listed if e.total_return is not None]

    report = IPOReport(
        entries=all_entries,
        scan_date=datetime.now(timezone.utc),
        lookback_years=lookback_years,
        total_detected=len(detected),
    )

    if total_returns:
        report.median_total_return = float(np.median(total_returns))
        report.mean_total_return = float(np.mean(total_returns))
        report.pct_positive = sum(1 for r in total_returns if r > 0) / len(total_returns)
        report.best_performer = max(listed, key=lambda e: e.total_return or -999)
        report.worst_performer = min(listed, key=lambda e: e.total_return or 999)

    return report


# ──────────────────────────────────────────────────────────────────────
# HTML report generation
# ──────────────────────────────────────────────────────────────────────

_CSS = """
<style>
body{font-family:system-ui,sans-serif;margin:2em auto;max-width:1200px;color:#1a1a2e;background:#f8f9fa}
h1{border-bottom:3px solid #0d6efd;padding-bottom:.3em}
h2{color:#0d6efd;margin-top:2em}
.meta{color:#6c757d;font-size:.9em}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.12);padding:1.2em;margin:.8em 0}
.grid{display:grid;gap:1em}.grid-3{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.grid-4{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.kv{display:flex;justify-content:space-between;padding:.3em 0;border-bottom:1px solid #eee}
.k{font-weight:600;color:#495057}.v{text-align:right}
table{width:100%;border-collapse:collapse;margin:1em 0}
th{background:#0d6efd;color:#fff;padding:.6em .8em;text-align:left;font-size:.85em}
td{padding:.5em .8em;border-bottom:1px solid #eee;font-size:.85em}
tr:hover{background:#f0f6ff}
.positive{color:#198754;font-weight:600}.negative{color:#dc3545;font-weight:600}
.badge{display:inline-block;padding:.15em .6em;border-radius:4px;font-size:.8em;font-weight:600}
.badge-listed{background:#d1e7dd;color:#0f5132}
.badge-upcoming{background:#cff4fc;color:#055160}
.badge-rumoured{background:#fff3cd;color:#664d03}
.badge-withdrawn{background:#f8d7da;color:#842029}
footer{margin-top:3em;padding:1em 0;border-top:1px solid #dee2e6;color:#6c757d;font-size:.85em}
@media(max-width:768px){
  body{margin:1em;padding:0}
  .grid-3,.grid-4{grid-template-columns:1fr}
  table{font-size:.78em}
  th,td{padding:.35em .4em}
}
@media print{
  body{background:#fff;color:#000;max-width:100%}
  .card{box-shadow:none;border:1px solid #ccc;page-break-inside:avoid}
}
</style>
"""


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    css = "positive" if v >= 0 else "negative"
    return f'<span class="{css}">{v:+.1%}</span>'


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:,.2f}"


def _fmt_mcap(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1e9:
        return f"${v / 1e9:,.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:,.0f}M"
    return f"${v:,.0f}"


def _status_badge(status: str) -> str:
    return f'<span class="badge badge-{status}">{status.title()}</span>'


def generate_ipo_report(
    report: IPOReport,
    output_path: str | Path = "ipo_report.html",
    output_csv: str | Path | None = None,
) -> Path:
    """Generate a standalone HTML report for IPO tracker results.

    Also writes a CSV if ``output_csv`` is provided.
    """
    parts: list[str] = []

    # Header
    parts.append(
        f"<h1>ASX IPO Tracker</h1>"
        f'<p class="meta">Lookback: {report.lookback_years:.0f} years · '
        f"Scan date: {report.scan_date.strftime('%Y-%m-%d %H:%M UTC')} · "
        f"IPOs found: {report.total_detected}</p>"
    )

    # Summary stats
    listed = report.listed_entries()
    upcoming = report.upcoming_entries()

    parts.append('<div class="card"><h2>Overview</h2><div class="grid grid-4">')
    parts.append(
        '<div class="card">'
        f'<div class="kv"><span class="k">Listed IPOs</span><span class="v">{len(listed)}</span></div>'
        f'<div class="kv"><span class="k">Upcoming / Rumoured</span><span class="v">{len(upcoming)}</span></div>'
        f'<div class="kv"><span class="k">Report Format</span>'
        f'<span class="v"><a href="#csv-download" style="font-size:.8rem">CSV ↓</a></span></div>'
        "</div>"
    )
    parts.append(
        '<div class="card">'
        f'<div class="kv"><span class="k">Median Return</span><span class="v">{_fmt_pct(report.median_total_return)}</span></div>'
        f'<div class="kv"><span class="k">Mean Return</span><span class="v">{_fmt_pct(report.mean_total_return)}</span></div>'
        "</div>"
    )
    parts.append(
        '<div class="card">'
        f'<div class="kv"><span class="k">% Positive</span>'
        f'<span class="v">{report.pct_positive:.0%}</span></div>'
        "</div>"
        if report.pct_positive is not None
        else ""
    )
    if report.best_performer:
        bp = report.best_performer
        parts.append(
            '<div class="card">'
            f'<div class="kv"><span class="k">Best</span>'
            f'<span class="v">{html.escape(bp.ticker)} {_fmt_pct(bp.total_return)}</span></div>'
            f'<div class="kv"><span class="k">Worst</span>'
            f'<span class="v">{html.escape(report.worst_performer.ticker if report.worst_performer else "—")} '
            f"{_fmt_pct(report.worst_performer.total_return if report.worst_performer else None)}</span></div>"
            "</div>"
        )
    parts.append("</div></div>")

    # Listed IPOs table
    if listed:
        parts.append('<div class="card"><h2>Recent IPOs — Performance</h2>')
        parts.append(
            "<table><thead><tr>"
            "<th>Ticker</th><th>Company</th><th>Sector</th><th>Listed</th>"
            "<th>IPO Price</th><th>Current</th><th>Total Return</th>"
            "<th>1W</th><th>1M</th><th>3M</th><th>6M</th><th>1Y</th>"
            "<th>Mkt Cap</th>"
            "</tr></thead><tbody>"
        )
        for e in sorted(listed, key=lambda x: x.listing_date, reverse=True):
            parts.append(
                f"<tr>"
                f"<td><strong>{html.escape(e.ticker)}</strong></td>"
                f"<td>{html.escape(e.company_name)}</td>"
                f"<td>{html.escape(e.sector)}</td>"
                f"<td>{e.listing_date.strftime('%Y-%m-%d')}</td>"
                f"<td>{_fmt_price(e.ipo_price)}</td>"
                f"<td>{_fmt_price(e.current_price)}</td>"
                f"<td>{_fmt_pct(e.total_return)}</td>"
                f"<td>{_fmt_pct(e.returns.get('1w'))}</td>"
                f"<td>{_fmt_pct(e.returns.get('1m'))}</td>"
                f"<td>{_fmt_pct(e.returns.get('3m'))}</td>"
                f"<td>{_fmt_pct(e.returns.get('6m'))}</td>"
                f"<td>{_fmt_pct(e.returns.get('1y'))}</td>"
                f"<td>{_fmt_mcap(e.market_cap)}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    # Upcoming / rumoured
    if upcoming:
        parts.append('<div class="card"><h2>Upcoming & Rumoured IPOs</h2>')
        parts.append(
            "<table><thead><tr>"
            "<th>Company</th><th>Ticker</th><th>Sector</th>"
            "<th>Expected Date</th><th>IPO Price</th><th>Status</th><th>Notes</th>"
            "</tr></thead><tbody>"
        )
        for e in sorted(upcoming, key=lambda x: x.listing_date):
            parts.append(
                f"<tr>"
                f"<td><strong>{html.escape(e.company_name)}</strong></td>"
                f"<td>{html.escape(e.ticker)}</td>"
                f"<td>{html.escape(e.sector)}</td>"
                f"<td>{e.listing_date.strftime('%Y-%m-%d')}</td>"
                f"<td>{_fmt_price(e.ipo_price)}</td>"
                f"<td>{_status_badge(e.status)}</td>"
                f"<td>{html.escape(e.notes)}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    # CSV download section (embedded)
    csv_data = report.to_csv()
    if csv_data:
        csv_b64 = (
            '<div class="card" id="csv-download"><h2>CSV Export</h2>'
            "<p style='font-size:.85rem;color:#6c757d'>"
            '<a href="#" onclick="downloadCSV()">Download CSV</a> · '
            f"{len(report.entries)} rows</p>"
            '<pre id="csv-data" style="display:none">'
            f"{html.escape(csv_data)}</pre>"
            "</div>"
            "<script>"
            "function downloadCSV(){"
            "var csv=document.getElementById('csv-data').innerText;"
            "var blob=new Blob([csv],{type:'text/csv'});"
            "var a=document.createElement('a');"
            "a.href=URL.createObjectURL(blob);"
            "a.download='ipo_report.csv';a.click();}"
            "</script>"
        )
        parts.append(csv_b64)

    # Footer
    body = "\n".join(parts)
    doc = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ASX IPO Tracker — ainalyst</title>{_CSS}</head><body>{body}"
        "<footer>Generated by <strong>ainalyst v0.1.0</strong> · "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        "</footer></body></html>"
    )

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    log.info("IPO report saved to %s", p)

    # Write standalone CSV if requested
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(csv_data, encoding="utf-8")
        log.info("IPO CSV saved to %s", csv_path)

    return p
