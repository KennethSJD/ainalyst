"""Reporting Engine — generate standalone interactive HTML reports."""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.fundamentals import FundamentalSnapshot
from ainalyst.valuation.dcf import DCFResult
from ainalyst.valuation.comps import CompsResult
from ainalyst.sector import SectorSummary

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

def _fmt_currency(val: float | None, decimals: int = 2, symbol: str = "$") -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e9:
        return f"{symbol}{val / 1e9:,.{decimals}f}B"
    if abs(val) >= 1e6:
        return f"{symbol}{val / 1e6:,.{decimals}f}M"
    if abs(val) >= 1e3:
        return f"{symbol}{val / 1e3:,.{decimals}f}K"
    return f"{symbol}{val:,.{decimals}f}"


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:+.{decimals}f}%"


def _fmt_multiple(val: float | None, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}x"


def _fmt_price(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def _signal_badge(signal: str) -> str:
    colours = {
        "UNDERVALUED": ("#065f46", "#d1fae5"),
        "OVERVALUED": ("#991b1b", "#fee2e2"),
        "FAIRLY VALUED": ("#92400e", "#fef3c7"),
        "N/A": ("#6b7280", "#f3f4f6"),
    }
    fg, bg = colours.get(signal, ("#6b7280", "#f3f4f6"))
    return (
        f'<span style="display:inline-block;padding:4px 14px;border-radius:9999px;'
        f'font-weight:700;font-size:0.85rem;color:{fg};background:{bg}">'
        f'{html.escape(signal)}</span>'
    )


# ──────────────────────────────────────────────────────────────────────
# HTML template fragments
# ──────────────────────────────────────────────────────────────────────

_CSS = """
<style>
  :root{--bg:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;
    --accent:#2563eb;--accent-light:#dbeafe}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:Inter,system-ui,-apple-system,sans-serif;background:var(--bg);
    color:var(--text);line-height:1.6;padding:2rem;max-width:1200px;margin:0 auto}
  h1{font-size:1.75rem;margin-bottom:.25rem}
  h2{font-size:1.25rem;color:var(--accent);margin:2rem 0 .75rem;
    border-bottom:2px solid var(--accent-light);padding-bottom:.35rem}
  h3{font-size:1.05rem;margin:1.25rem 0 .5rem}
  .meta{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}
  .card{background:var(--card);border:1px solid var(--border);border-radius:.75rem;
    padding:1.25rem;margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .grid{display:grid;gap:1rem}
  .grid-2{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
  .grid-3{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
  .kv{display:flex;justify-content:space-between;padding:.35rem 0;
    border-bottom:1px solid var(--border)}
  .kv:last-child{border-bottom:none}
  .kv .k{color:var(--muted);font-size:.85rem}
  .kv .v{font-weight:600;font-size:.9rem}
  table{width:100%;border-collapse:collapse;font-size:.85rem;margin:.5rem 0}
  th,td{text-align:right;padding:.45rem .6rem;border-bottom:1px solid var(--border)}
  th{background:var(--accent-light);color:var(--accent);font-weight:600;
    position:sticky;top:0}
  td:first-child,th:first-child{text-align:left}
  tr:hover td{background:#f1f5f9}
  .football{display:flex;align-items:center;gap:0;height:36px;margin:.75rem 0;
    border-radius:6px;overflow:hidden;font-size:.75rem;font-weight:600}
  .football>div{height:100%;display:flex;align-items:center;justify-content:center;
    color:#fff;white-space:nowrap;padding:0 10px;min-width:60px}
  .marker{position:relative;width:3px;background:#1e293b;z-index:2}
  .marker::after{content:attr(data-label);position:absolute;top:-22px;left:50%;
    transform:translateX(-50%);font-size:.7rem;color:var(--text);white-space:nowrap;
    font-weight:700}
  .legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.78rem;color:var(--muted);margin-top:.35rem}
  .legend span::before{content:'';display:inline-block;width:10px;height:10px;
    border-radius:2px;margin-right:4px;vertical-align:middle}
  .l-dcf::before{background:#2563eb}
  .l-comps::before{background:#7c3aed}
  .l-price::before{background:#1e293b}
  @media(max-width:768px){
    body{padding:1rem}
    .grid-2,.grid-3{grid-template-columns:1fr}
    table{font-size:.75em}
    th,td{padding:.35em .4em}
  }
  @media print{
    body{background:#fff;color:#000;max-width:100%}
    .card{box-shadow:none;border:1px solid #ccc;page-break-inside:avoid}
  }
  .trend-up{color:#198754}.trend-down{color:#dc3545}
  .trend-flat{color:#6c757d}
  footer{margin-top:3rem;text-align:center;color:var(--muted);font-size:.75rem}
</style>
"""


# ──────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────

class ReportBuilder:
    """Assemble a full equity-research HTML report."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    # ── Section generators ──────────────────────────────────────────

    def add_header(self, snap: FundamentalSnapshot) -> "ReportBuilder":
        self._parts.append(
            f"<h1>{html.escape(snap.company_name)} ({html.escape(snap.ticker)})</h1>"
            f'<p class="meta">Sector: {html.escape(snap.sector)} · '
            f'Currency: {html.escape(snap.currency)} · '
            f'Report generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>'
        )
        return self

    def add_executive_summary(
        self,
        snap: FundamentalSnapshot,
        dcf: DCFResult | None = None,
        comps: CompsResult | None = None,
    ) -> "ReportBuilder":
        parts: list[str] = ['<div class="card"><h2>Executive Summary</h2><div class="grid grid-3">']

        # Price card
        parts.append(
            '<div class="card">'
            f'<div class="kv"><span class="k">Current Price</span>'
            f'<span class="v">{_fmt_price(snap.current_price)}</span></div>'
            f'<div class="kv"><span class="k">Market Cap</span>'
            f'<span class="v">{_fmt_currency(snap.market_cap)}</span></div>'
            f'<div class="kv"><span class="k">EV</span>'
            f'<span class="v">{_fmt_currency(snap.enterprise_value)}</span></div>'
            "</div>"
        )

        # DCF card
        if dcf:
            parts.append(
                '<div class="card">'
                f'<div class="kv"><span class="k">DCF Intrinsic Value</span>'
                f'<span class="v">{_fmt_price(dcf.intrinsic_per_share)}</span></div>'
                f'<div class="kv"><span class="k">Margin of Safety</span>'
                f'<span class="v">{_fmt_pct(dcf.margin_of_safety_pct)}</span></div>'
                f'<div class="kv"><span class="k">Signal</span>'
                f'<span class="v">{_signal_badge(dcf.valuation_signal)}</span></div>'
                "</div>"
            )

        # Comps card
        if comps:
            parts.append(
                '<div class="card">'
                f'<div class="kv"><span class="k">Comps Composite</span>'
                f'<span class="v">{_fmt_price(comps.composite_value)}</span></div>'
                f'<div class="kv"><span class="k">Margin of Safety</span>'
                f'<span class="v">{_fmt_pct(comps.margin_of_safety_pct)}</span></div>'
                f'<div class="kv"><span class="k">Signal</span>'
                f'<span class="v">{_signal_badge(comps.valuation_signal)}</span></div>'
                "</div>"
            )

        parts.append("</div></div>")
        self._parts.append("\n".join(parts))
        return self

    def add_financial_snapshot(self, snap: FundamentalSnapshot) -> "ReportBuilder":
        d = snap.to_dict()
        income_keys = [
            ("Revenue", "total_revenue"), ("Gross Profit", "gross_profit"),
            ("EBIT", "operating_income"), ("EBITDA", "ebitda"),
            ("Net Income", "net_income"),
        ]
        balance_keys = [
            ("Total Assets", "total_assets"), ("Total Liabilities", "total_liabilities"),
            ("Total Debt", "total_debt"), ("Cash & Equivalents", "cash_and_equivalents"),
            ("Equity", "total_equity"),
        ]
        margin_keys = [
            ("Gross Margin", "gross_margin"), ("Operating Margin", "operating_margin"),
            ("Net Margin", "net_margin"), ("ROE", "roe"),
        ]
        multiple_keys = [
            ("P/E", "pe_ratio"), ("EV/EBITDA", "ev_ebitda"), ("EV/Sales", "ev_sales"),
        ]

        h = '<div class="card"><h2>Financial Snapshot</h2><div class="grid grid-2">'
        h += self._kv_card("Income Statement", [(k, _fmt_currency(d[v])) for k, v in income_keys])
        h += self._kv_card("Balance Sheet", [(k, _fmt_currency(d[v])) for k, v in balance_keys])
        h += self._kv_card("Margins", [(k, _fmt_pct(d[v])) for k, v in margin_keys])
        h += self._kv_card("Multiples", [(k, _fmt_multiple(d[v])) for k, v in multiple_keys])
        h += "</div></div>"
        self._parts.append(h)
        return self

    def add_dcf_detail(self, dcf: DCFResult) -> "ReportBuilder":
        h = '<div class="card"><h2>DCF Valuation</h2>'

        # Projection table
        h += "<h3>Projected Free Cash Flows</h3><table><tr><th>Year</th>"
        for i in range(len(dcf.projected_fcf)):
            h += f"<th>Y{i+1}</th>"
        h += "</tr><tr><td>FCF</td>"
        for f in dcf.projected_fcf:
            h += f"<td>{_fmt_currency(f)}</td>"
        h += "</tr><tr><td>PV(FCF)</td>"
        for pv in dcf.pv_fcfs:
            h += f"<td>{_fmt_currency(pv)}</td>"
        h += "</tr></table>"

        # Terminal
        h += (
            '<div class="grid grid-3" style="margin-top:1rem">'
            + self._kv_card("Terminal Value", [
                ("Undiscounted TV", _fmt_currency(dcf.terminal_value)),
                ("PV of TV", _fmt_currency(dcf.pv_terminal)),
            ])
            + self._kv_card("Bridge", [
                ("Enterprise Value", _fmt_currency(dcf.enterprise_value)),
                ("Equity Value", _fmt_currency(dcf.equity_value)),
                ("Intrinsic / Share", _fmt_price(dcf.intrinsic_per_share)),
            ])
            + "</div>"
        )

        # Sensitivity
        h += "<h3>Sensitivity Matrix (Intrinsic Value / Share)</h3>"
        h += "<p style='font-size:.8rem;color:var(--muted)'>Rows = WACC · Columns = Terminal Growth Rate</p>"
        h += self._df_to_table(dcf.sensitivity, index_label="WACC \\ TGR")

        h += "</div>"
        self._parts.append(h)
        return self

    def add_comps_detail(self, comps: CompsResult) -> "ReportBuilder":
        h = '<div class="card"><h2>Trading Comparables</h2>'
        h += f"<p style='font-size:.85rem;color:var(--muted)'>Peers analysed: {comps.peer_count}</p>"

        if not comps.peer_table.empty:
            display = comps.peer_table.copy()
            for col in ("Revenue", "EBITDA", "Net Income", "Market Cap", "EV"):
                if col in display.columns:
                    display[col] = display[col].apply(lambda v: _fmt_currency(v))
            for col in ("P/E", "EV/EBITDA", "EV/Sales"):
                if col in display.columns:
                    display[col] = display[col].apply(lambda v: _fmt_multiple(v))
            h += self._df_to_table(display)

        h += '<div class="grid grid-3" style="margin-top:1rem">'
        h += self._kv_card("Median Multiples", [
            ("P/E", _fmt_multiple(comps.median_pe)),
            ("EV/EBITDA", _fmt_multiple(comps.median_ev_ebitda)),
            ("EV/Sales", _fmt_multiple(comps.median_ev_sales)),
        ])
        h += self._kv_card("Implied Value / Share", [
            ("via P/E", _fmt_price(comps.implied_pe_value)),
            ("via EV/EBITDA", _fmt_price(comps.implied_ev_ebitda_value)),
            ("via EV/Sales", _fmt_price(comps.implied_ev_sales_value)),
            ("Composite", _fmt_price(comps.composite_value)),
        ])
        h += "</div></div>"
        self._parts.append(h)
        return self

    def add_football_field(
        self,
        current_price: float | None,
        dcf: DCFResult | None = None,
        comps: CompsResult | None = None,
    ) -> "ReportBuilder":
        """CSS-styled valuation range ('football field') chart."""
        ranges: list[tuple[str, float, float, str]] = []  # (label, lo, hi, color)

        if dcf and not dcf.sensitivity.empty:
            vals = dcf.sensitivity.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                ranges.append(("DCF Range", float(np.min(vals)), float(np.max(vals)), "#2563eb"))

        if comps:
            imp = [v for v in (comps.implied_pe_value, comps.implied_ev_ebitda_value,
                               comps.implied_ev_sales_value) if v is not None]
            if imp:
                ranges.append(("Comps Range", min(imp), max(imp), "#7c3aed"))

        if not ranges:
            return self

        # Determine scale
        all_vals = [v for _, lo, hi, _ in ranges for v in (lo, hi)]
        if current_price is not None:
            all_vals.append(current_price)
        lo_bound = max(0, min(all_vals) * 0.8)
        hi_bound = max(all_vals) * 1.2
        span = hi_bound - lo_bound if hi_bound > lo_bound else 1.0

        def pct(v: float) -> float:
            return max(0, min(100, (v - lo_bound) / span * 100))

        h = '<div class="card"><h2>Valuation Range (Football Field)</h2>'
        for label, lo, hi, color in ranges:
            left = pct(lo)
            width = pct(hi) - left
            h += (
                f'<p style="font-size:.8rem;font-weight:600;margin-top:.5rem">{html.escape(label)}</p>'
                f'<div class="football" style="background:#e2e8f0">'
                f'<div style="width:{left}%;background:transparent"></div>'
                f'<div style="width:{max(width, 2)}%;background:{color};opacity:0.75">'
                f'{_fmt_price(lo)} – {_fmt_price(hi)}</div>'
            )
            if current_price is not None:
                mp = pct(current_price)
                h += f'<div class="marker" style="position:absolute;left:{mp}%" data-label="Price {_fmt_price(current_price)}"></div>'
            h += "</div>"

        h += (
            '<div class="legend">'
            '<span class="l-dcf">DCF</span>'
            '<span class="l-comps">Comps</span>'
            '<span class="l-price">Current Price</span>'
            "</div></div>"
        )
        self._parts.append(h)
        return self

    def add_sector_summary(self, summary: SectorSummary) -> "ReportBuilder":
        h = f'<div class="card"><h2>Sector Overview: {html.escape(summary.sector)}</h2>'
        h += f"<p style='font-size:.85rem;color:var(--muted)'>{summary.count} companies analysed</p>"
        h += '<div class="grid grid-2">'
        h += self._kv_card("Multiples (Median)", [
            ("P/E", _fmt_multiple(summary.median_pe)),
            ("EV/EBITDA", _fmt_multiple(summary.median_ev_ebitda)),
            ("EV/Sales", _fmt_multiple(summary.median_ev_sales)),
        ])
        h += self._kv_card("Margins (Median)", [
            ("Gross", _fmt_pct(summary.median_gross_margin)),
            ("Operating", _fmt_pct(summary.median_operating_margin)),
            ("Net", _fmt_pct(summary.median_net_margin)),
            ("ROE", _fmt_pct(summary.median_roe)),
        ])
        h += "</div>"

        if not summary.companies.empty:
            display = summary.companies[
                ["ticker", "company_name", "total_revenue", "ebitda", "net_income",
                 "pe_ratio", "ev_ebitda", "ev_sales"]
            ].copy()
            display.columns = ["Ticker", "Company", "Revenue", "EBITDA", "Net Income",
                               "P/E", "EV/EBITDA", "EV/Sales"]
            for col in ("Revenue", "EBITDA", "Net Income"):
                display[col] = display[col].apply(lambda v: _fmt_currency(v))
            for col in ("P/E", "EV/EBITDA", "EV/Sales"):
                display[col] = display[col].apply(lambda v: _fmt_multiple(v))
            h += "<h3>Company Detail</h3>" + self._df_to_table(display)

        h += "</div>"
        self._parts.append(h)
        return self

    def add_assumptions(
        self,
        dcf: DCFResult | None = None,
        comps: CompsResult | None = None,
    ) -> "ReportBuilder":
        h = '<div class="card"><h2>Data Definitions & Assumptions</h2>'
        if dcf:
            a = dcf.assumptions
            h += "<h3>DCF Assumptions</h3>"
            h += '<div class="grid grid-2">'
            h += self._kv_card("Growth & Margins", [
                ("Projection Years", str(a.get("projection_years", "—"))),
                ("Revenue Growth Rates", ", ".join(f"{r:.1%}" for r in a.get("revenue_growth_rates", []))),
                ("Target Op. Margin", f"{a.get('target_operating_margin', 0):.1%}"),
            ])
            h += self._kv_card("Discount & Terminal", [
                ("Tax Rate", f"{a.get('tax_rate', 0):.1%}"),
                ("CapEx % Rev", f"{a.get('capex_pct_revenue', 0):.1%}"),
                ("ΔNWC % Rev", f"{a.get('delta_nwc_pct_revenue', 0):.1%}"),
                ("WACC", f"{a.get('wacc', 0):.1%}"),
                ("Terminal Growth", f"{a.get('terminal_growth_rate', 0):.1%}"),
            ])
            h += "</div>"
        if comps:
            peers = comps.assumptions.get("peer_tickers", [])
            h += f"<h3>Comps Peers</h3><p style='font-size:.85rem'>{', '.join(peers)}</p>"

        h += (
            "<h3>Methodology Notes</h3>"
            "<ul style='font-size:.85rem;color:var(--muted);padding-left:1.5rem'>"
            "<li>DCF uses unlevered Free Cash Flow to Firm (FCFF) discounted at WACC.</li>"
            "<li>Terminal value via Gordon Growth Model (perpetuity growth).</li>"
            "<li>EV → Equity bridge: Equity = EV − Total Debt + Cash.</li>"
            "<li>Comps apply median peer multiples to target financials.</li>"
            "<li>Outlier peers (>3σ from median) are excluded from comps aggregation.</li>"
            "<li>All financial data sourced from Yahoo Finance via yfinance.</li>"
            "</ul></div>"
        )
        self._parts.append(h)
        return self

    def add_historical_trends(self, snapshots: list[FundamentalSnapshot]) -> "ReportBuilder":
        """Add a multi-year financial trend table from historical snapshots."""
        if not snapshots:
            return self
        h = '<div class="card"><h2>Multi-Year Financial Trends</h2>'
        h += '<div style="overflow-x:auto"><table><thead><tr>'
        h += '<th>Metric</th>'
        for s in snapshots:
            h += f'<th>${s.total_revenue/1e6:,.0f}M rev</th>'
        h += '</tr></thead><tbody>'

        rows: list[tuple[str, list, bool]] = [
            ("Revenue", [s.total_revenue for s in snapshots], True),
            ("Gross Profit", [s.gross_profit for s in snapshots], True),
            ("EBIT", [s.operating_income for s in snapshots], True),
            ("EBITDA", [s.ebitda for s in snapshots], True),
            ("Net Income", [s.net_income for s in snapshots], True),
            ("Free Cash Flow", [s.free_cashflow for s in snapshots], True),
            ("Total Assets", [s.total_assets for s in snapshots], True),
            ("Total Debt", [s.total_debt for s in snapshots], True),
            ("Gross Margin", [s.gross_margin for s in snapshots], False),
            ("Operating Margin", [s.operating_margin for s in snapshots], False),
            ("Net Margin", [s.net_margin for s in snapshots], False),
            ("ROE", [s.roe for s in snapshots], False),
        ]
        for label, vals, is_currency in rows:
            h += f'<tr><td style="font-weight:600">{label}</td>'
            for v in vals:
                if v is None:
                    h += '<td>N/A</td>'
                elif is_currency:
                    h += f'<td>{_fmt_currency(v)}</td>'
                else:
                    cls = "trend-up" if (v or 0) > 0 else "trend-down" if (v or 0) < 0 else "trend-flat"
                    h += f'<td class="{cls}">{_fmt_pct(v)}</td>'
            h += '</tr>'
        h += '</tbody></table></div></div>'
        self._parts.append(h)
        return self

    # ── Render ──────────────────────────────────────────────────────

    def render(self, title: str = "Ainalyst Equity Research Report") -> str:
        """Return the complete HTML document as a string."""
        body = "\n".join(self._parts)
        return (
            "<!DOCTYPE html><html lang='en'><head>"
            f"<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"{_CSS}</head><body>{body}"
            f"<footer>Generated by <strong>ainalyst v0.1.0</strong> · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            f"</footer></body></html>"
        )

    def save(self, path: str | Path, title: str = "Ainalyst Equity Research Report") -> Path:
        """Render and write the report to *path*."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(title), encoding="utf-8")
        log.info("Report saved to %s", p)
        return p

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _kv_card(title: str, items: list[tuple[str, str]]) -> str:
        h = f'<div class="card"><h3>{html.escape(title)}</h3>'
        for k, v in items:
            h += f'<div class="kv"><span class="k">{html.escape(k)}</span><span class="v">{v}</span></div>'
        h += "</div>"
        return h

    @staticmethod
    def _df_to_table(df: pd.DataFrame, index_label: str | None = None) -> str:
        h = "<div style='overflow-x:auto'><table>"
        h += "<tr>"
        if index_label is not None or df.index.name:
            h += f"<th>{html.escape(index_label or df.index.name or '')}</th>"
        for col in df.columns:
            h += f"<th>{html.escape(str(col))}</th>"
        h += "</tr>"
        for idx, row in df.iterrows():
            h += "<tr>"
            if index_label is not None or df.index.name:
                h += f"<td style='font-weight:600'>{html.escape(str(idx))}</td>"
            for val in row:
                display = "N/A" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
                h += f"<td>{html.escape(display)}</td>"
            h += "</tr>"
        h += "</table></div>"
        return h


# ──────────────────────────────────────────────────────────────────────
# Convenience: one-shot full report
# ──────────────────────────────────────────────────────────────────────

def generate_full_report(
    snap: FundamentalSnapshot,
    dcf: DCFResult | None = None,
    comps: CompsResult | None = None,
    sector: SectorSummary | None = None,
    historical: list[FundamentalSnapshot] | None = None,
    output_path: str | Path = "report.html",
    output_csv: str | Path | None = None,
) -> Path:
    """Build and save a complete equity research report."""
    rb = ReportBuilder()
    rb.add_header(snap)
    rb.add_executive_summary(snap, dcf=dcf, comps=comps)
    rb.add_financial_snapshot(snap)
    rb.add_football_field(snap.current_price, dcf=dcf, comps=comps)
    if historical:
        rb.add_historical_trends(historical)
    if dcf:
        rb.add_dcf_detail(dcf)
    if comps:
        rb.add_comps_detail(comps)
    if sector:
        rb.add_sector_summary(sector)
    rb.add_assumptions(dcf=dcf, comps=comps)
    title = f"Ainalyst — {snap.company_name} ({snap.ticker})"

    # Write CSV if requested
    if output_csv:
        import csv
        from io import StringIO
        d = snap.to_dict()
        if dcf:
            d["dcf_intrinsic"] = dcf.intrinsic_per_share
            d["dcf_mos"] = dcf.margin_of_safety_pct
            d["dcf_signal"] = dcf.valuation_signal
        if comps:
            d["comps_composite"] = comps.composite_value
            d["comps_mos"] = comps.margin_of_safety_pct
            d["comps_signal"] = comps.valuation_signal
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(d.keys()))
        writer.writeheader()
        writer.writerow(d)
        csv_path.write_text(buf.getvalue(), encoding="utf-8")
        log.info("CSV saved to %s", csv_path)

    return rb.save(output_path, title=title)
