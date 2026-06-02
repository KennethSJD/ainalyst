# ainalyst

ASX company valuation engine — **DCF**, **Trading Comps**, **Sector Analysis**, and an **IPO Tracker** with interactive HTML reports.

## Architecture

```
ainalyst/
├── config.py           # Central constants, GICS sectors, default assumptions
├── acquisition.py      # Data Acquisition Engine (yfinance, auto .AX suffix)
├── directory.py        # ASX company directory fetch (full universe + listing dates)
├── fundamentals.py     # Fundamentals Parser → standardised FundamentalSnapshot
├── ipo.py              # IPO Tracker (recent IPO detection + performance + watchlist)
├── valuation/
│   ├── bridge.py       # EV ↔ Equity Value bridge math
│   ├── dcf.py          # DCF model with sensitivity matrix
│   └── comps.py        # Trading Comps (peer multiples → implied value, market-cap filtering)
├── sector.py           # Aggregate Sector Analyzer (hardcoded peers or full directory)
├── report.py           # HTML + CSV Reporting Engine (standalone, mobile-responsive)
└── cli.py              # CLI entry point

.env                    # Secrets and config (not committed — see .env.example)
asx_scraper.py          # Standalone script → asx_directory.csv
ipo_watchlist.yaml      # User-maintained watchlist of upcoming/rumoured IPOs
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set up environment
cp .env .env.local          # edit if API token rotates

# Fetch the ASX directory (cached 1 day)
python asx_scraper.py

# Run a full valuation
ainalyst analyse CSL
```

## Configuration

Secrets and rate limits live in `.env` (gitignored):

```
AINALYST_MARKIT_TOKEN=83ff96335c2d45a094df02a206a39ff4   # ASX directory API token
AINALYST_RATE_LIMIT=0.15                                  # seconds between yfinance calls
```

## Usage

### CLI — Single-Company Valuation

```bash
# Full DCF + Comps + Sector with HTML report
ainalyst analyse CSL
ainalyst analyse BHP --wacc 0.09 --peers RIO FMG S32 MIN

# Customise assumptions
ainalyst analyse WOW --wacc 0.08 --tgr 0.03 --margin 0.12

# Skip valuation models
ainalyst analyse CSL --no-dcf
ainalyst analyse BHP --no-comps

# Market-cap-filtered peers (±50% default, configurable)
ainalyst analyse CSL --mc-filter 0.5

# Multi-year financial trend table in report
ainalyst analyse CSL --history

# Export CSV alongside HTML
ainalyst analyse CSL --csv CSL_data.csv

# Override output path
ainalyst analyse CSL -o reports/my_report.html
```

### CLI — Sector Overview

```bash
# Basic sector (uses built-in peer list)
ainalyst sector "Materials"

# Full directory-based sector analysis (all ASX companies in that industry)
ainalyst sector "Materials" --use-directory

# Arbitrary GICS industry group from the directory
ainalyst sector "Pharmaceuticals" --use-directory

# With HTML output
ainalyst sector "Financials" -o sector_report.html
```

### CLI — IPO Tracker

```bash
# All recent IPOs from the full ASX directory (last 2 years)
ainalyst ipo

# Quick run — cap number of IPOs enriched
ainalyst ipo --max 20

# Custom lookback window
ainalyst ipo --years 1.5

# Force fresh directory fetch (bypass 1-day cache)
ainalyst ipo --refresh

# Check specific tickers for recent IPO status
ainalyst ipo BHP CSL XRO

# Export CSV alongside HTML report
ainalyst ipo --csv ipo_data.csv

# Custom output path
ainalyst ipo -o reports/ipo_tracker.html --csv reports/ipo_tracker.csv
```

### CLI — Watchlist Management

```bash
# Add an upcoming IPO to the watchlist
ainalyst ipo-add --ticker ABC --name "ABC Holdings" --sector Financials \
    --date 2026-07-15 --price 1.50 --status upcoming \
    --notes "Raising $30M. Lead manager: Macquarie."

# Add a rumoured float
ainalyst ipo-add --ticker TBD --name "Rumoured Mining Co" --sector Materials \
    --date 2026-09-01 --status rumoured

# Update an existing entry (same ticker overwrites)
ainalyst ipo-add --ticker ABC --name "ABC Holdings Ltd" --status listed

# Remove an entry
ainalyst ipo-remove ABC

# Use a custom watchlist file
ainalyst ipo-add --ticker XYZ --name "XYZ" --watchlist my_private_list.yaml
ainalyst ipo --watchlist my_private_list.yaml
```

### Fetching the ASX Directory

```bash
python asx_scraper.py [output.csv]
```

The directory is fetched from the ASX / Markit Digital research CSV endpoint
and cached locally as `asx_directory.csv` (1-day TTL). The IPO tracker
refreshes automatically when stale. Pass `--refresh` to force a fresh fetch.

### Python API

```python
from ainalyst.acquisition import ASXTicker
from ainalyst.fundamentals import build_snapshot, build_historical_snapshots
from ainalyst.valuation import DCFModel, CompsModel
from ainalyst.report import generate_full_report

# Single company analysis
ticker = ASXTicker("CSL")
snap = build_snapshot(ticker)

dcf = DCFModel().run(snap)
comps = CompsModel().run(snap, market_cap_filter=0.5)

generate_full_report(snap, dcf, comps, output_path="CSL_report.html")

# With multi-year trends
historical = build_historical_snapshots(ticker)
generate_full_report(snap, dcf, comps, historical=historical,
                     output_path="CSL_report.html", output_csv="CSL_data.csv")
```

```python
# IPO tracker
from ainalyst.ipo import build_ipo_report, generate_ipo_report

report = build_ipo_report(lookback_years=2.0)
generate_ipo_report(report, output_path="ipo_report.html",
                    output_csv="ipo_report.csv")

# Access the raw data
df = report.to_dataframe()
csv_str = report.to_csv()
```

```python
# Watchlist management
from ainalyst.ipo import add_to_watchlist, remove_from_watchlist, load_watchlist

add_to_watchlist({
    "ticker": "ABC",
    "company_name": "ABC Holdings",
    "sector": "Financials",
    "expected_date": "2026-07-15",
    "ipo_price": 1.50,
    "status": "upcoming",
    "notes": "Raising $30M",
})

remove_from_watchlist("ABC")
entries = load_watchlist()
```

```python
# ASX directory
from ainalyst.directory import load_directory, recent_ipos_from_directory

df = load_directory()                              # full ASX universe (cached)
recent = recent_ipos_from_directory(df, lookback_years=2.0)
df = load_directory(refresh=True)                  # force fresh fetch
```

```python
# Sector analysis with full directory
from ainalyst.sector import SectorAnalyzer

analyzer = SectorAnalyzer()
summary = analyzer.analyse_sector("Materials", use_directory=True)
print(f"{summary.sector}: {summary.count} companies, Median P/E: {summary.median_pe}")
```

## Valuation Methodology

**DCF (Discounted Cash Flow)**
- Projects unlevered FCFF over N years using configurable growth rates and margins
- Uses actual Depreciation & Amortisation from financial statements (not a crude ratio)
- Terminal value via Gordon Growth Model
- WACC × Terminal Growth Rate sensitivity matrix (5×5 default)
- Bridge: `Equity = EV − Total Debt + Cash & Equivalents`
- Signals: Undervalued (>15% upside), Fairly Valued (±15%), Overvalued (>15% downside)

**Trading Comps**
- Fetches peer financials from the same GICS sector (10 default per sector)
- Optional market-cap filtering: `market_cap_filter=0.5` keeps peers within ±50% of target
- Computes median P/E, EV/EBITDA, EV/Sales with 3σ outlier filtering
- Each multiple → implied per-share value via the EV/Equity bridge
- Composite = equally weighted average of available implied values

**IPO Tracker**
- Detects recent IPOs from the full ASX directory (~1,800 listed companies) using authoritative listing dates
- Fetches price history via yfinance for post-listing performance across standard windows (1d, 1w, 1m, 3m, 6m, 1y, 2y)
- Total return includes dividends: `(current / ipo) - 1 + cumulative_dividends / ipo_price`
- Rate-limited yfinance calls (`AINALYST_RATE_LIMIT`, default 0.15s between requests)
- Aggregates median/mean return, % positive, best/worst performers
- Merges `ipo_watchlist.yaml` for upcoming & rumoured floats (no market data yet)
- HTML report includes embedded CSV download, mobile-responsive layout

**Sector Analyzer**
- Built-in peer lists for 12 GICS sectors via `SECTOR_PEERS`
- `use_directory=True` queries the full ASX directory for arbitrary industry groups
- Computes median multiples (P/E, EV/EBITDA, EV/Sales) and margins (Gross, Operating, Net, ROE)

## Report Features

- Standalone HTML — no external CSS/JS dependencies
- Mobile-responsive (`@media max-width: 768px`) and print-friendly styles
- Football-field valuation range chart (DCF + Comps vs current price)
- DCF sensitivity matrix and projection detail tables
- Multi-year financial trend table (`--history` flag)
- Colour-coded valuation signals (green/yellow/red badges)
- Embedded CSV export in IPO reports
- Sidecar CSV output via `--csv` flag on all report commands

## Testing

```bash
pytest tests/ -v                 # all tests (72)
pytest tests/test_dcf.py -v      # DCF model
pytest tests/test_comps.py -v    # comps + market-cap filtering
pytest tests/test_ipo.py -v      # IPO tracker + watchlist CRUD
pytest tests/test_sector.py -v   # sector + directory fallback
pytest tests/test_directory.py -v # directory fetch + cache
```

## License

MIT
