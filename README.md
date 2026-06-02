# ainalyst

ASX company valuation engine — **DCF**, **Trading Comps**, and **Sector Analysis** with interactive HTML reports.

## Architecture

```
ainalyst/
├── config.py           # Central constants, GICS sectors, default assumptions
├── acquisition.py      # Data Acquisition Engine (yfinance, auto .AX suffix)
├── fundamentals.py     # Fundamentals Parser → standardised FundamentalSnapshot
├── valuation/
│   ├── bridge.py       # EV ↔ Equity Value bridge math
│   ├── dcf.py          # DCF model with sensitivity matrix
│   └── comps.py        # Trading Comps (peer multiples → implied value)
├── sector.py           # Aggregate Sector Analyzer
├── report.py           # HTML Reporting Engine (standalone, no external deps)
└── cli.py              # CLI entry point
```

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Full analysis with HTML report
ainalyst analyse CSL
ainalyst analyse BHP --wacc 0.09 --peers RIO FMG S32

# Sector overview
ainalyst sector "Materials" -o sector_report.html
```

### Python API

```python
from ainalyst.acquisition import ASXTicker
from ainalyst.fundamentals import build_snapshot
from ainalyst.valuation import DCFModel, CompsModel
from ainalyst.report import generate_full_report

ticker = ASXTicker("CSL")
snap = build_snapshot(ticker)
dcf = DCFModel().run(snap)
comps = CompsModel().run(snap)
generate_full_report(snap, dcf, comps, output_path="CSL_report.html")
```

## Valuation Methodology

**DCF (Discounted Cash Flow)**
- Projects unlevered FCFF over N years using configurable growth rates and margins
- Terminal value via Gordon Growth Model
- WACC × Terminal Growth Rate sensitivity matrix
- Bridge: `Equity = EV − Debt + Cash`

**Trading Comps**
- Fetches peer financials from the same GICS sector
- Computes median P/E, EV/EBITDA, EV/Sales (3σ outlier filtering)
- Applies median multiples to target → implied per-share value
- Composite = average of available implied values

## Testing

```bash
pytest tests/ -v
```

## License

MIT
