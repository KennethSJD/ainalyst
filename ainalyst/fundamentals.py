"""Fundamentals Parser — normalise financial statements into clean DataFrames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ainalyst.acquisition import ASXTicker

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Safe extraction helpers
# ──────────────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, label: str, col_idx: int = 0) -> float:
    """Extract a scalar from *df* by row label, returning 0.0 on miss."""
    if df.empty or label not in df.index:
        return 0.0
    try:
        val = df.loc[label].iloc[col_idx]
        return 0.0 if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
    except (IndexError, TypeError, ValueError):
        return 0.0


def _get_any(df: pd.DataFrame, labels: list[str], col_idx: int = 0) -> float:
    """Try multiple row *labels* in order, returning the first hit."""
    for lbl in labels:
        v = _get(df, lbl, col_idx)
        if v != 0.0:
            return v
    return 0.0


# ──────────────────────────────────────────────────────────────────────
# Snapshot dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FundamentalSnapshot:
    """Standardised point-in-time financial snapshot for a single company."""

    ticker: str
    company_name: str
    sector: str
    currency: str

    # Income statement
    total_revenue: float
    cost_of_revenue: float
    gross_profit: float
    operating_income: float  # EBIT
    ebitda: float
    net_income: float
    interest_expense: float
    tax_provision: float

    # Balance sheet
    total_assets: float
    total_liabilities: float
    total_debt: float
    cash_and_equivalents: float
    total_equity: float

    # Cash flow
    operating_cashflow: float
    capital_expenditure: float
    free_cashflow: float
    depreciation: float

    # Market data
    shares_outstanding: float
    current_price: float | None
    market_cap: float | None

    # Derived
    @property
    def enterprise_value(self) -> float:
        mc = self.market_cap or 0.0
        return mc + self.total_debt - self.cash_and_equivalents

    @property
    def pe_ratio(self) -> float | None:
        if self.net_income <= 0 or self.shares_outstanding <= 0:
            return None
        eps = self.net_income / self.shares_outstanding
        if eps <= 0 or self.current_price is None:
            return None
        return self.current_price / eps

    @property
    def ev_ebitda(self) -> float | None:
        if self.ebitda <= 0:
            return None
        return self.enterprise_value / self.ebitda

    @property
    def ev_sales(self) -> float | None:
        if self.total_revenue <= 0:
            return None
        return self.enterprise_value / self.total_revenue

    @property
    def gross_margin(self) -> float | None:
        if self.total_revenue <= 0:
            return None
        return self.gross_profit / self.total_revenue

    @property
    def operating_margin(self) -> float | None:
        if self.total_revenue <= 0:
            return None
        return self.operating_income / self.total_revenue

    @property
    def net_margin(self) -> float | None:
        if self.total_revenue <= 0:
            return None
        return self.net_income / self.total_revenue

    @property
    def roe(self) -> float | None:
        if self.total_equity <= 0:
            return None
        return self.net_income / self.total_equity

    def to_dict(self) -> dict[str, Any]:
        """Flat dict including derived properties."""
        base = {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "currency": self.currency,
            "total_revenue": self.total_revenue,
            "cost_of_revenue": self.cost_of_revenue,
            "gross_profit": self.gross_profit,
            "operating_income": self.operating_income,
            "ebitda": self.ebitda,
            "net_income": self.net_income,
            "interest_expense": self.interest_expense,
            "tax_provision": self.tax_provision,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "total_debt": self.total_debt,
            "cash_and_equivalents": self.cash_and_equivalents,
            "total_equity": self.total_equity,
            "operating_cashflow": self.operating_cashflow,
            "capital_expenditure": self.capital_expenditure,
            "free_cashflow": self.free_cashflow,
            "depreciation": self.depreciation,
            "shares_outstanding": self.shares_outstanding,
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "pe_ratio": self.pe_ratio,
            "ev_ebitda": self.ev_ebitda,
            "ev_sales": self.ev_sales,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "roe": self.roe,
        }
        return base


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────

_REVENUE_LABELS = ["Total Revenue", "TotalRevenue", "Revenue"]
_COGS_LABELS = ["Cost Of Revenue", "CostOfRevenue"]
_GROSS_LABELS = ["Gross Profit", "GrossProfit"]
_EBIT_LABELS = ["Operating Income", "OperatingIncome", "EBIT"]
_EBITDA_LABELS = ["EBITDA", "Normalized EBITDA", "NormalizedEBITDA"]
_NET_INCOME_LABELS = ["Net Income", "NetIncome", "Net Income Common Stockholders"]
_INTEREST_LABELS = ["Interest Expense", "InterestExpense"]
_TAX_LABELS = ["Tax Provision", "TaxProvision", "Income Tax Expense"]

_TOTAL_ASSETS_LABELS = ["Total Assets", "TotalAssets"]
_TOTAL_LIAB_LABELS = ["Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest", "Total Liab"]
_TOTAL_DEBT_LABELS = ["Total Debt", "TotalDebt", "Net Debt"]
_CASH_LABELS = ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"]
_EQUITY_LABELS = ["Total Equity Gross Minority Interest", "TotalEquityGrossMinorityInterest", "Stockholders Equity", "StockholdersEquity"]

_OCF_LABELS = ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"]
_CAPEX_LABELS = ["Capital Expenditure", "CapitalExpenditure"]
_FCF_LABELS = ["Free Cash Flow", "FreeCashFlow"]
_DEPR_LABELS = ["Depreciation And Amortization", "DepreciationAndAmortization", "Depreciation"]


def build_snapshot(ticker: ASXTicker) -> FundamentalSnapshot:
    """Parse the latest annual financials from *ticker* into a :class:`FundamentalSnapshot`."""
    inc = ticker.income_stmt
    bs = ticker.balance_sheet
    cf = ticker.cashflow

    revenue = _get_any(inc, _REVENUE_LABELS)
    cogs = _get_any(inc, _COGS_LABELS)
    gross = _get_any(inc, _GROSS_LABELS)
    if gross == 0.0 and revenue > 0:
        gross = revenue - abs(cogs)

    ebit = _get_any(inc, _EBIT_LABELS)
    ebitda = _get_any(inc, _EBITDA_LABELS)
    depr = _get_any(cf, _DEPR_LABELS)
    if ebitda == 0.0 and ebit != 0.0:
        ebitda = ebit + abs(depr)

    net_inc = _get_any(inc, _NET_INCOME_LABELS)
    interest = _get_any(inc, _INTEREST_LABELS)
    tax = _get_any(inc, _TAX_LABELS)

    total_assets = _get_any(bs, _TOTAL_ASSETS_LABELS)
    total_liab = _get_any(bs, _TOTAL_LIAB_LABELS)
    total_debt = _get_any(bs, _TOTAL_DEBT_LABELS)
    cash = _get_any(bs, _CASH_LABELS)
    equity = _get_any(bs, _EQUITY_LABELS)
    if equity == 0.0 and total_assets > 0:
        equity = total_assets - total_liab

    ocf = _get_any(cf, _OCF_LABELS)
    capex = _get_any(cf, _CAPEX_LABELS)
    fcf = _get_any(cf, _FCF_LABELS)
    if fcf == 0.0 and ocf != 0.0:
        fcf = ocf - abs(capex)

    shares = ticker.shares_outstanding or 0.0
    price = ticker.current_price
    mcap = ticker.market_cap

    return FundamentalSnapshot(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        sector=ticker.sector,
        currency=ticker.info.get("currency", "AUD"),
        total_revenue=revenue,
        cost_of_revenue=cogs,
        gross_profit=gross,
        operating_income=ebit,
        ebitda=ebitda,
        net_income=net_inc,
        interest_expense=interest,
        tax_provision=tax,
        total_assets=total_assets,
        total_liabilities=total_liab,
        total_debt=total_debt,
        cash_and_equivalents=cash,
        total_equity=equity,
        operating_cashflow=ocf,
        capital_expenditure=capex,
        free_cashflow=fcf,
        depreciation=depr,
        shares_outstanding=shares,
        current_price=price,
        market_cap=mcap,
    )


def build_snapshots(tickers: list[ASXTicker]) -> list[FundamentalSnapshot]:
    """Build snapshots for multiple tickers, skipping failures."""
    results: list[FundamentalSnapshot] = []
    for t in tickers:
        try:
            results.append(build_snapshot(t))
        except Exception as exc:
            log.warning("Skipping %s: %s", t.symbol, exc)
    return results


def build_historical_snapshots(ticker: ASXTicker) -> list[FundamentalSnapshot]:
    """Build a time-series of snapshots from all available annual columns.

    Parses every column in the income statement, balance sheet, and cash flow
    statement, producing one :class:`FundamentalSnapshot` per reporting period.
    Returns list sorted newest-first.
    """
    inc = ticker.income_stmt
    bs = ticker.balance_sheet
    cf = ticker.cashflow

    if inc.empty:
        log.warning("No income statement data for %s", ticker.symbol)
        return []

    snapshots: list[FundamentalSnapshot] = []
    for col_idx in range(len(inc.columns)):
        try:
            snap = _build_snapshot_at_col(ticker, inc, bs, cf, col_idx)
            if snap.total_revenue > 0:
                snapshots.append(snap)
        except Exception as exc:
            log.debug("Skipping column %d for %s: %s", col_idx, ticker.symbol, exc)

    # Sort by revenue descending (proxy for newest-first since columns are dated)
    snapshots.sort(key=lambda s: -s.total_revenue)
    return snapshots


def _build_snapshot_at_col(
    ticker: ASXTicker,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    col_idx: int,
) -> FundamentalSnapshot:
    """Build a snapshot for a specific column index."""
    revenue = _get_any(inc, _REVENUE_LABELS, col_idx)
    cogs = _get_any(inc, _COGS_LABELS, col_idx)
    gross = _get_any(inc, _GROSS_LABELS, col_idx)
    if gross == 0.0 and revenue > 0:
        gross = revenue - abs(cogs)

    ebit = _get_any(inc, _EBIT_LABELS, col_idx)
    ebitda = _get_any(inc, _EBITDA_LABELS, col_idx)
    depr = _get_any(cf, _DEPR_LABELS, col_idx)
    if ebitda == 0.0 and ebit != 0.0:
        ebitda = ebit + abs(depr)

    net_inc = _get_any(inc, _NET_INCOME_LABELS, col_idx)
    interest = _get_any(inc, _INTEREST_LABELS, col_idx)
    tax = _get_any(inc, _TAX_LABELS, col_idx)

    total_assets = _get_any(bs, _TOTAL_ASSETS_LABELS, col_idx)
    total_liab = _get_any(bs, _TOTAL_LIAB_LABELS, col_idx)
    total_debt = _get_any(bs, _TOTAL_DEBT_LABELS, col_idx)
    cash = _get_any(bs, _CASH_LABELS, col_idx)
    equity = _get_any(bs, _EQUITY_LABELS, col_idx)
    if equity == 0.0 and total_assets > 0:
        equity = total_assets - total_liab

    ocf = _get_any(cf, _OCF_LABELS, col_idx)
    capex = _get_any(cf, _CAPEX_LABELS, col_idx)
    fcf = _get_any(cf, _FCF_LABELS, col_idx)
    if fcf == 0.0 and ocf != 0.0:
        fcf = ocf - abs(capex)

    shares = ticker.shares_outstanding or 0.0
    price = ticker.current_price
    mcap = ticker.market_cap

    return FundamentalSnapshot(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        sector=ticker.sector,
        currency=ticker.info.get("currency", "AUD"),
        total_revenue=revenue,
        cost_of_revenue=cogs,
        gross_profit=gross,
        operating_income=ebit,
        ebitda=ebitda,
        net_income=net_inc,
        interest_expense=interest,
        tax_provision=tax,
        total_assets=total_assets,
        total_liabilities=total_liab,
        total_debt=total_debt,
        cash_and_equivalents=cash,
        total_equity=equity,
        operating_cashflow=ocf,
        capital_expenditure=capex,
        free_cashflow=fcf,
        depreciation=depr,
        shares_outstanding=shares,
        current_price=price,
        market_cap=mcap,
    )
