"""Data Acquisition Engine — Yahoo Finance pricing & ASX metadata."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from ainalyst.config import normalise_ticker

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Core ticker wrapper
# ──────────────────────────────────────────────────────────────────────

class ASXTicker:
    """Thin wrapper around ``yfinance.Ticker`` with automatic .AX suffixing."""

    def __init__(self, symbol: str) -> None:
        self.raw_symbol = symbol
        self.symbol = normalise_ticker(symbol)
        self._yf = yf.Ticker(self.symbol)
        self._info: dict[str, Any] | None = None

    # -- lazy info cache --------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            try:
                self._info = self._yf.info or {}
            except Exception:
                log.warning("Failed to fetch info for %s", self.symbol)
                self._info = {}
        return self._info

    # -- pricing ----------------------------------------------------------

    def history(
        self,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV history for the ticker."""
        try:
            df = self._yf.history(period=period, interval=interval)
            if df.empty:
                log.warning("Empty price history for %s", self.symbol)
            return df
        except Exception as exc:
            log.error("Price history error for %s: %s", self.symbol, exc)
            return pd.DataFrame()

    @property
    def current_price(self) -> float | None:
        """Best-effort current / last-close price."""
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            val = self.info.get(key)
            if val is not None:
                return float(val)
        # fallback: last row of 5d history
        hist = self.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None

    # -- financial statements --------------------------------------------

    @property
    def income_stmt(self) -> pd.DataFrame:
        return _safe_df(self._yf.income_stmt)

    @property
    def quarterly_income_stmt(self) -> pd.DataFrame:
        return _safe_df(self._yf.quarterly_income_stmt)

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return _safe_df(self._yf.balance_sheet)

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        return _safe_df(self._yf.quarterly_balance_sheet)

    @property
    def cashflow(self) -> pd.DataFrame:
        return _safe_df(self._yf.cashflow)

    @property
    def quarterly_cashflow(self) -> pd.DataFrame:
        return _safe_df(self._yf.quarterly_cashflow)

    # -- convenience ------------------------------------------------------

    @property
    def sector(self) -> str:
        return self.info.get("sector", "Unknown")

    @property
    def company_name(self) -> str:
        return self.info.get("shortName") or self.info.get("longName") or self.symbol

    @property
    def market_cap(self) -> float | None:
        mc = self.info.get("marketCap")
        return float(mc) if mc is not None else None

    @property
    def shares_outstanding(self) -> float | None:
        for key in ("sharesOutstanding", "impliedSharesOutstanding"):
            val = self.info.get(key)
            if val is not None:
                return float(val)
        return None

    def __repr__(self) -> str:
        return f"ASXTicker({self.symbol!r})"


# ──────────────────────────────────────────────────────────────────────
# Batch fetch
# ──────────────────────────────────────────────────────────────────────

def fetch_tickers(symbols: list[str]) -> list[ASXTicker]:
    """Return a list of :class:`ASXTicker` for each symbol (auto-suffixed)."""
    return [ASXTicker(s) for s in symbols]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return *df* or an empty DataFrame on ``None`` / exception."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    return df
