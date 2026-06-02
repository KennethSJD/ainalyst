"""ASX Company Directory — fetch the full list of listed companies + listing dates.

Source: ASX / Markit Digital research CSV endpoint. Returns every listed
company with its GICS industry group, listing date, and market cap.

The legacy ``asx.com.au/asx/1/company/directory`` JSON API was retired
(returns 404). The legacy ``ASXListedCompanies.csv`` still works but omits
listing dates, so it is unusable for IPO detection.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Markit Digital research CSV — carries Listing date + Market Cap.
# access_token loaded from AINALYST_MARKIT_TOKEN env var with fallback.
_MARKIT_TOKEN = os.environ.get(
    "AINALYST_MARKIT_TOKEN", "83ff96335c2d45a094df02a206a39ff4"
)
_DIRECTORY_URL = (
    "https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file"
    f"?access_token={_MARKIT_TOKEN}"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

_DEFAULT_CSV = Path(__file__).parent.parent / "asx_directory.csv"

# Canonical column names after normalisation
_COLS = ["ticker", "name", "industry", "listing_date", "market_cap"]


def fetch_asx_directory(timeout: int = 20) -> pd.DataFrame:
    """Fetch the full ASX company directory.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker`` (CODE.AX), ``name``, ``industry``,
        ``listing_date`` (tz-aware UTC datetime), ``market_cap`` (float).
        Rows with unparseable listing dates are dropped.
    """
    log.info("Fetching ASX company directory …")
    resp = requests.get(_DIRECTORY_URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    log.info("Directory fetched: %d raw records", len(df))

    # Normalise the upstream column names (defensive — they vary in casing).
    rename = {
        "ASX code": "code",
        "Company name": "name",
        "GICs industry group": "industry",
        "GICS industry group": "industry",
        "Listing date": "listing_date",
        "Market Cap": "market_cap",
    }
    df = df.rename(columns=rename)

    missing = {"code", "listing_date"} - set(df.columns)
    if missing:
        raise ValueError(f"Directory CSV missing expected columns: {missing}")

    df["ticker"] = df["code"].astype(str).str.strip() + ".AX"
    df["name"] = df.get("name", "").astype(str).str.strip()
    df["industry"] = df.get("industry", "Unknown").astype(str).str.strip()
    df["market_cap"] = pd.to_numeric(df.get("market_cap"), errors="coerce")
    df["listing_date"] = pd.to_datetime(
        df["listing_date"], format="%d/%m/%Y", errors="coerce", utc=True
    )

    before = len(df)
    df = df.dropna(subset=["listing_date"])
    log.info("Dropped %d rows with unparseable listing dates", before - len(df))

    return df[_COLS].reset_index(drop=True)


def save_directory(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Write the directory DataFrame to CSV."""
    p = Path(path) if path else _DEFAULT_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    log.info("Directory saved → %s (%d rows)", p, len(df))
    return p


def load_directory(
    path: Path | str | None = None,
    refresh: bool = False,
    max_age_days: int = 1,
) -> pd.DataFrame:
    """Load the directory from local CSV, fetching if stale or missing.

    Parameters
    ----------
    path : cached CSV location (default: ``asx_directory.csv``).
    refresh : force a fresh fetch regardless of cache age.
    max_age_days : refetch if the cached CSV is older than this.
    """
    p = Path(path) if path else _DEFAULT_CSV

    if not refresh and p.exists():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            p.stat().st_mtime, tz=timezone.utc
        )
        if age <= timedelta(days=max_age_days):
            df = pd.read_csv(p, parse_dates=["listing_date"])
            df["listing_date"] = pd.to_datetime(df["listing_date"], utc=True)
            log.info("Loaded cached directory (%d rows, age %s)", len(df), age)
            return df

    df = fetch_asx_directory()
    save_directory(df, p)
    return df


def recent_ipos_from_directory(
    df: pd.DataFrame | None = None,
    lookback_years: float = 2.0,
) -> pd.DataFrame:
    """Filter the directory to companies listed within *lookback_years*.

    Returns rows sorted by listing date (newest first).
    """
    if df is None:
        df = load_directory()
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(lookback_years * 365.25))
    recent = df[df["listing_date"] >= cutoff].copy()
    return recent.sort_values("listing_date", ascending=False).reset_index(drop=True)
