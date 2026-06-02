"""Tests for the IPO tracker module."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ainalyst.ipo import (
    IPOEntry,
    IPOReport,
    _compute_returns,
    _watchlist_to_entries,
    build_ipo_report,
    generate_ipo_report,
    load_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    detect_explicit_tickers,
)


# ──────────────────────────────────────────────────────────────────────
# IPOEntry unit tests
# ──────────────────────────────────────────────────────────────────────


def _make_entry(**overrides) -> IPOEntry:
    defaults = dict(
        ticker="TST",
        company_name="Test Corp",
        sector="Materials",
        listing_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        ipo_price=2.00,
        current_price=3.00,
        market_cap=300_000_000,
        shares_outstanding=100_000_000,
    )
    defaults.update(overrides)
    return IPOEntry(**defaults)


class TestIPOEntry:
    def test_total_return_positive(self):
        e = _make_entry(ipo_price=2.00, current_price=3.00)
        assert e.total_return == pytest.approx(0.5)

    def test_total_return_negative(self):
        e = _make_entry(ipo_price=2.00, current_price=1.00)
        assert e.total_return == pytest.approx(-0.5)

    def test_total_return_with_dividends(self):
        e = _make_entry(ipo_price=2.00, current_price=3.00, dividend_return=0.10)
        assert e.total_return == pytest.approx(0.6)  # 0.5 price + 0.10 div

    def test_total_return_none_when_missing(self):
        e = _make_entry(ipo_price=None, current_price=3.00)
        assert e.total_return is None

    def test_total_return_none_when_zero_ipo(self):
        e = _make_entry(ipo_price=0.0, current_price=3.00)
        assert e.total_return is None

    def test_days_since_listing(self):
        e = _make_entry()
        assert e.days_since_listing > 0

    def test_to_dict_has_expected_keys(self):
        e = _make_entry(returns={"1w": 0.05, "1m": 0.10})
        d = e.to_dict()
        assert d["ticker"] == "TST"
        assert d["return_1w"] == 0.05
        assert d["return_1m"] == 0.10
        assert "total_return" in d
        assert "days_since_listing" in d
        assert "dividend_return" in d


# ──────────────────────────────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────────────────────────────


class TestComputeReturns:
    def test_basic_returns(self):
        dates = pd.date_range("2024-01-01", periods=600, freq="B")
        prices = [1.0 + (i / 600) for i in range(600)]
        hist = pd.DataFrame({"Close": prices}, index=dates)

        returns = _compute_returns(hist, ipo_price=1.0)
        assert "1d" in returns
        assert "1w" in returns
        assert "1m" in returns
        assert returns["1d"] is not None
        assert returns["1d"] > 0

    def test_empty_hist(self):
        assert _compute_returns(pd.DataFrame(), 1.0) == {}

    def test_zero_ipo_price(self):
        assert _compute_returns(pd.DataFrame({"Close": [1.0]}), 0.0) == {}


# ──────────────────────────────────────────────────────────────────────
# Watchlist tests
# ──────────────────────────────────────────────────────────────────────


class TestWatchlist:
    def test_watchlist_to_entries(self):
        raw = [
            {
                "ticker": "ABC",
                "company_name": "ABC Holdings",
                "sector": "Financials",
                "listing_date": "2026-07-15",
                "ipo_price": 1.50,
                "status": "upcoming",
                "notes": "Raising $30M",
            }
        ]
        entries = _watchlist_to_entries(raw)
        assert len(entries) == 1
        assert entries[0].ticker == "ABC"
        assert entries[0].status == "upcoming"
        assert entries[0].ipo_price == 1.50
        assert entries[0].listing_date.year == 2026

    def test_watchlist_with_expected_date(self):
        raw = [{"company_name": "Mystery Co", "expected_date": "2026-09-01", "status": "rumoured"}]
        entries = _watchlist_to_entries(raw)
        assert len(entries) == 1
        assert entries[0].status == "rumoured"

    def test_load_watchlist_missing_file(self, tmp_path):
        result = load_watchlist(tmp_path / "nonexistent.yaml")
        assert result == []

    def test_load_watchlist_empty_ipos_key(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("ipos:\n")
        assert load_watchlist(p) == []

    def test_load_watchlist_valid(self, tmp_path):
        p = tmp_path / "test_watchlist.yaml"
        p.write_text(
            "ipos:\n"
            "  - ticker: XYZ\n"
            "    company_name: XYZ Ltd\n"
            "    sector: Energy\n"
            "    listing_date: '2026-08-01'\n"
            "    status: upcoming\n"
        )
        result = load_watchlist(p)
        assert len(result) == 1
        assert result[0]["ticker"] == "XYZ"


class TestWatchlistManagement:
    def test_add_new_entry(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("ipos: []\n")
        add_to_watchlist(
            {"ticker": "NEW", "company_name": "NewCo", "status": "upcoming"},
            path=p,
        )
        loaded = load_watchlist(p)
        assert len(loaded) == 1
        assert loaded[0]["ticker"] == "NEW"

    def test_add_updates_existing(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("ipos:\n  - ticker: DUP\n    company_name: Old Name\n    status: upcoming\n")
        add_to_watchlist(
            {"ticker": "DUP", "company_name": "New Name", "status": "listed"},
            path=p,
        )
        loaded = load_watchlist(p)
        assert len(loaded) == 1
        assert loaded[0]["company_name"] == "New Name"
        assert loaded[0]["status"] == "listed"

    def test_remove_existing(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("ipos:\n  - ticker: REM\n    company_name: RemoveMe\n    status: upcoming\n")
        assert remove_from_watchlist("REM", path=p) is True
        assert load_watchlist(p) == []

    def test_remove_nonexistent(self, tmp_path):
        p = tmp_path / "wl.yaml"
        p.write_text("ipos: []\n")
        assert remove_from_watchlist("NOPE", path=p) is False


# ──────────────────────────────────────────────────────────────────────
# Detection tests (mocked yfinance + directory)
# ──────────────────────────────────────────────────────────────────────


class TestDetectExplicitTickers:
    @patch("ainalyst.ipo.load_directory")
    @patch("ainalyst.ipo.time.sleep", return_value=None)
    @patch("ainalyst.ipo.yf.Ticker")
    def test_detects_recent_from_directory(self, mock_ticker_cls, mock_sleep, mock_load):
        recent_date = datetime.now(timezone.utc) - timedelta(days=100)
        mock_load.return_value = pd.DataFrame({
            "ticker": ["NEW.AX"],
            "name": ["New Co"],
            "industry": ["Materials"],
            "listing_date": [pd.Timestamp(recent_date)],
            "market_cap": [500_000_000],
        })

        mock_yft = MagicMock()
        mock_yft.history.return_value = pd.DataFrame(
            {"Close": [2.0, 2.5, 3.0]},
            index=pd.date_range(recent_date, periods=3, freq="B"),
        )
        mock_yft.dividends = pd.Series()
        mock_ticker_cls.return_value = mock_yft

        entries = detect_explicit_tickers(["NEW"], lookback_years=2.0)
        assert len(entries) == 1
        assert entries[0].ticker == "NEW"
        assert entries[0].ipo_price == pytest.approx(2.0)
        assert entries[0].current_price == 3.0

    @patch("ainalyst.ipo.load_directory")
    def test_skips_old_listing(self, mock_load):
        old_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        mock_load.return_value = pd.DataFrame({
            "ticker": ["OLD.AX"],
            "name": ["Old Co"],
            "industry": ["Materials"],
            "listing_date": [pd.Timestamp(old_date)],
            "market_cap": [100_000_000],
        })

        entries = detect_explicit_tickers(["OLD"], lookback_years=2.0)
        assert len(entries) == 0


class TestIPOReport:
    def test_report_aggregates(self):
        entries = [
            _make_entry(ticker="A", ipo_price=1.0, current_price=2.0),
            _make_entry(ticker="B", ipo_price=1.0, current_price=0.5),
            _make_entry(ticker="C", ipo_price=1.0, current_price=1.5),
        ]
        report = IPOReport(
            entries=entries,
            scan_date=datetime.now(timezone.utc),
            lookback_years=2.0,
            total_detected=3,
        )
        listed = report.listed_entries()
        assert len(listed) == 3
        assert report.to_dataframe().shape[0] == 3

    def test_to_csv_produces_output(self):
        entries = [
            _make_entry(ticker="A", ipo_price=1.0, current_price=2.0),
        ]
        report = IPOReport(
            entries=entries,
            scan_date=datetime.now(timezone.utc),
            lookback_years=2.0,
            total_detected=1,
        )
        csv_out = report.to_csv()
        assert "ticker" in csv_out
        assert "A" in csv_out

    def test_upcoming_entries_filtered(self):
        entries = [
            _make_entry(ticker="A", status="listed"),
            _make_entry(ticker="B", status="upcoming"),
            _make_entry(ticker="C", status="rumoured"),
        ]
        report = IPOReport(
            entries=entries,
            scan_date=datetime.now(timezone.utc),
            lookback_years=2.0,
            total_detected=1,
        )
        upcoming = report.upcoming_entries()
        assert len(upcoming) == 2
        assert all(e.status in ("upcoming", "rumoured") for e in upcoming)


class TestGenerateIPOReport:
    def test_generates_html(self, tmp_path):
        entries = [
            _make_entry(ticker="A", ipo_price=1.0, current_price=2.0),
            _make_entry(ticker="B", ipo_price=1.0, current_price=0.5, status="upcoming"),
        ]
        report = IPOReport(
            entries=entries,
            scan_date=datetime.now(timezone.utc),
            lookback_years=2.0,
            total_detected=1,
            median_total_return=1.0,
            mean_total_return=1.0,
            pct_positive=1.0,
            best_performer=entries[0],
            worst_performer=entries[0],
        )
        out = tmp_path / "test_ipo.html"
        path = generate_ipo_report(report, output_path=out)
        assert path.exists()
        content = path.read_text()
        assert "ASX IPO Tracker" in content
        assert "Ticker" in content

    def test_generates_csv_sidecar(self, tmp_path):
        entries = [
            _make_entry(ticker="A", ipo_price=1.0, current_price=2.0),
        ]
        report = IPOReport(
            entries=entries,
            scan_date=datetime.now(timezone.utc),
            lookback_years=2.0,
            total_detected=1,
        )
        html_out = tmp_path / "test.html"
        csv_out = tmp_path / "test.csv"
        generate_ipo_report(report, output_path=html_out, output_csv=csv_out)
        assert csv_out.exists()
        content = csv_out.read_text()
        assert "ticker" in content
        assert "A" in content
