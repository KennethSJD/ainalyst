"""Tests for the ASX directory module."""

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from ainalyst.directory import (
    fetch_asx_directory,
    save_directory,
    load_directory,
    recent_ipos_from_directory,
    _MARKIT_TOKEN,
)


class TestDirectoryToken:
    def test_token_from_env(self, monkeypatch):
        monkeypatch.setenv("AINALYST_MARKIT_TOKEN", "test-token-123")
        # Re-import to pick up env var
        import importlib
        import ainalyst.directory
        importlib.reload(ainalyst.directory)
        assert ainalyst.directory._MARKIT_TOKEN == "test-token-123"

    def test_token_fallback(self):
        # Without env var, uses built-in default
        assert len(_MARKIT_TOKEN) > 10


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        df = pd.DataFrame({
            "ticker": ["ABC.AX", "XYZ.AX"],
            "name": ["ABC Corp", "XYZ Ltd"],
            "industry": ["Materials", "Tech"],
            "listing_date": [
                pd.Timestamp("2020-01-15", tz="UTC"),
                pd.Timestamp("2021-06-01", tz="UTC"),
            ],
            "market_cap": [1e9, 500e6],
        })
        p = save_directory(df, tmp_path / "test_dir.csv")
        loaded = pd.read_csv(p, parse_dates=["listing_date"])
        assert len(loaded) == 2
        assert loaded["ticker"].iloc[0] == "ABC.AX"

    def test_load_refresh(self, tmp_path, monkeypatch):
        # Create stale cached file
        p = tmp_path / "stale.csv"
        df = pd.DataFrame({
            "ticker": ["OLD.AX"],
            "name": ["Old"],
            "industry": ["Old"],
            "listing_date": [pd.Timestamp("2020-01-01")],
            "market_cap": [1e6],
        })
        df.to_csv(p, index=False)

        with patch("ainalyst.directory.fetch_asx_directory") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                "ticker": ["NEW.AX"],
                "name": ["New"],
                "industry": ["New"],
                "listing_date": [pd.Timestamp("2024-01-01")],
                "market_cap": [2e6],
            })
            result = load_directory(p, refresh=True)
            assert len(result) == 1
            assert result["ticker"].iloc[0] == "NEW.AX"


class TestRecentIPOs:
    def test_filters_by_lookback(self):
        today = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ticker": ["NEW.AX", "OLD.AX"],
            "name": ["NewCo", "OldCo"],
            "industry": ["Tech", "Tech"],
            "listing_date": [
                pd.Timestamp(today.replace(year=today.year - 1)),
                pd.Timestamp("2000-01-01"),
            ],
            "market_cap": [1e9, 100e6],
        })
        df["listing_date"] = pd.to_datetime(df["listing_date"], utc=True)
        recent = recent_ipos_from_directory(df, lookback_years=2.0)
        assert len(recent) == 1
        assert recent["ticker"].iloc[0] == "NEW.AX"
