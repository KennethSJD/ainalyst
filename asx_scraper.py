#!/usr/bin/env python3
"""Fetch the full ASX company directory → asx_directory.csv.

Thin CLI wrapper around ``ainalyst.directory``. Pulls every listed ASX
company with ticker, name, GICS industry, listing date, and market cap.

Usage:
    python asx_scraper.py [output.csv]
"""

from __future__ import annotations

import logging
import sys

from ainalyst.directory import fetch_asx_directory, save_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        df = fetch_asx_directory()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    path = save_directory(df, out)
    print(f"Success: {len(df)} companies saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
