#!/usr/bin/env python
"""Scrape current S&P 500 constituents from Wikipedia and save to parquet."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.data.universe import fetch_sp500_constituents

if __name__ == "__main__":
    df = fetch_sp500_constituents(force_refresh=True)
    print(f"\nUniverse: {len(df)} tickers")
    print(df.head(10).to_string(index=False))
    print(f"\nSectors:\n{df['sector'].value_counts().to_string()}")
