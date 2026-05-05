#!/usr/bin/env python
"""Download daily OHLCV from yfinance for the full S&P 500 universe."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.config import INGEST_END, INGEST_START
from src.data.ingest import ingest_universe
from src.data.universe import get_tickers

if __name__ == "__main__":
    tickers = get_tickers()
    print(f"Ingesting {len(tickers)} tickers from {INGEST_START} to {INGEST_END}...")

    statuses = ingest_universe(tickers, start=INGEST_START, end=INGEST_END)

    ok = [t for t, s in statuses.items() if s == "ok"]
    skipped = [t for t, s in statuses.items() if s == "skipped"]
    errors = [t for t, s in statuses.items() if s == "error"]

    print(f"\nResults: {len(ok)} ok, {len(skipped)} skipped, {len(errors)} errors")
    if skipped:
        print(f"Skipped (no data / delisted): {', '.join(sorted(skipped))}")
    if errors:
        print(f"Errors: {', '.join(sorted(errors))}")
