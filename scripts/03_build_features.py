#!/usr/bin/env python
"""Build feature panel from OHLCV data and save to parquet."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.config import FEATURES_PATH
from src.data.features import build_features
from src.data.storage import read_ohlcv

if __name__ == "__main__":
    print("Reading OHLCV from SQLite...")
    ohlcv = read_ohlcv()
    print(f"  {len(ohlcv):,} rows, {ohlcv['ticker'].nunique()} tickers")

    print("Building features...")
    panel = build_features(ohlcv)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_PATH, index=False)
    from src.config import TARGET_COL
    print(f"\nSaved feature panel: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"  Target column: {TARGET_COL}")
    print(f"  Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  Tickers: {panel['ticker'].nunique()}")
    print(f"\nFeature columns:\n{[c for c in panel.columns if c not in ('ticker','date')]}")
