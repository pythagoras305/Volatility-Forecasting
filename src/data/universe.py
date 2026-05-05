"""Fetch and cache the current S&P 500 constituent list from Wikipedia."""

import logging

import io

import pandas as pd
import requests

from src.config import UNIVERSE_PATH

logger = logging.getLogger(__name__)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """Return DataFrame[ticker, company, sector] for current S&P 500 members.

    Results are cached to UNIVERSE_PATH. Pass force_refresh=True to re-scrape.
    Tickers are normalized: dots replaced with hyphens (yfinance convention).
    """
    if UNIVERSE_PATH.exists() and not force_refresh:
        logger.info("Loading universe from cache: %s", UNIVERSE_PATH)
        return pd.read_parquet(UNIVERSE_PATH)

    logger.info("Scraping S&P 500 constituents from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; equity-vol-forecaster/1.0)"}
    tables = pd.read_html(
        io.StringIO(requests.get(WIKIPEDIA_URL, timeout=30, headers=headers).text),
        flavor="lxml",
    )
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "company", "sector"]

    # yfinance uses hyphens where Wikipedia uses dots (e.g. BRK.B → BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df = df.sort_values("ticker").reset_index(drop=True)

    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNIVERSE_PATH, index=False)
    logger.info("Saved %d tickers to %s", len(df), UNIVERSE_PATH)
    return df


def get_tickers(force_refresh: bool = False) -> list[str]:
    """Return sorted list of S&P 500 ticker symbols."""
    return fetch_sp500_constituents(force_refresh)["ticker"].tolist()
