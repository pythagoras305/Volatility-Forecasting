"""Download daily OHLCV data from yfinance and store in SQLite."""

import logging
import time

import pandas as pd
import yfinance as yf

from src.config import INGEST_END, INGEST_START
from src.data.storage import init_db, write_ohlcv

logger = logging.getLogger(__name__)

# yfinance bulk-download chunk size (avoids timeouts on large universes)
CHUNK_SIZE = 50
RETRY_SLEEP = 5  # seconds between retries
MAX_RETRIES = 3


def _normalize_yf_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Convert yfinance multi-index or single-ticker DataFrame to a flat frame."""
    if raw is None or raw.empty:
        return None

    df = raw.copy()

    # yfinance >= 0.2 returns MultiIndex columns when downloading multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(1):
            return None
        df = df.xs(ticker, axis=1, level=1)

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None

    df = df[list(needed)].dropna(how="all").copy()
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def download_ticker(
    ticker: str,
    start: str = INGEST_START,
    end: str = INGEST_END,
) -> pd.DataFrame | None:
    """Download OHLCV for a single ticker. Returns None if data is unavailable."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return _normalize_yf_df(raw, ticker)
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, ticker, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    return None


def ingest_universe(
    tickers: list[str],
    start: str = INGEST_START,
    end: str = INGEST_END,
    db_path=None,
) -> dict[str, str]:
    """Download OHLCV for all tickers and store in SQLite.

    Returns a dict mapping ticker → status ('ok' | 'skipped' | 'error').
    """
    init_db(db_path)
    statuses: dict[str, str] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Ingesting %s", i, total, ticker)
        try:
            df = download_ticker(ticker, start=start, end=end)
            if df is None or df.empty:
                logger.warning("No data returned for %s — skipping (possibly delisted)", ticker)
                statuses[ticker] = "skipped"
                continue
            write_ohlcv(df, db_path)
            statuses[ticker] = "ok"
        except Exception as exc:
            logger.error("Error ingesting %s: %s", ticker, exc)
            statuses[ticker] = "error"

    ok = sum(1 for s in statuses.values() if s == "ok")
    skipped = sum(1 for s in statuses.values() if s == "skipped")
    errors = sum(1 for s in statuses.values() if s == "error")
    logger.info("Ingestion complete: %d ok, %d skipped, %d errors", ok, skipped, errors)
    return statuses
