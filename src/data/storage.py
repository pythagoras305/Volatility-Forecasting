"""SQLite read/write helpers for OHLCV data."""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator

import pandas as pd

from src.config import OHLCV_DB_PATH

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (ticker, date)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv (ticker);
"""


@contextmanager
def get_conn(db_path=None) -> Generator[sqlite3.Connection, None, None]:
    path = db_path or OHLCV_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def init_db(db_path=None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for stmt in CREATE_INDEX_SQL.strip().split("\n"):
            conn.execute(stmt)
        conn.commit()
    logger.debug("Database initialised at %s", db_path or OHLCV_DB_PATH)


def write_ohlcv(df: pd.DataFrame, db_path=None) -> int:
    """Upsert OHLCV rows. df must have columns: ticker, date, open, high, low, close, volume.

    Returns number of rows written.
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = df[list(required)].copy()
    rows["date"] = rows["date"].astype(str)

    with get_conn(db_path) as conn:
        rows.to_sql("ohlcv", conn, if_exists="append", index=False, method="multi")
        conn.execute(
            "DELETE FROM ohlcv WHERE rowid NOT IN "
            "(SELECT MIN(rowid) FROM ohlcv GROUP BY ticker, date)"
        )
        conn.commit()

    logger.debug("Wrote %d rows to ohlcv", len(rows))
    return len(rows)


def read_ohlcv(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    db_path=None,
) -> pd.DataFrame:
    """Read OHLCV data, optionally filtered by tickers and/or date range."""
    clauses = []
    params: list = []

    if tickers:
        placeholders = ",".join("?" * len(tickers))
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT ticker, date, open, high, low, close, volume FROM ohlcv {where} ORDER BY ticker, date"

    with get_conn(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])

    df["date"] = pd.to_datetime(df["date"])
    return df


def get_stored_tickers(db_path=None) -> list[str]:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def get_date_range(ticker: str, db_path=None) -> tuple[str, str]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0], row[1]
