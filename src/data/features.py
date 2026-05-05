"""Feature engineering for the volatility forecaster.

All features are point-in-time correct: as of date t, only data through t is used.
The target log_rv_5_next uses dates t+1 … t+5 and is therefore NaN for the last
RV_HORIZON rows of each ticker — those rows are dropped before modelling.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    CS_RETURN_WINDOW,
    MA_WINDOWS,
    PARKINSON_WINDOWS,
    RETURN_LAGS,
    RV_HORIZON,
    RV_WINDOWS,
    TARGET_COL,
    VOL_ZSCORE_WINDOW,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers (operate on a single-ticker sorted DataFrame)
# ---------------------------------------------------------------------------


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Trailing realized volatility = sqrt(sum of squared log returns over window)."""
    return np.sqrt(log_ret.pow(2).rolling(window, min_periods=window).sum())


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson high-low range estimator over a rolling window."""
    hl2 = np.log(high / low).pow(2)
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * hl2.rolling(window, min_periods=window).mean())


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mu) / sigma.replace(0, np.nan)


def _target(log_ret: pd.Series, horizon: int = RV_HORIZON) -> pd.Series:
    """Forward realized volatility (next `horizon` days), then log-transformed."""
    fwd_sum_sq = log_ret.pow(2).shift(-horizon).rolling(horizon, min_periods=horizon).sum()
    # shift(-horizon) then rolling(horizon) gives sum of t+1 … t+horizon
    # But rolling looks backward; we need forward sum.
    # Correct approach: sum of the next `horizon` returns squared.
    sq = log_ret.pow(2)
    fwd = sum(sq.shift(-(k + 1)) for k in range(horizon))
    rv_fwd = np.sqrt(fwd)
    return np.log(rv_fwd.replace(0, np.nan))


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------


def build_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features for a single ticker.

    df must be sorted by date and contain: date, open, high, low, close, volume.
    Returns a new DataFrame with all feature columns and the target column.
    """
    df = df.sort_values("date").copy()
    ret = _log_returns(df["close"])

    out = pd.DataFrame({"date": df["date"].values, "ticker": df["ticker"].values})

    # Lagged log returns
    for lag in RETURN_LAGS:
        out[f"ret_lag_{lag}"] = ret.shift(lag).values

    # Realized vol over trailing windows
    for w in RV_WINDOWS:
        rv = _realized_vol(ret, w)
        out[f"rv_{w}d"] = rv.values
        out[f"log_rv_{w}d"] = np.log(rv.replace(0, np.nan)).values

    # Parkinson high-low range vol
    for w in PARKINSON_WINDOWS:
        out[f"parkinson_{w}d"] = _parkinson_vol(df["high"], df["low"], w).values

    # Log volume z-score
    log_vol = np.log(df["volume"].replace(0, np.nan))
    out[f"log_vol_zscore_{VOL_ZSCORE_WINDOW}d"] = _zscore(log_vol, VOL_ZSCORE_WINDOW).values

    # Day of week (0=Monday … 4=Friday)
    out["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.values

    # Distance from MA (z-score)
    for w in MA_WINDOWS:
        ma = df["close"].rolling(w, min_periods=w).mean()
        dist = (df["close"] - ma) / ma.replace(0, np.nan)
        out[f"close_dist_ma_{w}d"] = _zscore(dist, w).values

    # Target: log forward realized vol
    out[TARGET_COL] = _target(ret).values

    return out


def _add_cs_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional return rank (decile 0–9) within each date."""
    lag_cols = [f"ret_lag_{lag}" for lag in RETURN_LAGS if lag <= CS_RETURN_WINDOW]
    trailing_ret = panel[lag_cols].sum(axis=1)
    panel = panel.copy()
    panel["_trailing_ret"] = trailing_ret
    panel["cs_ret_rank_decile"] = panel.groupby("date")["_trailing_ret"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    panel.drop(columns=["_trailing_ret"], inplace=True)
    return panel


# ---------------------------------------------------------------------------
# Main panel builder
# ---------------------------------------------------------------------------


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature panel from raw OHLCV data.

    ohlcv: DataFrame with columns [ticker, date, open, high, low, close, volume].
    Returns a panel DataFrame with one row per (ticker, date), all features, and target.
    Rows where the target is NaN (last RV_HORIZON days per ticker) are dropped.
    """
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    chunks = []
    tickers = ohlcv["ticker"].unique()
    logger.info("Building features for %d tickers...", len(tickers))

    for ticker in tickers:
        sub = ohlcv[ohlcv["ticker"] == ticker].sort_values("date")
        if len(sub) < max(RV_WINDOWS) + RV_HORIZON + 5:
            logger.warning("Insufficient data for %s (%d rows) — skipping", ticker, len(sub))
            continue
        chunks.append(build_ticker_features(sub))

    if not chunks:
        raise ValueError("No ticker produced valid features.")

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.dropna(subset=[TARGET_COL])

    # Add cross-sectional rank
    panel = _add_cs_rank(panel)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    logger.info(
        "Feature panel: %d rows, %d columns, %d tickers, dates %s to %s",
        len(panel),
        len(panel.columns),
        panel["ticker"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )
    return panel
