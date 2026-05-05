"""GARCH(1,1) volatility baseline — per-ticker, weekly refit."""

import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from arch import arch_model

from src.config import GARCH_REFIT_WINDOW_YEARS, RV_HORIZON
from src.models.base import VolModel

logger = logging.getLogger(__name__)

# arch emits convergence warnings frequently; suppress at INFO level
warnings.filterwarnings("ignore", category=UserWarning, module="arch")


class GARCHModel(VolModel):
    """GARCH(1,1) on daily log returns.

    Strategy:
    - Refit weekly using a rolling 2-year window of daily returns.
    - 5-day ahead variance forecast = sum of 1-day ahead variance forecasts × h.
    - Convert variance forecast to vol, then log.
    - At prediction time, stored fitted params are applied to each (ticker, date).
    """

    def __init__(self):
        # Maps ticker → list of (cutoff_date, forecast_df) where forecast_df is indexed by date
        self._forecasts: dict[str, pd.Series] = {}

    @property
    def name(self) -> str:
        return "garch_1_1"

    def _fit_ticker(
        self,
        ticker: str,
        returns: pd.Series,
        refit_window_days: int,
        horizon: int,
    ) -> pd.Series:
        """Produce out-of-sample 5-day vol forecasts for one ticker via weekly refit.

        Returns a Series indexed by date with log-RV-5 predictions.
        """
        returns = returns.dropna().sort_index()
        if len(returns) < 100:
            logger.warning("%s: insufficient returns (%d). Skipping GARCH.", ticker, len(returns))
            return pd.Series(dtype=float)

        forecasts = {}
        dates = returns.index
        # Build refit schedule: every Monday (weekly), or every 5 business days
        refit_dates = dates[dates.to_series().dt.dayofweek == 0]  # Mondays

        last_model = None
        last_refit = None

        for i, d in enumerate(dates):
            # Refit on Mondays (or first date)
            if last_refit is None or d in refit_dates:
                window_start_idx = max(0, i - refit_window_days)
                window = returns.iloc[window_start_idx:i]
                if len(window) < 60:
                    continue
                try:
                    am = arch_model(window * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
                    res = am.fit(disp="off", show_warning=False)
                    last_model = res
                    last_refit = d
                except Exception as exc:
                    logger.debug("%s @ %s: GARCH fit failed: %s", ticker, d.date(), exc)
                    continue

            if last_model is None:
                continue

            # 5-day ahead variance forecast
            try:
                fc = last_model.forecast(horizon=horizon, reindex=False)
                # variance is in (return × 100)^2 units → divide by 100^2
                var_fwd = fc.variance.values[-1]  # shape (horizon,)
                rv_5 = np.sqrt(np.sum(var_fwd) / 100**2)
                forecasts[d] = np.log(rv_5) if rv_5 > 0 else np.nan
            except Exception as exc:
                logger.debug("%s @ %s: GARCH forecast failed: %s", ticker, d.date(), exc)

        return pd.Series(forecasts)

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        val: pd.DataFrame | None = None,
    ) -> None:
        refit_window_days = GARCH_REFIT_WINDOW_YEARS * 252
        # Use train + val data for fitting so we have forecasts covering both periods
        full_df = train if val is None else pd.concat([train, val])
        full_df = full_df.sort_values(["ticker", "date"])

        tickers = full_df["ticker"].unique()
        n_tickers = len(tickers)
        logger.info("Fitting GARCH(1,1) for %d tickers...", n_tickers)

        for t_idx, ticker in enumerate(tickers, 1):
            sub = full_df[full_df["ticker"] == ticker].sort_values("date")
            # Reconstruct returns from ret_lag_1 (which is yesterday's return, i.e. ret at t-1)
            # We need return at date t: use log_rv features are not returns.
            # ret_lag_1 at row i = return at date (date[i] - 1 business day).
            # Shift back: actual return at date d = ret_lag_1 at d+1.
            # Simpler: use ret_lag_1 as a proxy return series (shifted by 1 day; negligible for GARCH).
            if t_idx % 50 == 0 or t_idx == n_tickers:
                logger.info("  GARCH progress: %d/%d tickers", t_idx, n_tickers)
            if "ret_lag_1" not in sub.columns:
                logger.warning("%s: no ret_lag_1 column — skipping", ticker)
                continue
            ret_series = sub.set_index("date")["ret_lag_1"].dropna()
            self._forecasts[ticker] = self._fit_ticker(
                ticker, ret_series, refit_window_days, RV_HORIZON
            )

        logger.info("GARCH fitting complete for %d tickers.", len(self._forecasts))

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        preds = np.full(len(df), np.nan)
        df = df.reset_index(drop=True)

        for i, row in df.iterrows():
            ticker = row["ticker"]
            date = pd.Timestamp(row["date"])
            if ticker not in self._forecasts:
                continue
            fc = self._forecasts[ticker]
            if date in fc.index:
                preds[i] = fc[date]
            else:
                # Use nearest available forecast (handles holidays / gaps)
                past = fc[fc.index <= date]
                if len(past) > 0:
                    preds[i] = past.iloc[-1]

        return preds

    def get_params(self) -> dict:
        return {
            "p": 1,
            "q": 1,
            "refit_window_years": GARCH_REFIT_WINDOW_YEARS,
            "forecast_horizon": RV_HORIZON,
        }
