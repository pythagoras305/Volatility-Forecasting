"""Rolling historical volatility baseline."""

import logging

import numpy as np
import pandas as pd

from src.config import ROLLING_VOL_WINDOW_CANDIDATES
from src.eval.metrics import qlike
from src.models.base import VolModel

logger = logging.getLogger(__name__)


class RollingHistVol(VolModel):
    """Predict log_rv_5_next as the log of trailing realized vol.

    The best window is selected by minimizing QLIKE on the validation set.
    At prediction time no fitting is required — the relevant trailing-RV
    column is already present in the feature matrix.
    """

    def __init__(self, window: int | None = None):
        self._window = window  # None = tune on val
        self._best_window: int | None = window

    @property
    def name(self) -> str:
        w = self._best_window or "unfit"
        return f"rolling_hist_vol_{w}d"

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        val: pd.DataFrame | None = None,
    ) -> None:
        if self._window is not None:
            self._best_window = self._window
            return

        # Tune window on validation set; fall back to training set if val is None
        eval_df = val if val is not None and len(val) > 0 else train
        best_qlike = np.inf

        for w in ROLLING_VOL_WINDOW_CANDIDATES:
            col = f"log_rv_{w}d"
            if col not in eval_df.columns:
                continue
            preds = eval_df[col].values
            actuals = eval_df[target_col].values
            mask = np.isfinite(preds) & np.isfinite(actuals)
            if mask.sum() < 10:
                continue
            rv_a = np.exp(actuals[mask])
            rv_p = np.exp(preds[mask])
            ql = qlike(rv_a, rv_p)
            logger.debug("window=%dd QLIKE=%.4f", w, ql)
            if ql < best_qlike:
                best_qlike = ql
                self._best_window = w

        logger.info("RollingHistVol best window: %dd (QLIKE=%.4f)", self._best_window, best_qlike)

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self._best_window is None:
            raise RuntimeError("Model has not been fit. Call fit() first.")
        col = f"log_rv_{self._best_window}d"
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame. Available: {df.columns.tolist()}")
        return df[col].values.copy()

    def get_params(self) -> dict:
        return {"window": self._best_window}
