"""Abstract base class for all volatility models."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class VolModel(ABC):
    """Interface every volatility model must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (used in MLflow run names)."""

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        val: pd.DataFrame | None = None,
    ) -> None:
        """Fit the model on training data.

        Args:
            train: Training rows (ticker, date, features, target).
            feature_cols: Ordered list of feature column names.
            target_col: Name of the target column (log_rv_5_next).
            val: Optional validation DataFrame for early stopping / tuning.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Return log-RV predictions for every row in df.

        Args:
            df: DataFrame with at least the columns in feature_cols.
            feature_cols: Same ordered list used during fit.

        Returns:
            1-D array of log-RV predictions, same length as df.
        """

    def get_params(self) -> dict:
        """Return hyperparameters to log in MLflow. Override as needed."""
        return {}
