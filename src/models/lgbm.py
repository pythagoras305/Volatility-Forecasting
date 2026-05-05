"""LightGBM regressor for volatility forecasting.

Single global model across all S&P 500 tickers; ticker is included as a
categorical feature. Hyperparameters are tuned via Optuna, optimizing QLIKE
on the validation set.
"""

import logging
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.config import LGBM_OPTUNA_TRIALS
from src.eval.metrics import qlike
from src.models.base import VolModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMVolModel(VolModel):
    """Global LightGBM model with Optuna hyperparameter search."""

    def __init__(self, n_trials: int = LGBM_OPTUNA_TRIALS):
        self.n_trials = n_trials
        self._model: lgb.Booster | None = None
        self._best_params: dict = {}
        self._ticker_encoder: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "lgbm"

    def _encode_tickers(self, tickers: pd.Series, fit: bool = False) -> np.ndarray:
        if fit:
            unique = sorted(tickers.unique())
            self._ticker_encoder = {t: i for i, t in enumerate(unique)}
        return tickers.map(self._ticker_encoder).fillna(-1).astype(int).values

    def _prepare(
        self, df: pd.DataFrame, feature_cols: list[str], fit: bool = False
    ) -> tuple[np.ndarray, list[str]]:
        cols = list(feature_cols)
        X = df[cols].values.astype(np.float32)
        ticker_enc = self._encode_tickers(df["ticker"], fit=fit).reshape(-1, 1)
        X = np.hstack([X, ticker_enc])
        all_cols = cols + ["ticker_enc"]
        return X, all_cols

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        val: pd.DataFrame | None = None,
    ) -> None:
        X_train, all_cols = self._prepare(train, feature_cols, fit=True)
        y_train = train[target_col].values.astype(np.float32)

        ticker_enc_idx = len(feature_cols)  # last column

        X_val, y_val = None, None
        if val is not None and len(val) > 0:
            X_val, _ = self._prepare(val, feature_cols)
            y_val = val[target_col].values.astype(np.float32)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": 42,
            }

            eval_data = []
            if X_val is not None:
                eval_data = [(X_val, y_val)]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_data if eval_data else None,
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                    categorical_feature=[ticker_enc_idx],
                )

            if X_val is not None and y_val is not None:
                preds = model.predict(X_val)
                rv_a = np.exp(y_val)
                rv_p = np.exp(preds)
                return qlike(rv_a, rv_p)
            else:
                preds = model.predict(X_train)
                rv_a = np.exp(y_train)
                rv_p = np.exp(preds)
                return qlike(rv_a, rv_p)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self._best_params = study.best_params
        logger.info("LGBM best params (QLIKE=%.4f): %s", study.best_value, self._best_params)

        # Refit final model with best params on full train+val
        full_X = X_train if X_val is None else np.vstack([X_train, X_val])
        full_y = y_train if y_val is None else np.concatenate([y_train, y_val])

        final_params = {
            "objective": "regression",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            "categorical_feature": [ticker_enc_idx],
            **self._best_params,
        }
        # Remove early-stopping-only params
        final_params.pop("early_stopping_rounds", None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = lgb.LGBMRegressor(**final_params)
            self._model.fit(
                full_X,
                full_y,
                categorical_feature=[ticker_enc_idx],
                callbacks=[lgb.log_evaluation(-1)],
            )

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit. Call fit() first.")
        X, _ = self._prepare(df, feature_cols)
        return self._model.predict(X)

    def get_params(self) -> dict:
        return {"n_trials": self.n_trials, **self._best_params}
