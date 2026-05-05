"""Run a VolModel through the eval framework and log results to MLflow."""

import logging
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, TARGET_COL, VOL_REGIME_QUINTILES
from src.eval.metrics import bootstrap_qlike_diff, compute_all_metrics, metrics_by_regime
from src.eval.splits import Split, filter_split, get_fixed_split

if TYPE_CHECKING:
    from src.models.base import VolModel

logger = logging.getLogger(__name__)


def _feature_cols(panel: pd.DataFrame) -> list[str]:
    exclude = {"ticker", "date", TARGET_COL}
    return [c for c in panel.columns if c not in exclude]


def run_model(
    model: "VolModel",
    panel: pd.DataFrame,
    experiment_name: str,
    split: Split | None = None,
    extra_params: dict | None = None,
    baseline_rv_pred: np.ndarray | None = None,
    tag_final: bool = False,
) -> dict:
    """Fit model on train, evaluate on val and test, log to MLflow.

    Returns a dict with keys: val_metrics, test_metrics, val_preds, test_preds.
    """
    if split is None:
        split = get_fixed_split()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    feature_cols = _feature_cols(panel)

    with mlflow.start_run(run_name=model.name) as run:
        # --- Log params ---
        mlflow.log_params({"model": model.name, "split": split.name, **(extra_params or {})})
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols[:50]))  # truncate for readability
        if tag_final:
            mlflow.set_tag("final", "true")

        # --- Train ---
        train_df = filter_split(panel, split, "train")
        logger.info(
            "[%s] Training on %d rows (%s to %s)",
            model.name,
            len(train_df),
            split.train_start.date(),
            split.train_end.date(),
        )
        val_df = filter_split(panel, split, "val")
        model.fit(train_df, feature_cols, TARGET_COL, val=val_df)

        # --- Validate ---
        val_preds = model.predict(val_df, feature_cols)
        val_actuals = val_df[TARGET_COL].values
        val_metrics = compute_all_metrics(val_actuals, val_preds)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        logger.info("[%s] Val   — QLIKE=%.4f RMSE=%.4f", model.name, val_metrics["qlike"], val_metrics["rmse_log_vol"])

        # --- Test ---
        test_df = filter_split(panel, split, "test")
        test_preds = model.predict(test_df, feature_cols)
        test_actuals = test_df[TARGET_COL].values
        test_metrics = compute_all_metrics(test_actuals, test_preds)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        logger.info("[%s] Test  — QLIKE=%.4f RMSE=%.4f", model.name, test_metrics["qlike"], test_metrics["rmse_log_vol"])

        # --- Regime metrics (test) ---
        regime_df = metrics_by_regime(test_actuals, test_preds, n_quintiles=VOL_REGIME_QUINTILES)
        for q, row in regime_df.iterrows():
            mlflow.log_metric(f"test_qlike_q{q}", row["qlike"])
            mlflow.log_metric(f"test_rmse_q{q}", row["rmse_log_vol"])

        # --- Bootstrap CI vs baseline ---
        if baseline_rv_pred is not None:
            rv_actual = np.exp(test_actuals)
            rv_model = np.exp(test_preds)
            ci = bootstrap_qlike_diff(rv_actual, rv_model, baseline_rv_pred)
            mlflow.log_metric("test_qlike_diff_vs_baseline_point", ci["point"])
            mlflow.log_metric("test_qlike_diff_vs_baseline_ci_low", ci["ci_low"])
            mlflow.log_metric("test_qlike_diff_vs_baseline_ci_high", ci["ci_high"])

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "val_actuals": val_actuals,
        "test_actuals": test_actuals,
    }
