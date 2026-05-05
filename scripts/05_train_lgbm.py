#!/usr/bin/env python
"""Train LightGBM model with Optuna hyperparameter search; log to MLflow."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import pandas as pd

from src.config import EXPERIMENT_BASELINE_ROLLING, EXPERIMENT_LGBM, FEATURES_PATH
from src.eval.runner import run_model
from src.eval.splits import filter_split, get_fixed_split
from src.eval.metrics import qlike
from src.models.lgbm import LGBMVolModel
from src.models.rolling_hist import RollingHistVol

import numpy as np
import mlflow

if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH}...")
    panel = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # Build rolling baseline RV predictions on test set for bootstrap CI
    from src.config import MLFLOW_TRACKING_URI, TARGET_COL
    split = get_fixed_split()
    test_df = filter_split(panel, split, "test")

    rolling = RollingHistVol()
    train_df = filter_split(panel, split, "train")
    val_df = filter_split(panel, split, "val")
    feat_cols = [c for c in panel.columns if c not in ("ticker", "date", TARGET_COL)]
    rolling.fit(train_df, feat_cols, TARGET_COL, val=val_df)
    rolling_test_log_preds = rolling.predict(test_df, feat_cols)
    baseline_rv_pred = np.exp(rolling_test_log_preds)

    print(f"\nTraining LightGBM ({LGBMVolModel().n_trials} Optuna trials)...")
    lgbm = LGBMVolModel()
    results = run_model(
        lgbm,
        panel,
        experiment_name=EXPERIMENT_LGBM,
        extra_params=lgbm.get_params(),
        baseline_rv_pred=baseline_rv_pred,
        tag_final=True,
    )

    print(f"\nLightGBM results:")
    print(f"  Val  QLIKE={results['val_metrics']['qlike']:.4f}  RMSE={results['val_metrics']['rmse_log_vol']:.4f}")
    print(f"  Test QLIKE={results['test_metrics']['qlike']:.4f}  RMSE={results['test_metrics']['rmse_log_vol']:.4f}")
    print(f"  MLflow run ID: {results['run_id']}")
