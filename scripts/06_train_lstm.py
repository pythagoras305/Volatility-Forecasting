#!/usr/bin/env python
"""Train LSTM model; log to MLflow."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import numpy as np
import pandas as pd

from src.config import EXPERIMENT_LSTM, FEATURES_PATH, TARGET_COL
from src.eval.runner import run_model
from src.eval.splits import filter_split, get_fixed_split
from src.models.lstm import LSTMVolModel
from src.models.rolling_hist import RollingHistVol

if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH}...")
    panel = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # Build rolling baseline predictions on test for bootstrap CI
    split = get_fixed_split()
    feat_cols = [c for c in panel.columns if c not in ("ticker", "date", TARGET_COL)]
    train_df = filter_split(panel, split, "train")
    val_df = filter_split(panel, split, "val")
    test_df = filter_split(panel, split, "test")

    rolling = RollingHistVol()
    rolling.fit(train_df, feat_cols, TARGET_COL, val=val_df)
    baseline_rv_pred = np.exp(rolling.predict(test_df, feat_cols))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining LSTM on {device}...")

    lstm = LSTMVolModel()
    results = run_model(
        lstm,
        panel,
        experiment_name=EXPERIMENT_LSTM,
        extra_params=lstm.get_params(),
        baseline_rv_pred=baseline_rv_pred,
        tag_final=True,
    )

    print(f"\nLSTM results:")
    print(f"  Val  QLIKE={results['val_metrics']['qlike']:.4f}  RMSE={results['val_metrics']['rmse_log_vol']:.4f}")
    print(f"  Test QLIKE={results['test_metrics']['qlike']:.4f}  RMSE={results['test_metrics']['rmse_log_vol']:.4f}")
    print(f"  MLflow run ID: {results['run_id']}")
