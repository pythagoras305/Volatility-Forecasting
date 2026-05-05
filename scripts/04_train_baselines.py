#!/usr/bin/env python
"""Train rolling historical vol and GARCH(1,1) baselines; log to MLflow."""

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

from src.config import (
    EXPERIMENT_BASELINE_GARCH,
    EXPERIMENT_BASELINE_ROLLING,
    FEATURES_PATH,
)
from src.eval.runner import run_model
from src.models.garch import GARCHModel
from src.models.rolling_hist import RollingHistVol

if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH}...")
    panel = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # --- Rolling historical vol ---
    print("\n[1/2] Rolling historical vol baseline...")
    rolling = RollingHistVol()
    rolling_results = run_model(
        rolling,
        panel,
        experiment_name=EXPERIMENT_BASELINE_ROLLING,
        extra_params=rolling.get_params(),
        tag_final=True,
    )
    print(f"  Val  QLIKE={rolling_results['val_metrics']['qlike']:.4f}  RMSE={rolling_results['val_metrics']['rmse_log_vol']:.4f}")
    print(f"  Test QLIKE={rolling_results['test_metrics']['qlike']:.4f}  RMSE={rolling_results['test_metrics']['rmse_log_vol']:.4f}")

    # --- GARCH(1,1) ---
    print("\n[2/2] GARCH(1,1) baseline (this will take a few minutes)...")
    garch = GARCHModel()
    garch_results = run_model(
        garch,
        panel,
        experiment_name=EXPERIMENT_BASELINE_GARCH,
        extra_params=garch.get_params(),
        tag_final=True,
    )
    print(f"  Val  QLIKE={garch_results['val_metrics']['qlike']:.4f}  RMSE={garch_results['val_metrics']['rmse_log_vol']:.4f}")
    print(f"  Test QLIKE={garch_results['test_metrics']['qlike']:.4f}  RMSE={garch_results['test_metrics']['rmse_log_vol']:.4f}")

    print("\nBaseline runs logged to MLflow.")
