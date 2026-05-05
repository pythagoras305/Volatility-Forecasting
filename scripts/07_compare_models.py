#!/usr/bin/env python
"""Query MLflow for final runs, build comparison table, write reports/comparison.md."""

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
import mlflow

from src.config import (
    EXPERIMENT_BASELINE_GARCH,
    EXPERIMENT_BASELINE_ROLLING,
    EXPERIMENT_LGBM,
    EXPERIMENT_LSTM,
    FEATURES_PATH,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    TARGET_COL,
    VOL_REGIME_QUINTILES,
)
from src.eval.metrics import (
    bootstrap_qlike_diff,
    compute_all_metrics,
    metrics_by_regime,
    qlike,
)
from src.eval.splits import filter_split, get_fixed_split


def fetch_final_run(experiment_name: str) -> dict | None:
    """Return metrics dict for the latest run tagged final=true in an experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.final = 'true'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {"run_id": run.info.run_id, "metrics": run.data.metrics, "params": run.data.params}
    except Exception as e:
        logger.warning("Could not fetch run for %s: %s", experiment_name, e)
        return None


logger = logging.getLogger(__name__)

EXPERIMENTS = {
    "Rolling HV": EXPERIMENT_BASELINE_ROLLING,
    "GARCH(1,1)": EXPERIMENT_BASELINE_GARCH,
    "LightGBM": EXPERIMENT_LGBM,
    "LSTM": EXPERIMENT_LSTM,
}

METRIC_DISPLAY = {
    "test_qlike": "QLIKE (↓)",
    "test_rmse_log_vol": "RMSE log-RV (↓)",
    "test_mae_log_vol": "MAE log-RV (↓)",
    "val_qlike": "Val QLIKE (↓)",
}


def build_comparison_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, run in runs.items():
        if run is None:
            rows.append({"Model": model_name, **{k: "N/A" for k in METRIC_DISPLAY}})
            continue
        m = run["metrics"]
        row = {"Model": model_name}
        for key, label in METRIC_DISPLAY.items():
            row[label] = f"{m.get(key, float('nan')):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_regime_table(runs: dict[str, dict]) -> pd.DataFrame | None:
    """Extract per-regime QLIKE from MLflow metrics."""
    rows = []
    for model_name, run in runs.items():
        if run is None:
            continue
        m = run["metrics"]
        for q in range(VOL_REGIME_QUINTILES):
            key = f"test_qlike_q{q}"
            if key in m:
                rows.append({
                    "Model": model_name,
                    "Quintile": q,
                    "QLIKE": round(m[key], 4),
                })
    if not rows:
        return None
    return pd.DataFrame(rows).pivot(index="Model", columns="Quintile", values="QLIKE")


def bootstrap_ci_section(runs: dict[str, dict]) -> str:
    """Build text block describing bootstrap CI for ML vs best baseline."""
    baseline_models = ["Rolling HV", "GARCH(1,1)"]
    ml_models = ["LightGBM", "LSTM"]

    lines = ["### Bootstrap CI: QLIKE difference (ML − best baseline)\n"]
    lines.append("Negative = ML is better. CI computed over 1 000 bootstrap resamples of test predictions.\n")

    for ml in ml_models:
        if runs.get(ml) is None:
            continue
        for base in baseline_models:
            if runs.get(base) is None:
                continue
            ml_run = runs[ml]
            base_run = runs[base]
            point = ml_run["metrics"].get("test_qlike_diff_vs_baseline_point")
            lo = ml_run["metrics"].get("test_qlike_diff_vs_baseline_ci_low")
            hi = ml_run["metrics"].get("test_qlike_diff_vs_baseline_ci_high")
            if point is None:
                lines.append(f"- {ml} vs {base}: bootstrap CI not available (run scripts/05/06 with baseline_rv_pred)\n")
                continue
            beats = "✓ beats baseline" if hi < 0 else ("✗ does not beat baseline" if lo > 0 else "~ inconclusive")
            lines.append(f"- **{ml} vs {base}**: Δ={point:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]) — {beats}\n")

    return "\n".join(lines)


def df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


if __name__ == "__main__":
    print("Querying MLflow for final runs...")
    runs = {name: fetch_final_run(exp) for name, exp in EXPERIMENTS.items()}

    found = [n for n, r in runs.items() if r is not None]
    missing = [n for n, r in runs.items() if r is None]
    print(f"  Found: {found}")
    if missing:
        print(f"  Missing (not yet trained): {missing}")

    comparison_df = build_comparison_table(runs)
    regime_df = build_regime_table(runs)
    ci_section = bootstrap_ci_section(runs)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "comparison.md"

    lines = [
        "# Model Comparison Report\n",
        f"_Generated on {pd.Timestamp.now().date()}_\n\n",
        "## Overall metrics (test set: 2022-01-01 → 2024-12-31)\n\n",
        df_to_md(comparison_df) + "\n\n",
        "_Primary metric: QLIKE. Lower is better. QLIKE = mean(RV_actual/RV_pred − log(RV_actual/RV_pred) − 1)._\n\n",
    ]

    if regime_df is not None:
        lines += [
            "## QLIKE by volatility regime (test set)\n\n",
            "Quintile 0 = lowest vol, Quintile 4 = highest vol.\n\n",
            regime_df.to_markdown() + "\n\n",
        ]

    lines += [ci_section + "\n\n"]

    lines += [
        "## Notes\n\n",
        "- **Survivorship bias**: Universe is current S&P 500 constituents. "
        "Companies that were delisted or removed between 2014–2024 are excluded. "
        "This likely overstates model quality on stable large-caps.\n",
        "- **COVID shock (2020–2021)**: The validation period includes the March 2020 vol spike. "
        "Models saw this regime during validation but not training.\n",
        "- **No transaction costs**: These are forecast quality metrics, not strategy returns.\n",
        "- **Daily only**: No intraday microstructure, no overnight gaps modelled separately.\n",
    ]

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print("\n--- Comparison table ---")
    # Use ASCII-safe column names for console output on Windows
    safe = comparison_df.rename(columns=lambda c: c.replace("↓", "(lower=better)"))
    print(safe.to_string(index=False))
