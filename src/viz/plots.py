"""Visualization utilities for model comparison and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError:
        raise ImportError("matplotlib is required for plots: pip install matplotlib")


def plot_pred_vs_actual(
    log_rv_actual: np.ndarray,
    log_rv_pred: np.ndarray,
    model_name: str,
    ax=None,
):
    """Scatter plot of predicted vs actual log-RV."""
    plt, _ = _try_import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    mask = np.isfinite(log_rv_actual) & np.isfinite(log_rv_pred)
    a, p = log_rv_actual[mask], log_rv_pred[mask]
    ax.scatter(a, p, alpha=0.15, s=2, rasterized=True)
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual log-RV")
    ax.set_ylabel("Predicted log-RV")
    ax.set_title(f"{model_name} — predicted vs actual")
    ax.legend(fontsize=8)
    return ax


def plot_qlike_by_regime(regime_df: pd.DataFrame, model_name: str, ax=None):
    """Bar chart of QLIKE per vol-regime quintile."""
    plt, _ = _try_import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    quintiles = regime_df.index.tolist()
    qlikes = regime_df["qlike"].values
    ax.bar(quintiles, qlikes)
    ax.set_xlabel("Vol regime quintile (0=low, 4=high)")
    ax.set_ylabel("QLIKE")
    ax.set_title(f"{model_name} — QLIKE by vol regime")
    return ax


def plot_test_qlike_timeseries(
    dates: pd.DatetimeIndex,
    log_rv_actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    rolling_window: int = 63,
    ax=None,
):
    """Rolling-window QLIKE over time for each model."""
    from src.eval.metrics import qlike as qlike_fn

    plt, _ = _try_import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    dates = pd.to_datetime(dates)
    rv_actual = np.exp(log_rv_actual)

    for model_name, log_pred in predictions.items():
        rv_pred = np.exp(log_pred)
        mask = np.isfinite(rv_actual) & np.isfinite(rv_pred)
        ql_series = []
        date_series = []
        for i in range(rolling_window, len(dates)):
            idx = np.where(mask)[0]
            window_idx = idx[(idx >= i - rolling_window) & (idx < i)]
            if len(window_idx) < 10:
                continue
            ql_series.append(qlike_fn(rv_actual[window_idx], rv_pred[window_idx]))
            date_series.append(dates[i])
        ax.plot(date_series, ql_series, label=model_name, linewidth=1)

    ax.set_ylabel(f"Rolling {rolling_window}d QLIKE")
    ax.set_title("Test-set QLIKE over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def save_comparison_figure(
    results: dict[str, dict],  # model_name → {test_actuals, test_preds, regime_df}
    output_path: str | None = None,
):
    """Generate a multi-panel comparison figure and optionally save it."""
    plt, gridspec = _try_import_matplotlib()
    n_models = len(results)
    fig = plt.figure(figsize=(6 * n_models, 12))
    gs = gridspec.GridSpec(3, n_models, figure=fig)

    for col, (model_name, r) in enumerate(results.items()):
        ax1 = fig.add_subplot(gs[0, col])
        plot_pred_vs_actual(r["test_actuals"], r["test_preds"], model_name, ax=ax1)

        ax2 = fig.add_subplot(gs[1, col])
        if "regime_df" in r:
            plot_qlike_by_regime(r["regime_df"], model_name, ax=ax2)

    # Time-series plot spans all columns
    ax3 = fig.add_subplot(gs[2, :])
    first_actuals = next(iter(results.values()))["test_actuals"]
    if "dates" in next(iter(results.values())):
        dates = next(iter(results.values()))["dates"]
        predictions = {name: r["test_preds"] for name, r in results.items()}
        plot_test_qlike_timeseries(dates, first_actuals, predictions, ax=ax3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
