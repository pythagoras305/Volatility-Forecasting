"""Evaluation metrics for volatility forecasts."""

import numpy as np
import pandas as pd


def qlike(rv_actual: np.ndarray, rv_pred: np.ndarray) -> float:
    """QLIKE loss: mean(actual/pred - log(actual/pred) - 1).

    Both inputs must be in variance or volatility space (not log); this
    implementation expects RV (volatility, not variance) and operates on RV directly.
    Rows where pred <= 0 or actual <= 0 are dropped.
    """
    rv_actual = np.asarray(rv_actual, dtype=float)
    rv_pred = np.asarray(rv_pred, dtype=float)
    mask = (rv_actual > 0) & (rv_pred > 0) & np.isfinite(rv_actual) & np.isfinite(rv_pred)
    a, p = rv_actual[mask], rv_pred[mask]
    ratio = a / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse_log_vol(log_rv_actual: np.ndarray, log_rv_pred: np.ndarray) -> float:
    """RMSE on log realized volatility."""
    a = np.asarray(log_rv_actual, dtype=float)
    p = np.asarray(log_rv_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))


def mae_log_vol(log_rv_actual: np.ndarray, log_rv_pred: np.ndarray) -> float:
    """MAE on log realized volatility."""
    a = np.asarray(log_rv_actual, dtype=float)
    p = np.asarray(log_rv_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(a[mask] - p[mask])))


def compute_all_metrics(
    log_rv_actual: np.ndarray,
    log_rv_pred: np.ndarray,
) -> dict[str, float]:
    """Compute QLIKE, RMSE, and MAE from log-RV predictions.

    Converts log-RV back to RV for QLIKE.
    """
    rv_actual = np.exp(np.asarray(log_rv_actual, dtype=float))
    rv_pred = np.exp(np.asarray(log_rv_pred, dtype=float))
    return {
        "qlike": qlike(rv_actual, rv_pred),
        "rmse_log_vol": rmse_log_vol(log_rv_actual, log_rv_pred),
        "mae_log_vol": mae_log_vol(log_rv_actual, log_rv_pred),
    }


def bootstrap_qlike_diff(
    rv_actual: np.ndarray,
    rv_pred_model: np.ndarray,
    rv_pred_baseline: np.ndarray,
    n_samples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for QLIKE(model) - QLIKE(baseline).

    A negative difference means the model is better (lower QLIKE).
    Returns: point estimate, lower CI bound, upper CI bound.
    """
    rng = np.random.default_rng(seed)
    n = len(rv_actual)
    diffs = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = qlike(rv_actual[idx], rv_pred_model[idx]) - qlike(
            rv_actual[idx], rv_pred_baseline[idx]
        )

    alpha = 1 - ci
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    point = float(np.mean(diffs))
    return {"point": point, "ci_low": lo, "ci_high": hi}


def metrics_by_regime(
    log_rv_actual: np.ndarray,
    log_rv_pred: np.ndarray,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Compute metrics split into vol-regime quintiles based on actual log-RV.

    Returns a DataFrame with one row per quintile.
    """
    a = np.asarray(log_rv_actual, dtype=float)
    p = np.asarray(log_rv_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]

    quintile_labels = pd.qcut(a, n_quintiles, labels=False, duplicates="drop")
    rows = []
    for q in range(n_quintiles):
        idx = quintile_labels == q
        if idx.sum() == 0:
            continue
        m = compute_all_metrics(a[idx], p[idx])
        m["quintile"] = q
        m["n"] = int(idx.sum())
        m["rv_mean"] = float(np.exp(a[idx]).mean())
        rows.append(m)

    return pd.DataFrame(rows).set_index("quintile")
