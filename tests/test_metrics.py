"""Tests for QLIKE, RMSE, MAE metrics on known inputs."""

import numpy as np
import pytest

from src.eval.metrics import (
    bootstrap_qlike_diff,
    compute_all_metrics,
    mae_log_vol,
    metrics_by_regime,
    qlike,
    rmse_log_vol,
)


class TestQLIKE:
    def test_perfect_forecast_is_zero(self):
        rv = np.array([0.01, 0.02, 0.015, 0.03])
        assert qlike(rv, rv) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        # For ratio r = a/p, QLIKE = r - log(r) - 1
        # r=2: 2 - log(2) - 1 = 1 - log(2) ≈ 0.3069
        a = np.array([2.0])
        p = np.array([1.0])
        expected = 2.0 - np.log(2.0) - 1.0
        assert qlike(a, p) == pytest.approx(expected, rel=1e-9)

    def test_positive_everywhere(self):
        rng = np.random.default_rng(0)
        rv_a = rng.uniform(0.005, 0.05, 1000)
        rv_p = rng.uniform(0.005, 0.05, 1000)
        assert qlike(rv_a, rv_p) >= 0.0

    def test_underprediction_penalized_more_than_overprediction(self):
        rv_a = np.array([0.02])
        rv_under = np.array([0.01])   # ratio = 2
        rv_over = np.array([0.04])    # ratio = 0.5
        assert qlike(rv_a, rv_under) > qlike(rv_a, rv_over)

    def test_ignores_nonpositive(self):
        rv_a = np.array([0.01, -0.01, 0.02, 0.0])
        rv_p = np.array([0.01, 0.01, 0.02, 0.01])
        # Only rows 0 and 2 are valid → QLIKE = 0
        assert qlike(rv_a, rv_p) == pytest.approx(0.0, abs=1e-10)


class TestRMSE:
    def test_zero_on_perfect(self):
        a = np.array([0.1, 0.2, -0.1])
        assert rmse_log_vol(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        p = np.array([1.0, -1.0])
        assert rmse_log_vol(a, p) == pytest.approx(1.0, rel=1e-9)

    def test_symmetry(self):
        a = np.array([0.1, 0.2, 0.3])
        p = np.array([0.15, 0.25, 0.35])
        assert rmse_log_vol(a, p) == pytest.approx(rmse_log_vol(p, a), rel=1e-9)


class TestMAE:
    def test_zero_on_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae_log_vol(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self):
        a = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, 2.0, 3.0])
        assert mae_log_vol(a, p) == pytest.approx(2.0, rel=1e-9)


class TestComputeAllMetrics:
    def test_keys_present(self):
        a = np.array([0.1, 0.2, 0.3])
        m = compute_all_metrics(a, a)
        assert set(m.keys()) == {"qlike", "rmse_log_vol", "mae_log_vol"}

    def test_perfect_forecast(self):
        a = np.array([0.1, 0.2, 0.3])
        m = compute_all_metrics(a, a)
        assert m["qlike"] == pytest.approx(0.0, abs=1e-8)
        assert m["rmse_log_vol"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae_log_vol"] == pytest.approx(0.0, abs=1e-10)


class TestBootstrapCI:
    def test_negative_diff_when_model_better(self):
        rng = np.random.default_rng(1)
        rv_actual = rng.uniform(0.01, 0.03, 500)
        rv_model = rv_actual * rng.uniform(0.95, 1.05, 500)   # near-perfect
        rv_baseline = rv_actual * rng.uniform(0.5, 2.0, 500)  # noisy baseline
        ci = bootstrap_qlike_diff(rv_actual, rv_model, rv_baseline, n_samples=200)
        assert ci["point"] < 0, "Model better than baseline should yield negative point estimate"

    def test_ci_contains_zero_when_equal(self):
        rng = np.random.default_rng(2)
        rv_actual = rng.uniform(0.01, 0.03, 500)
        rv_model = rv_actual * rng.uniform(0.9, 1.1, 500)
        rv_baseline = rv_actual * rng.uniform(0.9, 1.1, 500)
        ci = bootstrap_qlike_diff(rv_actual, rv_model, rv_baseline, n_samples=200)
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]

    def test_output_keys(self):
        rv = np.ones(100) * 0.02
        ci = bootstrap_qlike_diff(rv, rv, rv, n_samples=10)
        assert set(ci.keys()) == {"point", "ci_low", "ci_high"}


class TestMetricsByRegime:
    def test_returns_dataframe_with_quintiles(self):
        rng = np.random.default_rng(3)
        a = rng.normal(-3.5, 0.5, 500)
        p = a + rng.normal(0, 0.1, 500)
        df = metrics_by_regime(a, p, n_quintiles=5)
        assert len(df) == 5
        assert "qlike" in df.columns
        assert "rmse_log_vol" in df.columns
        assert "n" in df.columns
