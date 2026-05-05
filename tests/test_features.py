"""Tests for feature engineering — focus on point-in-time correctness (no leakage)."""

import numpy as np
import pandas as pd
import pytest

from src.config import RV_HORIZON, TARGET_COL
from src.data.features import _target, build_ticker_features


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "ticker": "TEST",
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }
    )


class TestTargetNoLeakage:
    """The target must not use any information available at date t."""

    def test_target_uses_future_returns(self):
        df = _make_ohlcv(100)
        feats = build_ticker_features(df)
        # The target on row i uses close prices at t+1 … t+RV_HORIZON.
        # Permuting future prices must change the target.
        df_shuffled = df.copy()
        df_shuffled.loc[df_shuffled.index[-10:], "close"] = (
            df_shuffled.loc[df_shuffled.index[-10:], "close"].values[::-1]
        )
        feats_shuffled = build_ticker_features(df_shuffled)
        # At least one target value should differ (the rows just before the shuffled window)
        changed = (feats[TARGET_COL] != feats_shuffled[TARGET_COL]).sum()
        assert changed > 0, "Shuffling future prices had no effect on target — possible leakage"

    def test_target_nan_at_tail(self):
        df = _make_ohlcv(150)
        feats = build_ticker_features(df)
        tail = feats.tail(RV_HORIZON)
        # Last RV_HORIZON rows should have NaN target (no complete future window)
        assert tail[TARGET_COL].isna().all(), (
            f"Expected NaN in last {RV_HORIZON} target rows, got {tail[TARGET_COL].values}"
        )

    def test_target_not_nan_before_tail(self):
        df = _make_ohlcv(150)
        feats = build_ticker_features(df)
        interior = feats.iloc[70:-RV_HORIZON]
        nan_frac = interior[TARGET_COL].isna().mean()
        assert nan_frac < 0.05, f"Too many NaN targets in interior: {nan_frac:.1%}"


class TestFeatureNoLeakage:
    """Features as of date t must only use data through t."""

    def test_perturbing_past_changes_features(self):
        df = _make_ohlcv(150)
        feats = build_ticker_features(df)

        df2 = df.copy()
        # Change close prices in the past — features should change for later rows
        df2.loc[df2.index[10:20], "close"] *= 1.5
        feats2 = build_ticker_features(df2)

        # Changing rows 10-19 alters ret[10] and ret[20] (boundary effects).
        # rv_5d at row 14 uses ret[10-14], catching the change at ret[10].
        row_idx = 14
        assert feats.loc[row_idx, "rv_5d"] != feats2.loc[row_idx, "rv_5d"], (
            "Past price change did not propagate to future feature — suspect issue"
        )

    def test_perturbing_future_does_not_change_features(self):
        """Changing prices strictly after date t must not change features at t."""
        df = _make_ohlcv(150)
        feats = build_ticker_features(df)

        df2 = df.copy()
        # Perturb only the last 20 rows (future data for earlier rows)
        df2.loc[df2.index[-20:], "close"] *= 2.0
        feats2 = build_ticker_features(df2)

        # Features at row 50 (well before the perturbation) must be identical
        check_row = 50
        feature_cols = [
            c for c in feats.columns if c not in ("date", "ticker", TARGET_COL)
        ]
        for col in feature_cols:
            v1, v2 = feats.loc[check_row, col], feats2.loc[check_row, col]
            both_nan = pd.isna(v1) and pd.isna(v2)
            assert both_nan or v1 == pytest.approx(v2, rel=1e-9), (
                f"Feature '{col}' at row {check_row} changed when only future data was perturbed: "
                f"{v1} → {v2}"
            )


class TestFeatureValues:
    def test_ret_lag_1_matches_manual(self):
        df = _make_ohlcv(50)
        feats = build_ticker_features(df)
        manual_ret = np.log(df["close"] / df["close"].shift(1))
        for i in range(5, 30):
            expected = manual_ret.iloc[i - 1]
            actual = feats.loc[i, "ret_lag_1"]
            if pd.isna(expected):
                assert pd.isna(actual)
            else:
                assert actual == pytest.approx(expected, rel=1e-9)

    def test_rv_window_positive(self):
        df = _make_ohlcv(100)
        feats = build_ticker_features(df)
        rv_cols = [f"rv_{w}d" for w in [5, 10, 21]]
        for col in rv_cols:
            non_nan = feats[col].dropna()
            assert (non_nan >= 0).all(), f"{col} has negative values"

    def test_day_of_week_range(self):
        df = _make_ohlcv(100)
        feats = build_ticker_features(df)
        assert feats["day_of_week"].between(0, 4).all(), "day_of_week out of [0,4]"

    def test_feature_count(self):
        df = _make_ohlcv(200)
        feats = build_ticker_features(df)
        feature_cols = [c for c in feats.columns if c not in ("date", "ticker")]
        # build_ticker_features produces 21 features + 1 target = 22 cols (cs_rank added later)
        assert len(feature_cols) >= 20, f"Too few features: {len(feature_cols)}"
