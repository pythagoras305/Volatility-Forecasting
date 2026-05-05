"""Tests for walk-forward split logic — no overlap, proper purging."""

import pandas as pd
import pytest

from src.config import PURGE_DAYS, TEST_END, TEST_START, TRAIN_START, VAL_END
from src.eval.splits import (
    assert_no_overlap,
    filter_split,
    get_fixed_split,
    get_walk_forward_splits,
)


def _make_panel(start: str = "2014-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"date": dates, "ticker": "AAPL", "target": 0.0})


class TestFixedSplit:
    def test_purge_gap(self):
        s = get_fixed_split()
        gap = (s.val_start - s.train_end).days
        assert gap >= PURGE_DAYS

    def test_val_before_test(self):
        s = get_fixed_split()
        assert s.val_end < s.test_start

    def test_dates_in_order(self):
        s = get_fixed_split()
        assert s.train_start < s.train_end < s.val_start < s.val_end < s.test_start < s.test_end


class TestFilterSplit:
    def test_no_rows_in_purge_gap(self):
        panel = _make_panel()
        split = get_fixed_split()
        train = filter_split(panel, split, "train")
        val = filter_split(panel, split, "val")
        # No date in train should appear in val
        assert len(set(train["date"]) & set(val["date"])) == 0

    def test_train_val_test_non_overlapping(self):
        panel = _make_panel()
        split = get_fixed_split()
        train = filter_split(panel, split, "train")
        val = filter_split(panel, split, "val")
        test = filter_split(panel, split, "test")
        all_dates = pd.concat([train["date"], val["date"], test["date"]])
        assert all_dates.nunique() == len(all_dates), "Duplicate dates across subsets"

    def test_filter_subset_invalid(self):
        panel = _make_panel()
        split = get_fixed_split()
        with pytest.raises(ValueError):
            filter_split(panel, split, "bogus")


class TestWalkForwardSplits:
    def test_no_overlap_assertion_passes(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        assert len(splits) > 0
        assert_no_overlap(splits)

    def test_all_train_start_from_train_start(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        for s in splits:
            assert s.train_start == pd.Timestamp(TRAIN_START)

    def test_train_ends_progress(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        ends = [s.train_end for s in splits]
        assert ends == sorted(ends), "Walk-forward train ends not monotonically increasing"

    def test_val_within_val_period(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        for s in splits:
            assert s.val_start >= pd.Timestamp(TRAIN_START)
            assert s.val_end <= pd.Timestamp(VAL_END)

    def test_test_fixed_across_splits(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        fixed = get_fixed_split()
        for s in splits:
            # test_start is purge-adjusted from val_end, so compare against the fixed split
            assert s.test_start == fixed.test_start
            assert s.test_end == pd.Timestamp(TEST_END)

    def test_purge_gap_all_splits(self):
        panel = _make_panel()
        splits = get_walk_forward_splits(panel)
        for s in splits:
            gap = (s.val_start - s.train_end).days
            assert gap >= PURGE_DAYS, f"Split {s.name} purge gap {gap} < {PURGE_DAYS}"
