"""Purged walk-forward cross-validation splits."""

from dataclasses import dataclass

import pandas as pd

from src.config import (
    PURGE_DAYS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    WF_STEP_MONTHS,
)


@dataclass
class Split:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    # test is always fixed; included for reference
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def get_fixed_split() -> Split:
    """Return the single fixed train/val/test split defined in CLAUDE.md."""
    train_end = pd.Timestamp(TRAIN_END)
    val_start = train_end + pd.offsets.BDay(PURGE_DAYS)
    val_end = pd.Timestamp(VAL_END)
    test_start = val_end + pd.offsets.BDay(PURGE_DAYS)

    return Split(
        name="fixed",
        train_start=pd.Timestamp(TRAIN_START),
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=pd.Timestamp(TEST_END),
    )


def get_walk_forward_splits(
    panel: pd.DataFrame,
    step_months: int = WF_STEP_MONTHS,
    purge_days: int = PURGE_DAYS,
) -> list[Split]:
    """Generate purged walk-forward splits stepping through the validation + test period.

    Each split expands the training window by step_months and uses the next step_months
    as the validation window. The test period is always held out.

    The purge gap between train end and val start prevents target leakage
    (the 5-day forward RV target computed at the last training date overlaps
    with the first val dates).
    """
    dates = pd.to_datetime(panel["date"]).sort_values().unique()
    fixed = get_fixed_split()

    # Walk-forward steps start from TRAIN_END, advance through VAL_END
    steps = pd.date_range(
        start=fixed.train_end,
        end=fixed.val_end,
        freq=pd.DateOffset(months=step_months),
    )

    splits = []
    for i, step_train_end in enumerate(steps[:-1]):
        step_val_start = step_train_end + pd.offsets.BDay(purge_days)
        step_val_end = steps[i + 1]

        # Skip if val window has no dates
        val_dates = dates[(dates >= step_val_start) & (dates <= step_val_end)]
        if len(val_dates) == 0:
            continue

        splits.append(
            Split(
                name=f"wf_{step_train_end.strftime('%Y%m')}",
                train_start=fixed.train_start,
                train_end=step_train_end,
                val_start=step_val_start,
                val_end=step_val_end,
                test_start=fixed.test_start,
                test_end=fixed.test_end,
            )
        )

    return splits


def filter_split(panel: pd.DataFrame, split: Split, subset: str) -> pd.DataFrame:
    """Return rows of panel belonging to train, val, or test subset of a split."""
    dates = pd.to_datetime(panel["date"])
    if subset == "train":
        mask = (dates >= split.train_start) & (dates <= split.train_end)
    elif subset == "val":
        mask = (dates >= split.val_start) & (dates <= split.val_end)
    elif subset == "test":
        mask = (dates >= split.test_start) & (dates <= split.test_end)
    else:
        raise ValueError(f"subset must be 'train', 'val', or 'test', got '{subset}'")
    return panel[mask].copy()


def assert_no_overlap(splits: list[Split]) -> None:
    """Raise AssertionError if any val/test periods overlap with training data."""
    for s in splits:
        # train_end must be strictly before val_start (by at least purge_days)
        gap = (s.val_start - s.train_end).days
        assert gap >= PURGE_DAYS, (
            f"Split {s.name}: purge gap {gap} < {PURGE_DAYS} days "
            f"(train_end={s.train_end.date()}, val_start={s.val_start.date()})"
        )
        # val must not touch test
        assert s.val_end < s.test_start, (
            f"Split {s.name}: val_end {s.val_end.date()} >= test_start {s.test_start.date()}"
        )
