from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def _sorted_unique_dates(dates: pd.Series) -> np.ndarray:
    return np.sort(pd.to_datetime(dates).unique())


def time_holdout(df: pd.DataFrame, weeks: int = 8, date_col: str = "Date"):
    uniq = _sorted_unique_dates(df[date_col])
    if weeks >= len(uniq):
        raise ValueError(f"weeks={weeks} >= number of distinct weeks {len(uniq)}")
    cutoff = uniq[-weeks]
    d = pd.to_datetime(df[date_col]).to_numpy()
    train_idx = np.where(d < cutoff)[0]
    val_idx = np.where(d >= cutoff)[0]
    return train_idx, val_idx


def expanding_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    horizon: int = 8,
    date_col: str = "Date",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    uniq = _sorted_unique_dates(df[date_col])
    needed = horizon * n_splits
    if needed >= len(uniq):
        raise ValueError(
            f"Need > {needed} distinct weeks for {n_splits} folds of {horizon}; "
            f"have {len(uniq)}."
        )
    d = pd.to_datetime(df[date_col]).to_numpy()
    for k in range(n_splits):
        end_offset = needed - k * horizon
        start_offset = end_offset - horizon
        val_start = uniq[-end_offset]
        val_end = uniq[-start_offset] if start_offset > 0 else None
        train_idx = np.where(d < val_start)[0]
        if val_end is None:
            val_idx = np.where(d >= val_start)[0]
        else:
            val_idx = np.where((d >= val_start) & (d < val_end))[0]
        yield train_idx, val_idx
