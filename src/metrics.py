"""Competition metric: Weighted Mean Absolute Error (WMAE).

Holiday weeks are weighted 5x, all other weeks 1x:

    WMAE = (1 / sum(w_i)) * sum( w_i * |y_i - yhat_i| )

where w_i = 5 if the week is a holiday week, else 1.
"""
from __future__ import annotations

import numpy as np

from .config import HOLIDAY_WEIGHT, NON_HOLIDAY_WEIGHT


def weights_from_holiday(is_holiday) -> np.ndarray:
    """Map a boolean-like IsHoliday array to competition sample weights."""
    is_holiday = np.asarray(is_holiday).astype(bool)
    return np.where(is_holiday, HOLIDAY_WEIGHT, NON_HOLIDAY_WEIGHT).astype(float)


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted Mean Absolute Error used by the Kaggle leaderboard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = weights_from_holiday(is_holiday)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def wmae_from_weights(y_true, y_pred, sample_weight) -> float:
    """WMAE when weights are already computed (e.g. inside CV folds)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))
