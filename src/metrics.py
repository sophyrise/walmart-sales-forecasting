from __future__ import annotations

import numpy as np

from .config import HOLIDAY_WEIGHT, NON_HOLIDAY_WEIGHT


def weights_from_holiday(is_holiday) -> np.ndarray:
    is_holiday = np.asarray(is_holiday).astype(bool)
    return np.where(is_holiday, HOLIDAY_WEIGHT, NON_HOLIDAY_WEIGHT).astype(float)


def wmae(y_true, y_pred, is_holiday) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = weights_from_holiday(is_holiday)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def wmae_from_weights(y_true, y_pred, sample_weight) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))
