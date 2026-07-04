from __future__ import annotations

import numpy as np
import pandas as pd


def _iso_week(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.isocalendar().week.astype(int)


class SeasonalNaiveForecaster:
    def __init__(self, target: str = "Weekly_Sales"):
        self.target = target
        self.by_store_dept_week_: dict | None = None
        self.by_store_dept_: dict | None = None
        self.global_mean_: float | None = None

    def fit(self, df: pd.DataFrame, y=None) -> "SeasonalNaiveForecaster":
        d = df.copy()
        d["_week"] = _iso_week(d["Date"])
        self.by_store_dept_week_ = (
            d.groupby(["Store", "Dept", "_week"])[self.target].mean().to_dict()
        )
        self.by_store_dept_ = d.groupby(["Store", "Dept"])[self.target].mean().to_dict()
        self.global_mean_ = float(d[self.target].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.by_store_dept_week_ is None:
            raise RuntimeError("SeasonalNaiveForecaster must be fitted before predict.")
        weeks = _iso_week(df["Date"]).to_numpy()
        stores = df["Store"].to_numpy()
        depts = df["Dept"].to_numpy()
        out = np.empty(len(df), dtype=float)
        for i in range(len(df)):
            key3 = (stores[i], depts[i], weeks[i])
            if key3 in self.by_store_dept_week_:
                out[i] = self.by_store_dept_week_[key3]
                continue
            key2 = (stores[i], depts[i])
            out[i] = self.by_store_dept_.get(key2, self.global_mean_)
        return out
