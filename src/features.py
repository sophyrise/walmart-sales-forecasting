from __future__ import annotations

import numpy as np
import pandas as pd

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["Date"].dt
    df["Year"] = d.year
    df["Month"] = d.month
    df["Week"] = d.isocalendar().week.astype(int)
    df["Day"] = d.day
    df["DayOfYear"] = d.dayofyear
    df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["IsSuperBowl"] = df["Week"].isin([6]).astype(int)
    df["IsLaborDay"] = df["Week"].isin([36]).astype(int)
    df["IsThanksgiving"] = df["Week"].isin([47]).astype(int)
    df["IsChristmas"] = df["Week"].isin([51, 52]).astype(int)
    return df


def clean_markdowns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MARKDOWN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["MarkDown_total"] = df[MARKDOWN_COLS].sum(axis=1)
    df["MarkDown_active"] = (df["MarkDown_total"] > 0).astype(int)
    return df


def encode_store_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Type" in df.columns:
        df["Type_code"] = df["Type"].map({"A": 0, "B": 1, "C": 2}).astype("Int64")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "Weekly_Sales",
    lags: tuple[int, ...] = (1, 2, 3, 4, 52),
    rolling_windows: tuple[int, ...] = (4, 8, 12),
) -> pd.DataFrame:
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    g = df.groupby(["Store", "Dept"])[target]
    for lag in lags:
        df[f"{target}_lag{lag}"] = g.shift(lag)
    for w in rolling_windows:
        df[f"{target}_rollmean{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w).mean()
        )
        df[f"{target}_rollstd{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w).std()
        )
    return df


def build_features(df: pd.DataFrame, add_lags: bool = False) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = clean_markdowns(out)
    out = encode_store_type(out)
    if add_lags and "Weekly_Sales" in out.columns:
        out = add_lag_features(out)
    return out


NON_FEATURE_COLS = ["Store", "Dept", "Date", "Type", "Weekly_Sales"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
