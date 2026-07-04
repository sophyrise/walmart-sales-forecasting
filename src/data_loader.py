from __future__ import annotations

import pandas as pd

from . import config


def _read_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    )


def load_stores() -> pd.DataFrame:
    return pd.read_csv(config.STORES_CSV)


def load_features() -> pd.DataFrame:
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["Date"])
    df["IsHoliday"] = _read_bool(df["IsHoliday"])
    return df


def load_raw(split: str) -> pd.DataFrame:
    path = config.TRAIN_CSV if split == "train" else config.TEST_CSV
    df = pd.read_csv(path, parse_dates=["Date"])
    df["IsHoliday"] = _read_bool(df["IsHoliday"])
    return df


def load_merged(split: str = "train") -> pd.DataFrame:
    base = load_raw(split)
    stores = load_stores()
    feats = load_features().drop(columns=["IsHoliday"])

    df = base.merge(stores, on="Store", how="left")
    df = df.merge(feats, on=["Store", "Date"], how="left")
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    for split in ("train", "test"):
        d = load_merged(split)
        print(f"[{split}] shape={d.shape}")
        print(d.head(3).to_string())
        print(d.isna().sum().to_string())
        print("-" * 60)
