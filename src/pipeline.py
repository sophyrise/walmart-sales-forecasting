from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import data_loader, features


class WalmartPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stores: pd.DataFrame | None = None, feats: pd.DataFrame | None = None):
        self.stores = stores if stores is not None else data_loader.load_stores()
        self.feats = feats if feats is not None else data_loader.load_features().drop(columns=["IsHoliday"])
        self.feature_names_: list[str] | None = None

    def _merge(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Temperature" in df.columns:
            return df.copy()
        out = df.merge(self.stores, on="Store", how="left")
        out = out.merge(self.feats, on=["Store", "Date"], how="left")
        return out

    def _transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = self._merge(df)
        return features.build_features(merged, add_lags=False)

    def fit(self, X: pd.DataFrame, y=None):
        built = self._transform_frame(X)
        self.feature_names_ = features.feature_columns(built)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("WalmartPreprocessor must be fitted before transform.")
        built = self._transform_frame(X)
        for col in self.feature_names_:
            if col not in built.columns:
                built[col] = 0
        return built[self.feature_names_].fillna(0)


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", WalmartPreprocessor()), ("model", model)])
