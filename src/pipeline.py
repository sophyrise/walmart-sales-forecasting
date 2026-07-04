"""Scikit-learn preprocessing transformer + Pipeline helpers.

The assignment requires the best model of every architecture to be stored as a
Pipeline that runs *directly on the raw test set* (Store, Dept, Date,
IsHoliday) with no manual preprocessing. `WalmartPreprocessor` therefore carries
the stores / features side-tables and does the whole merge + feature build on
`.transform`, so:

    pipe = build_pipeline(model)            # model is any sklearn regressor
    pipe.fit(raw_train_df, y)               # raw_train_df = load_raw("train")
    preds = pipe.predict(raw_test_df)       # raw_test_df  = load_raw("test")

Target lag features are intentionally NOT part of this pipeline: they need the
target at inference time and make the transform stateful. Models that rely on
lags build them inside their own notebook instead.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import data_loader, features


class WalmartPreprocessor(BaseEstimator, TransformerMixin):
    """Turn raw (Store, Dept, Date, IsHoliday) rows into a numeric feature matrix.

    Holds the stores and weekly-features tables so a fitted pipeline is fully
    self-contained and can score the raw test set. If the input frame is already
    merged (has a 'Temperature' column) the merge step is skipped.
    """

    def __init__(self, stores: pd.DataFrame | None = None, feats: pd.DataFrame | None = None):
        self.stores = stores if stores is not None else data_loader.load_stores()
        self.feats = feats if feats is not None else data_loader.load_features().drop(columns=["IsHoliday"])
        self.feature_names_: list[str] | None = None

    def _merge(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Temperature" in df.columns:  # already merged
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
        # Guarantee identical column set/order as at fit time.
        for col in self.feature_names_:
            if col not in built.columns:
                built[col] = 0
        return built[self.feature_names_].fillna(0)


def build_pipeline(model) -> Pipeline:
    """Compose the raw->features preprocessor with any sklearn-style regressor."""
    return Pipeline([("preprocess", WalmartPreprocessor()), ("model", model)])
