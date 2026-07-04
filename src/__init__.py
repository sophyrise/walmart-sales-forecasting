"""Shared code for the Walmart store-sales forecasting project.

Importable both locally and on Kaggle (see config.ON_KAGGLE). Notebooks should
do `from src import ...` after the bootstrap cell puts the repo root on sys.path.
"""
from . import config, data_loader, features, metrics, pipeline, submission, validation

__all__ = [
    "config",
    "data_loader",
    "features",
    "metrics",
    "pipeline",
    "submission",
    "validation",
]
