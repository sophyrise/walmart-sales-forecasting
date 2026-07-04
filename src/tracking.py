from __future__ import annotations

import os

import mlflow

from . import config


def init_tracking(experiment: str | None = None) -> str:
    uri = config.MLFLOW_TRACKING_URI
    if uri:
        mlflow.set_tracking_uri(uri)
        if config.MLFLOW_TRACKING_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        if config.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
    else:
        mlflow.set_tracking_uri((config.PROJECT_ROOT / "mlruns").as_uri())

    if experiment:
        mlflow.set_experiment(experiment)

    return mlflow.get_tracking_uri()
