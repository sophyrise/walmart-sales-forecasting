"""MLflow / DagsHub experiment-tracking setup.

Call `init_tracking()` once at the top of each experiment notebook. It points
MLflow at the DagsHub remote configured in .env. Credentials come from the
environment (MLFLOW_TRACKING_USERNAME / _PASSWORD), never hard-coded.
"""
from __future__ import annotations

import os

import mlflow

from . import config


def init_tracking(experiment: str | None = None) -> str:
    """Configure MLflow against DagsHub and optionally select an experiment.

    Returns the tracking URI in use. Safe to call multiple times.
    """
    uri = config.MLFLOW_TRACKING_URI
    if uri:
        mlflow.set_tracking_uri(uri)
        # MLflow reads these from the environment for HTTP basic auth.
        if config.MLFLOW_TRACKING_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        if config.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
    else:
        # Fall back to a local ./mlruns store if DagsHub is not configured.
        mlflow.set_tracking_uri((config.PROJECT_ROOT / "mlruns").as_uri())

    if experiment:
        mlflow.set_experiment(experiment)

    return mlflow.get_tracking_uri()
