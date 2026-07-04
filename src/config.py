"""Central project configuration: paths, constants, and environment loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment detection (local vs Kaggle)
# ---------------------------------------------------------------------------
# On Kaggle the competition data lives under /kaggle/input/<slug>/ and files may
# be gzip-zipped (train.csv.zip). Locally they live in data/raw/ as plain CSVs.
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_COMPETITION = "walmart-recruiting-store-sales-forecasting"
ON_KAGGLE = KAGGLE_INPUT.exists()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
# Writable scratch: /kaggle/working on Kaggle, project dir locally.
WORKING_DIR = Path("/kaggle/working") if ON_KAGGLE else PROJECT_ROOT
PROCESSED_DIR = WORKING_DIR / "data" / "processed"
MODELS_DIR = WORKING_DIR / "models"

for _d in (PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if ON_KAGGLE:
    RAW_DIR = KAGGLE_INPUT / KAGGLE_COMPETITION
else:
    RAW_DIR = DATA_DIR / "raw"


def _resolve(stem: str) -> Path:
    """Find a raw file, tolerating the .csv / .csv.zip split between envs."""
    for name in (f"{stem}.csv", f"{stem}.csv.zip"):
        p = RAW_DIR / name
        if p.exists():
            return p
    # Default to plain .csv path even if missing (clearer error downstream).
    return RAW_DIR / f"{stem}.csv"


# Raw competition files (resolved per environment)
TRAIN_CSV = _resolve("train")
TEST_CSV = _resolve("test")
FEATURES_CSV = _resolve("features")
STORES_CSV = _resolve("stores")
SAMPLE_SUBMISSION_CSV = _resolve("sampleSubmission")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET = "Weekly_Sales"

# The four "special" holiday weeks are worth 5x in the competition metric (WMAE).
HOLIDAY_WEIGHT = 5
NON_HOLIDAY_WEIGHT = 1

# Submission Id format is "{Store}_{Dept}_{Date}".
ID_COLS = ["Store", "Dept", "Date"]

# ---------------------------------------------------------------------------
# Environment / experiment tracking
# ---------------------------------------------------------------------------
# Credential resolution order:
#   1. Kaggle Secrets (when running on Kaggle) — add them under
#      "Add-ons > Secrets" with the same names as the .env keys.
#   2. .env file / process environment (local).
_SECRET_KEYS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def _load_kaggle_secrets() -> None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        for key in _SECRET_KEYS:
            try:
                os.environ.setdefault(key, client.get_secret(key))
            except Exception:
                pass  # secret not set; fall through to whatever env has
    except Exception:
        pass  # not on Kaggle or add-on unavailable


if ON_KAGGLE:
    _load_kaggle_secrets()
else:
    load_dotenv(PROJECT_ROOT / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


def dagshub_repo() -> tuple[str, str] | None:
    """Parse (owner, repo) from the DagsHub MLflow tracking URI, if configured."""
    if not MLFLOW_TRACKING_URI or "dagshub.com" not in MLFLOW_TRACKING_URI:
        return None
    # https://dagshub.com/<owner>/<repo>.mlflow
    tail = MLFLOW_TRACKING_URI.split("dagshub.com/", 1)[1]
    owner, repo = tail.split("/", 1)
    repo = repo.replace(".mlflow", "").strip("/")
    return owner, repo
