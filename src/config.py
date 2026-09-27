"""Shared paths for the whole pipeline.

Every other script imports its directories from here. The challenge data root is read from
the AMAZON_ML_DATA_ROOT environment variable and defaults to <repo>/student_resource. Derived
data goes to <repo>/data, trained models to <repo>/models and submission files to <repo>/output.
Importing this module creates the output and models directories if they are missing.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    os.environ.get("AMAZON_ML_DATA_ROOT", PROJECT_DIR / "student_resource")
)

TRAIN_DIR = DATA_ROOT / "dataset" / "train"
TEST_DIR = DATA_ROOT / "dataset" / "test"

DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "entity_resolution.duckdb"
TEMP_DIR = DATA_DIR / "temp"

OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
