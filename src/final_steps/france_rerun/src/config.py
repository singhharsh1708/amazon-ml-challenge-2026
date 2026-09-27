"""Copy of a src/ module as used by the France re-run (fixed France normalization). Experiment/validation script config.py for this step.

Step context: Copy of a src/ module as used by the France re-run (fixed France normalization). It is put ahead of src/ on sys.path by the France re-run scripts.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/fr_rerun
    $REPO_DIR/student_resource
    $REPO_DIR/models
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
from pathlib import Path

PROJECT_DIR = Path(f"{WORK_DIR}/fr_rerun")

DATA_ROOT = Path(f"{REPO_DIR}/student_resource")

TRAIN_DIR = DATA_ROOT / "dataset" / "train"
TEST_DIR = DATA_ROOT / "dataset" / "test"

DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "entity_resolution.duckdb"
TEMP_DIR = PROJECT_DIR / "tmp"

OUTPUT_DIR = PROJECT_DIR / "out"
MODEL_DIR = Path(f"{REPO_DIR}/models")
