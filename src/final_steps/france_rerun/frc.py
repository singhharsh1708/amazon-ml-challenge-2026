"""Shared settings and helpers for the France re-run (paths from FR_RERUN_DIR, REPO_DIR, WORK_DIR; DuckDB connection; timers).

Step context: Step 4 input (France re-run). Re-runs the France slice of the pipeline with the fixed address normalization and writes france_assign_partial.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/student_resource/dataset/test/test_source{s}.tsv
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
import sys
import time

import duckdb

F = os.environ.get("FR_RERUN_DIR", "work/fr_rerun")
R = os.environ.get("REPO_DIR", ".")
SP = os.environ.get("WORK_DIR", "work")
sys.path.insert(0, f"{REPO_DIR}/src"); sys.path.insert(0, f"{FINAL_STEPS}/france_rerun/src")
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
FIXES = {"region", "ndeg", "suffix", "types"}


def con(tmp_name="main", db=None):
    """Open a DuckDB connection with the memory and thread limits used here."""
    c = duckdb.connect(db or ":memory:")
    c.execute("SET memory_limit='2500MB'; SET threads=3; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    c.execute(f"SET temp_directory='{F}/tmp/{tmp_name}'; SET max_temp_directory_size='8GB'")
    return c


def raw(s):
    """Return SQL that reads a raw test source file."""
    return (f"read_csv('{R}/student_resource/dataset/test/test_source{s}.tsv', delim='\\t', header=true, "
            "all_varchar=true, quote='', escape='')")


def normdir(variant):
    """Return the normalized-data directory of a variant."""
    return f"{F}/data/fr_{variant}"


def work(variant):
    """Return the work directory of a variant."""
    return f"{F}/work/{variant}"


class Timer:
    """Cross-encoder: transformer encoder with a linear scoring head."""
    def __init__(self):
        """Build the encoder and the scoring head."""
        self.t = time.time()

    def __call__(self, msg):
        """Print a message with the elapsed time."""
        print(f"[{time.time() - self.t:7.1f}s] {msg}", flush=True)
