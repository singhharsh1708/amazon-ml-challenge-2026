import os
import sys
import time

import duckdb

F = os.environ.get("FR_RERUN_DIR", "work/fr_rerun")
R = os.environ.get("REPO_DIR", ".")
SP = os.environ.get("WORK_DIR", "work")
sys.path.insert(0, f"{F}/src")
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
FIXES = {"region", "ndeg", "suffix", "types"}


def con(tmp_name="main", db=None):
    c = duckdb.connect(db or ":memory:")
    c.execute("SET memory_limit='2500MB'; SET threads=3; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    c.execute(f"SET temp_directory='{F}/tmp/{tmp_name}'; SET max_temp_directory_size='8GB'")
    return c


def raw(s):
    return (f"read_csv('{R}/student_resource/dataset/test/test_source{s}.tsv', delim='\\t', header=true, "
            "all_varchar=true, quote='', escape='')")


def normdir(variant):
    return f"{F}/data/fr_{variant}"


def work(variant):
    return f"{F}/work/{variant}"


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t:7.1f}s] {msg}", flush=True)
