"""Step 6 (name-key rescue). Experiment/validation script q.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb, sys
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb')
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 500)
for q in sys.stdin.read().split(';;'):
    q = q.strip()
    if not q: continue
    r = c.execute(q)
    if r.description: print(r.df().to_string()); print()
