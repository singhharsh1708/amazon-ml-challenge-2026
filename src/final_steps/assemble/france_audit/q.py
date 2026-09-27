"""Shared DuckDB connection and display helpers for the France audit scripts.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/france_audit/judge
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb, sys
J=f'{WORK_DIR}/france_audit/judge'
S=f'{WORK_DIR}'
D=S+'/france_audit/decide'
N=S+'/france_audit/norm'
R=f'{REPO_DIR}'
ID="(cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint))"
def con():
    """Open a DuckDB connection with the memory and thread limits used here."""
    c=duckdb.connect()
    c.execute("SET memory_limit='800MB'; SET threads=2; SET preserve_insertion_order=false;")
    c.execute(f"SET temp_directory='{J}/tmp'")
    return c
def show(c, q, n=60):
    """Print the first rows of a query result."""
    print(c.execute(q).fetchdf().head(n).to_string(max_colwidth=90)); sys.stdout.flush()
