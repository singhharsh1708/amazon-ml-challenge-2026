"""Shared paths and DuckDB connection for the validation scripts.

Step context: Shared helpers for the validation scripts (DuckDB connection, held-out split loading, F-score).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data
    $REPO_DIR/student_resource/dataset/train/train_ground_truth.tsv
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
D=f'{REPO_DIR}/data'
GT=f'{REPO_DIR}/student_resource/dataset/train/train_ground_truth.tsv'
SCR=f'{WORK_DIR}/wf/hunt-odd-one-out-model'
ID="cast(substr({c},2,1) as bigint)*10000000000+cast(substr({c},4) as bigint)"
def con(mem='1000MB', db=None):
    """Open a DuckDB connection with the memory and thread limits used here."""
    c=duckdb.connect(db) if db else duckdb.connect()
    c.execute(f"SET memory_limit='{mem}'; SET threads=2; SET temp_directory='{SCR}/tmp'; SET max_temp_directory_size='3GB'; SET preserve_insertion_order=false")
    return c
TRUTH=f"""SELECT {ID.format(c='m')} AS rid, {ID.format(c='source1_entity_id')} AS s1 FROM (SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m FROM read_csv('{GT}', delim='\\t', header=true, all_varchar=true, quote='', escape='') WHERE matched_entity_ids <> '')"""
