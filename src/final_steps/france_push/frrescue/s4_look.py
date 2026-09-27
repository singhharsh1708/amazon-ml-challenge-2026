"""Step 5 (France push). Experiment/validation script s4_look.py for this step.

Step context: Step 5 (France push). France rescue additions for records left without a match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frrescue
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import duckdb
W = f"{WORK_DIR}/push/frrescue"
c = duckdb.connect(f"{W}/work.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp'")
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
c.execute(f"""create or replace table rawall as
  select {ID} eid, entity_id, business_name rn, business_address ra from raw1 where country ilike 'fr%' or true
  union all select {ID}, entity_id, business_name, business_address from raw2
  union all select {ID}, entity_id, business_name, business_address from raw3""")
print(c.execute("select count(*) from rawall").fetchone())
