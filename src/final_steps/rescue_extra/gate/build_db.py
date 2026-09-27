"""Step 6 gate. Experiment/validation script build_db.py for this step.

Step context: Step 6 gate. Validation of the combined rescue families on the held-out split and the final nm1d addition set (nm1d_adds.parquet).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    {H}/valid_s1.parquet
    {H}/truth.parquet
    {E}/valid_v15_assigned_full.parquet
    {E}/valid_rbest.parquet
    {N}/train_s1.parquet
    {N}/train_s2.parquet
    {N}/train_s3.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
S = f'{WORK_DIR}'
H = f'{S}/wf/hunt-odd-one-out-model'
E = f'{S}/top50/gate'
N = f'{REPO_DIR}/data/norm'
ID = "cast(substr({c},2,1) as bigint)*10000000000+cast(substr({c},4) as bigint)"
c = duckdb.connect(f'{E}/err.duckdb')
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
c.execute(f"CREATE OR REPLACE TABLE vs1 AS SELECT * FROM '{H}/valid_s1.parquet'")
c.execute(f"CREATE OR REPLACE TABLE truth AS SELECT * FROM '{H}/truth.parquet'")
c.execute(f"CREATE OR REPLACE TABLE vtruth AS SELECT t.* FROM truth t JOIN vs1 USING (s1)")
c.execute(f"""CREATE OR REPLACE TABLE fa AS
  SELECT a.rid, a.s1, a.p, a.p1, a.fne, 'base' AS src FROM '{E}/valid_v15_assigned_full.parquet' a
  UNION ALL SELECT r.rid, r.s1, r.ps, NULL, NULL, 'rescue' FROM '{E}/valid_rbest.parquet' r WHERE r.ps >= 0.7""")
c.execute("CREATE OR REPLACE TABLE fa AS SELECT f.*, t.s1 AS ts FROM fa f LEFT JOIN truth t USING (rid)")
c.execute("CREATE OR REPLACE TABLE rids AS SELECT rid FROM fa UNION SELECT rid FROM vtruth")
c.execute(f"CREATE OR REPLACE TABLE s1txt AS SELECT {ID.format(c='entity_id')} AS s1, country, name_full, name_core, address, address_missing FROM '{N}/train_s1.parquet'")
c.execute(f"""CREATE OR REPLACE TABLE rtxt AS SELECT * FROM (
   SELECT {ID.format(c='entity_id')} AS rid, country, name_full, name_core, address, address_missing FROM read_parquet(['{N}/train_s2.parquet','{N}/train_s3.parquet'])) x
   SEMI JOIN rids USING (rid)""")
for t in ['vs1','vtruth','fa','rids','s1txt','rtxt']:
    print(t, c.execute(f"select count(*) from {t}").fetchone())
print(c.execute("""SELECT src, count(*), count(*) FILTER (WHERE ts=s1) tp, count(*) FILTER (WHERE ts IS NULL) fpd, count(*) FILTER (WHERE ts<>s1) fpw FROM fa GROUP BY src""").fetchall())
