"""Step 6 gate. Experiment/validation script check_test.py for this step.

Step context: Step 6 gate. Validation of the combined rescue families on the held-out split and the final nm1d addition set (nm1d_adds.parquet).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    $REPO_DIR/output/v18/matching_results_v18.tsv
    {N}/test_s1.parquet
    {N}/test_s2.parquet
    {N}/test_s3.parquet
    {S}/top50/errors/test_additions.parquet
    {S}/top50/errors/test_additions_conservative.parquet
    {S}/top50/reverse/test_additions.parquet
    {S}/top50/errors/test_removals.parquet
    {S}/rescue/test_rescue_adds.parquet
    {S}/v19b_add.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
S = f'{WORK_DIR}'
G = f'{S}/top50/gate'; N = f'{REPO_DIR}/data/norm'
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
c.execute(f"""CREATE TABLE v18 AS SELECT {ID.format(c='s')} AS s1, {ID.format(c='m')} AS rid FROM (SELECT source1_entity_id s, trim(unnest(string_split(matched_entity_ids, ','))) AS m
    FROM read_csv('{REPO_DIR}/output/v18/matching_results_v18.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '') WHERE m <> ''""")
c.execute(f"CREATE TABLE v18s1 AS SELECT DISTINCT {ID.format(c='source1_entity_id')} s1 FROM read_csv('{REPO_DIR}/output/v18/matching_results_v18.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')")
c.execute(f"CREATE TABLE ts1 AS SELECT {ID.format(c='entity_id')} AS s1, country FROM '{N}/test_s1.parquet'")
c.execute(f"CREATE TABLE trec AS SELECT {ID.format(c='entity_id')} AS rid, country FROM read_parquet(['{N}/test_s2.parquet','{N}/test_s3.parquet'])")
print('v18 pairs', c.execute("select count(*), count(distinct rid) from v18").fetchone(), 'v18 s1 rows', c.execute("select count(*) from v18s1").fetchone())
files = {'errors_D': f'{S}/top50/errors/test_additions.parquet', 'errors_A': f'{S}/top50/errors/test_additions_conservative.parquet',
         'reverse': f'{S}/top50/reverse/test_additions.parquet', 'errors_rem': f'{S}/top50/errors/test_removals.parquet'}
for k, f in files.items():
    print('==', k)
    cols = [r[0] for r in c.execute(f"DESCRIBE SELECT * FROM '{f}'").fetchall()]
    sel = "rid, s1" if 'rid' in cols else f"{ID.format(c='record_id')} AS rid, {ID.format(c='source1_entity_id')} AS s1"
    c.execute(f"CREATE OR REPLACE TABLE a AS SELECT {sel} FROM '{f}'")
    print(' rows, distinct rid, distinct pair', c.execute("SELECT count(*), count(DISTINCT rid), count(DISTINCT (rid, s1)) FROM a").fetchone())
    print(' null', c.execute("SELECT count(*) FROM a WHERE rid IS NULL OR s1 IS NULL").fetchone())
    print(' rid in v18 assigned', c.execute("SELECT count(*) FROM a SEMI JOIN v18 USING (rid)").fetchone()[0])
    print(' pair in v18', c.execute("SELECT count(*) FROM a SEMI JOIN v18 USING (rid, s1)").fetchone()[0])
    print(' s1 in v18 output', c.execute("SELECT count(*) FROM a SEMI JOIN v18s1 USING (s1)").fetchone()[0])
    print(' s1 country', c.execute("SELECT s.country, count(*) FROM a LEFT JOIN ts1 s USING (s1) GROUP BY 1").fetchall())
    print(' rec country', c.execute("SELECT r.country, count(*) FROM a LEFT JOIN trec r USING (rid) GROUP BY 1").fetchall())
    print(' rec country <> s1 country', c.execute("SELECT count(*) FROM a JOIN trec r USING (rid) JOIN ts1 s USING (s1) WHERE r.country <> s.country").fetchone()[0])
    print(' rid in rescue adds', c.execute(f"SELECT count(*) FROM a SEMI JOIN '{S}/rescue/test_rescue_adds.parquet' r USING (rid)").fetchone()[0])
    print(' rid in v19b_add', c.execute(f"SELECT count(*) FROM a SEMI JOIN '{S}/v19b_add.parquet' r USING (rid)").fetchone()[0])
c.execute(f"CREATE TABLE e AS SELECT {ID.format(c='record_id')} AS rid, {ID.format(c='source1_entity_id')} AS s1 FROM '{files['errors_D']}'")
c.execute(f"CREATE TABLE r AS SELECT rid, s1 FROM '{files['reverse']}'")
print('errorsD vs reverse: shared rid', c.execute("SELECT count(*), count(*) FILTER (WHERE e.s1 = r.s1) FROM e JOIN r USING (rid)").fetchone())
print('v19b_add schema', c.execute(f"DESCRIBE SELECT * FROM '{S}/v19b_add.parquet'").fetchall(), c.execute(f"SELECT count(*) FROM '{S}/v19b_add.parquet'").fetchone())
