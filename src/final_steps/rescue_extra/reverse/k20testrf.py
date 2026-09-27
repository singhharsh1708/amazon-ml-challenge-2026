"""Step 6 (reverse additions). Experiment/validation script k20testrf.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    $REPO_DIR/output/v18/matching_results_v18.tsv
    {W}/k20_test_pn.parquet
    {W}/k20_test_sel.parquet
    $REPO_DIR/data/candidates/test.parquet
    {W}/k20_test_rf.parquet

Outputs:
    {N}/test_s2.parquet
    {N}/test_s3.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import shutil, duckdb
S = f'{WORK_DIR}'
W = f'{S}/top50/reverse'
N = f'{REPO_DIR}/data/norm'
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp_k20trf'")
c.execute(f"""CREATE TABLE asg AS SELECT DISTINCT {ID.format(c='m')} AS rid FROM (SELECT trim(unnest(string_split(matched_entity_ids, ','))) AS m
    FROM read_csv('{REPO_DIR}/output/v18/matching_results_v18.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '') WHERE m <> ''""")
c.execute(f"""COPY (SELECT country, coalesce(name_core, '') AS nc, count(*) AS npool FROM read_parquet(['{N}/test_s2.parquet', '{N}/test_s3.parquet'])
    WHERE country IN ('us', 'india') AND {ID.format(c='entity_id')} NOT IN (SELECT rid FROM asg) GROUP BY ALL) TO '{W}/k20_test_pn.parquet' (FORMAT parquet)""")
c.execute(f"CREATE TABLE rids AS SELECT DISTINCT rid FROM '{W}/k20_test_sel.parquet'")
c.execute(f"""COPY (SELECT r.rid, coalesce(f.ncand, 0) AS ncand, coalesce(f.fmax, 0) AS fmax FROM rids r LEFT JOIN (SELECT rid, count(*) AS ncand, max(score) AS fmax
    FROM read_parquet('{REPO_DIR}/data/candidates/test.parquet') SEMI JOIN rids USING (rid) GROUP BY rid) f USING (rid)) TO '{W}/k20_test_rf.parquet' (FORMAT parquet)""")
print(c.execute(f"SELECT count(*), avg(ncand) FROM '{W}/k20_test_rf.parquet'").fetchone())
c.close(); shutil.rmtree(f'{W}/tmp_k20trf', ignore_errors=True)
