"""Step 6 (reverse additions). Experiment/validation script k20rf.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    {R}/valid_v15_assigned.parquet
    {R}/valid_v11_assigned.parquet
    {R}/valid_v11_scores.parquet
    {H}/truth.parquet
    $REPO_DIR/data/candidates/train.parquet
    {W}/k20_pn.parquet
    {W}/k20_slice.parquet
    {W}/k20_extra.parquet
    {W}/k20_rf.parquet

Outputs:
    {N}/train_s2.parquet
    {N}/train_s3.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import shutil, duckdb
S = f'{WORK_DIR}'
H = f'{S}/wf/hunt-odd-one-out-model'; R = f'{S}/rescue'; W = f'{S}/top50/reverse'
N = f'{REPO_DIR}/data/norm'
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp_k20rf'")
c.execute(f"""CREATE TABLE asg AS
    SELECT rid FROM '{R}/valid_v15_assigned.parquet' UNION SELECT rid FROM '{R}/valid_v11_assigned.parquet'
    UNION SELECT rid FROM '{R}/valid_v11_scores.parquet' GROUP BY rid HAVING max(p2) >= 0.85
    UNION SELECT t.rid FROM '{H}/truth.parquet' t SEMI JOIN (SELECT rid, s1 FROM read_parquet('{REPO_DIR}/data/candidates/train.parquet')) k USING (rid, s1)
        WHERE t.rid NOT IN (SELECT rid FROM '{R}/valid_v11_scores.parquet')""")
c.execute(f"""COPY (SELECT country, coalesce(name_core, '') AS nc, count(*) AS npool FROM read_parquet(['{N}/train_s2.parquet', '{N}/train_s3.parquet'])
    WHERE country IN ('us', 'india') AND {ID.format(c='entity_id')} NOT IN (SELECT rid FROM asg) GROUP BY ALL) TO '{W}/k20_pn.parquet' (FORMAT parquet)""")
c.execute(f"CREATE TABLE rids AS SELECT DISTINCT rid FROM read_parquet(['{W}/k20_slice.parquet', '{W}/k20_extra.parquet'])")
c.execute(f"""COPY (SELECT r.rid, coalesce(f.ncand, 0) AS ncand, coalesce(f.fmax, 0) AS fmax FROM rids r LEFT JOIN (SELECT rid, count(*) AS ncand, max(score) AS fmax
    FROM read_parquet('{REPO_DIR}/data/candidates/train.parquet') SEMI JOIN rids USING (rid) GROUP BY rid) f USING (rid)) TO '{W}/k20_rf.parquet' (FORMAT parquet)""")
print(c.execute(f"SELECT count(*) FROM '{W}/k20_rf.parquet'").fetchone(), c.execute(f"SELECT count(*) FROM '{W}/k20_pn.parquet'").fetchone())
c.close(); shutil.rmtree(f'{W}/tmp_k20rf', ignore_errors=True)
