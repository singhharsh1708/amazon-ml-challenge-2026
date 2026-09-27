"""France re-run step 2: blocking of France candidate pairs on the fixed normalization.

Step context: Step 4 input (France re-run). Re-runs the France slice of the pipeline with the fixed address normalization and writes france_assign_partial.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    }/cand.parquet
    {R}/data/norm/test_s1.parquet
"""
import shutil
import sys

from frc import F, R, con, normdir, work, Timer
import block_candidates as B

CHUNKS = 8


def main(variant):
    """Parse the command line and run the step."""
    T = Timer()
    nd = normdir(variant)
    out = f"{work(variant)}/cand.parquet"
    db = f"{F}/tmp/block_{variant}.duckdb"
    c = con(f"block_{variant}", db)
    B.NORM_DIR = type(B.NORM_DIR)(nd)
    n1 = c.execute(f"SELECT count(*) FROM read_parquet('{R}/data/norm/test_s1.parquet')").fetchone()[0]
    c.execute(f"CREATE OR REPLACE TABLE s1_keys AS {B.keys_sql('test', 1)}")
    c.execute(f"""
        CREATE OR REPLACE TABLE s1_index AS
        WITH df AS (SELECT h, unigram, count(*) AS df FROM s1_keys GROUP BY h, unigram)
        SELECT k.h, k.eid, ln({n1} / df.df) AS idf
        FROM s1_keys k JOIN df USING (h, unigram)
        WHERE df.df <= CASE WHEN df.unigram THEN {B.UNIGRAM_CAP} ELSE {B.BIGRAM_CAP} END
    """)
    c.execute("DROP TABLE s1_keys")
    T(f"{variant}: s1 index {c.execute('SELECT count(*) FROM s1_index').fetchone()[0]:,} postings, n1 {n1:,}")
    c.execute("CREATE OR REPLACE TABLE cand (rid BIGINT, s1 BIGINT, score DOUBLE, nkeys BIGINT, rk BIGINT)")
    for chunk in range(CHUNKS):
        where = f"WHERE hash(entity_id) % {CHUNKS} = {chunk}"
        c.execute(f"""
            INSERT INTO cand
            SELECT rid, s1, score, nkeys,
                row_number() OVER (PARTITION BY rid ORDER BY score DESC, s1) AS rk
            FROM (
                SELECT q.eid AS rid, s.eid AS s1, sum(s.idf) AS score, count(*) AS nkeys
                FROM (({B.keys_sql('test', 2, where)}) UNION ALL ({B.keys_sql('test', 3, where)})) q
                JOIN s1_index s ON q.h = s.h
                GROUP BY q.eid, s.eid
            )
            QUALIFY rk <= {B.TOP_K} AND score >= {B.SCORE_RATIO} * max(score) OVER (PARTITION BY rid)
        """)
        T(f"{variant}: chunk {chunk + 1}/{CHUNKS}")
    c.execute(f"COPY cand TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    T(f"{variant}: " + str(c.execute("SELECT count(*), count(DISTINCT rid), count(DISTINCT s1) FROM cand").fetchone()))
    c.close()
    import os
    for p in (db, db + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(f"{F}/tmp/block_{variant}", ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
