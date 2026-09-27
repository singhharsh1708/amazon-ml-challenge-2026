"""Writes the full test predictions parquet: stage-2 scores with the stacked band scores substituted, and the France least(p, guard) rule. Usage: mkpred2.py <out_pred.parquet> <stacked.parquet>.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/data/features/test_predictions_stage2.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, duckdb
R = f'{REPO_DIR}'
out, stacked = sys.argv[1], sys.argv[2]
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{R}/data/temp/mkpred'")
c.execute(f"CREATE TABLE fr AS SELECT {ID} AS rid FROM read_parquet('{R}/data/norm/test_s[23].parquet') WHERE country = 'france'")
c.execute(f"""COPY (SELECT n.rid, n.s1,
      (CASE WHEN f.rid IS NOT NULL THEN least(n.p, coalesce(st.ps, n.p)) ELSE coalesce(st.ps, n.p) END)::float AS p, g.p::float AS p_guard
    FROM read_parquet('{R}/data/v10/full/part*.parquet') n
    JOIN read_parquet('{R}/data/features/test_predictions_stage2.parquet') g USING (rid, s1)
    LEFT JOIN read_parquet('{stacked}') st USING (rid, s1) LEFT JOIN fr f ON f.rid = n.rid)
    TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
print(out, c.execute(f"SELECT count(*), avg(p) FROM '{out}'").fetchall())
