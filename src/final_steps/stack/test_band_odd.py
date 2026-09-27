"""Adds odd-one-out features to the test band.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/test_band.parquet
    {R}/data/features/test_predictions.parquet

Outputs:
    {CE}/test_band_odd.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time
sys.path.insert(0, f'{REPO_DIR}/src')
import duckdb, pyarrow.parquet as pq
from odd_features import feature_table, load_word_lists, sibling_index, sibling_sql
R = f'{REPO_DIR}'
CE = f'{WORK_DIR}/ce'
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
t = time.time()
c = duckdb.connect(); c.execute(f"SET memory_limit='1500MB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{CE}/tmp_odd'")
c.execute(f"CREATE TABLE band AS SELECT rid, s1 FROM '{CE}/test_band.parquet'")
c.execute(f"CREATE TABLE best AS SELECT rid, arg_max(s1, p) AS s1, max(p) AS p FROM read_parquet('{R}/data/features/test_predictions.parquet') GROUP BY rid")
c.execute("CREATE TABLE best5 AS SELECT * FROM best WHERE p >= 0.5 AND s1 IN (SELECT DISTINCT s1 FROM band)")
c.execute(f"""CREATE TABLE names AS SELECT {ID} AS eid, country, name_full, address FROM read_parquet('{R}/data/norm/test_s*.parquet')
    WHERE {ID} IN (SELECT rid FROM band UNION SELECT s1 FROM band UNION SELECT rid FROM best5)""")
print("prep", f"{time.time()-t:.0f}s", flush=True)
idx = sibling_index(c.execute(sibling_sql("best5", "names")).to_arrow_table())
text = c.execute("""SELECT b.rid, b.s1, q.country, q.name_full AS q_name, q.address AS q_addr, s.name_full AS s_name, s.address AS s_addr
    FROM band b JOIN names q ON q.eid = b.rid JOIN names s ON s.eid = b.s1""").to_arrow_table()
dec, rep = load_word_lists()
out = feature_table(text, idx, dec, rep)
pq.write_table(out, f'{CE}/test_band_odd.parquet')
print("odd features", out.num_rows, f"{time.time()-t:.0f}s")
