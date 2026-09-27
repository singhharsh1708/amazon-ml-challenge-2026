"""Selects the uncertain band of stage-2 pairs (validation and test) and writes their text for cross-encoder scoring.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {R}/data/norm/{split}_s1.parquet
    {S}/runs/new_slice_oof.parquet
    {CE}/valid_band.parquet
    {CE}/test_band.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
R = f'{REPO_DIR}'
CE = f'{WORK_DIR}/ce'
S = f'{WORK_DIR}/wf/hunt-odd-one-out-model'
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute("SET memory_limit='2GB'; SET threads=2; SET preserve_insertion_order=false")
TXT = "coalesce({a}.name_full, '') || ' | ' || coalesce({a}.address, '')"
def text(split, pairs, out):
    """Write the query/candidate text of the given pairs to a parquet file."""
    c.execute(f"""COPY (SELECT p.*, {TXT.format(a='q')} AS q_text, {TXT.format(a='s')} AS s_text, q.country
        FROM ({pairs}) p
        JOIN (SELECT {ID.format(c='entity_id')} AS eid, name_full, address, country FROM read_parquet('{R}/data/norm/{split}_s[23].parquet')) q ON q.eid = p.rid
        JOIN (SELECT {ID.format(c='entity_id')} AS eid, name_full, address FROM read_parquet('{R}/data/norm/{split}_s1.parquet')) s ON s.eid = p.s1)
        TO '{out}' (FORMAT parquet)""")
    print(out, c.execute(f"SELECT count(*), count(DISTINCT rid) FROM '{out}'").fetchall(), flush=True)
text('train', f"""SELECT rid, s1, p2 FROM '{S}/runs/new_slice_oof.parquet' WHERE hash(s1) % 20 = 0 AND p2 BETWEEN 0.002 AND 0.998""", f'{CE}/valid_band.parquet')
text('test', f"""SELECT rid, s1, p FROM (SELECT rid, s1, p, row_number() OVER (PARTITION BY rid ORDER BY p DESC) rk FROM read_parquet('{R}/data/v10/full/part*.parquet'))
    WHERE rk <= 2 AND p BETWEEN 0.01 AND 0.99""", f'{CE}/test_band.parquet')
