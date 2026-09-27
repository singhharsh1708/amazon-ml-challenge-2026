"""Applies step 3 to the test predictions. Usage: high_apply.py <pred_in.parquet> <pred_out.parquet>.

Step context: Step 3 (high-confidence recheck). US/India top-1 pairs with stage-2 p in (0.99, 0.999] are rescored by the ce1+ce2 stacker and p is replaced by least(p, stacked).

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/valid_band.parquet
    {CE}/valid_ce1.parquet
    {CE}/valid_ce2.parquet
    {D}/features/valid.parquet
    {D}/features/valid_predictions.parquet
    {H}/truth.parquet
    {CE}/test_high.parquet
    {CE}/test_high_ce1.parquet
    {CE}/test_high_ce2.parquet
    {D}/features/test_predictions.parquet

Outputs:
    {CE}/high_stacker_ce12.txt
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb, duckdb
R = f'{REPO_DIR}'; D = f'{R}/data'
CE = f'{WORK_DIR}/ce'
H = f'{WORK_DIR}/wf/hunt-odd-one-out-model'
pred_in, pred_out = sys.argv[1], sys.argv[2]
c = duckdb.connect(); c.execute("SET memory_limit='2500MB'; SET threads=3; SET preserve_insertion_order=false")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
v = c.execute(f"""SELECT b.p2, e0.ce AS ce_0, e1.ce AS ce_1, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, (t.s1 IS NOT NULL)::int AS y
    FROM '{CE}/valid_band.parquet' b JOIN '{CE}/valid_ce1.parquet' e0 USING (rid, s1) JOIN '{CE}/valid_ce2.parquet' e1 USING (rid, s1)
    JOIN (SELECT rid, s1, p, address_missing, source, rk FROM read_parquet('{D}/features/valid.parquet') AS vf JOIN read_parquet('{D}/features/valid_predictions.parquet') AS vp USING (rid, s1)) f USING (rid, s1)
    LEFT JOIN '{H}/truth.parquet' t ON t.rid = b.rid AND t.s1 = b.s1""").fetchnumpy()
M = lambda d: np.column_stack([lg(d['p2']), d['ce_0'], d['ce_1'], lg(d['p1']), d['am'], d['source'], d['rk']]).astype(np.float32)
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=3, seed=1)
m = lgb.train(params, lgb.Dataset(M(v), v['y']), 300)
m.save_model(f'{CE}/high_stacker_ce12.txt')
t = c.execute(f"""SELECT h.rid, h.s1, h.p AS p2, e0.ce AS ce_0, e1.ce AS ce_1, f.p AS p1, f.address_missing::int AS am, f.source, f.rk
    FROM '{CE}/test_high.parquet' h JOIN '{CE}/test_high_ce1.parquet' e0 USING (rid, s1) JOIN '{CE}/test_high_ce2.parquet' e1 USING (rid, s1)
    JOIN read_parquet('{D}/features/test_predictions.parquet') f USING (rid, s1)""").fetchnumpy()
ps = m.predict(M(t)).astype(np.float32)
c.register('hs', pa.table({'rid': t['rid'], 's1': t['s1'], 'ps': ps}))
c.execute(f"""COPY (SELECT p.rid, p.s1, (CASE WHEN hs.ps IS NOT NULL THEN least(p.p, hs.ps) ELSE p.p END)::float AS p, p.p_guard
    FROM read_parquet('{pred_in}') p LEFT JOIN hs USING (rid, s1)) TO '{pred_out}' (FORMAT parquet, COMPRESSION zstd)""")
print(f"high pairs {len(ps):,}; below 0.85 after recheck {int((ps < 0.85).sum()):,} (US/India); rows {c.execute(f'SELECT count(*) FROM read_parquet(\'{pred_out}\')').fetchone()[0]:,}")
