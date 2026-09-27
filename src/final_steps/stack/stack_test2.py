"""Fits the stacker on the validation band (logit p2, CE logits, logit p1, address_missing, source, rk) and applies it to the test band. Usage: stack_test2.py <valid_ce files,comma separated> <test_ce files,comma separated> <out.parquet>, run with the cross-encoder score files in the working directory.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1], argv[2], argv[3].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/valid_band.parquet
    {D}/features/valid.parquet
    {D}/features/valid_predictions.parquet
    {S}/truth.parquet
    {CE}/test_band.parquet
    {D}/features/test_predictions.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb, duckdb
R = f'{REPO_DIR}'; D = f'{R}/data'
CE = f'{WORK_DIR}/ce'
S = f'{WORK_DIR}/wf/hunt-odd-one-out-model'
valid_ces, test_ces, out = sys.argv[1].split(','), sys.argv[2].split(','), sys.argv[3]
c = duckdb.connect(); c.execute("SET memory_limit='2GB'; SET threads=3")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
vsel = ", ".join(f"e{i}.ce AS ce_{i}" for i in range(len(valid_ces)))
vjoin = " ".join(f"JOIN '{f}' e{i} USING (rid, s1)" for i, f in enumerate(valid_ces))
tsel = ", ".join(f"e{i}.ce AS ce_{i}" for i in range(len(test_ces)))
tjoin = " ".join(f"JOIN '{f}' e{i} USING (rid, s1)" for i, f in enumerate(test_ces))
v = c.execute(f"""SELECT b.p2, {vsel}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, (t.s1 IS NOT NULL)::int AS y
    FROM '{CE}/valid_band.parquet' b {vjoin}
    JOIN (SELECT rid, s1, p, address_missing, source, rk FROM read_parquet('{D}/features/valid.parquet') AS vf JOIN read_parquet('{D}/features/valid_predictions.parquet') AS vp USING (rid, s1)) f USING (rid, s1)
    LEFT JOIN '{S}/truth.parquet' t ON t.rid = b.rid AND t.s1 = b.s1""").fetchnumpy()
X = np.column_stack([lg(v['p2'])] + [v[f'ce_{i}'] for i in range(len(valid_ces))] + [lg(v['p1']), v['am'], v['source'], v['rk']]).astype(np.float32)
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=4, seed=1)
m = lgb.train(params, lgb.Dataset(X, v['y']), 300)
m.save_model(out.replace('.parquet', '_stacker.txt'))
t = c.execute(f"""SELECT b.rid, b.s1, b.p AS p2, {tsel}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk
    FROM '{CE}/test_band.parquet' b {tjoin}
    JOIN read_parquet('{D}/features/test_predictions.parquet') f USING (rid, s1)""").fetchnumpy()
Xt = np.column_stack([lg(t['p2'])] + [t[f'ce_{i}'] for i in range(len(test_ces))] + [lg(t['p1']), t['am'], t['source'], t['rk']]).astype(np.float32)
ps = m.predict(Xt).astype(np.float32)
pq.write_table(pa.table({'rid': t['rid'], 's1': t['s1'], 'ps': ps}), out)
print(f"stacked {len(ps):,} test band pairs; mean p2 {t['p2'].mean():.3f} -> ps {ps.mean():.3f}; crossing 0.85 up {int(((t['p2'] < 0.85) & (ps >= 0.85)).sum()):,} down {int(((t['p2'] >= 0.85) & (ps < 0.85)).sum()):,}")
