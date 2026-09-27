"""Step 6 (reverse additions). Experiment/validation script k20a.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {W}/train_revf20.parquet

Outputs:
    {W}/k20_stage1.txt
    {W}/k20_slice.parquet
    {W}/k20_extra.parquet
    {W}/k20_ce_in.parquet
    {W}/k20_tf.txt
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import numpy as np, pandas as pd, duckdb, lightgbm as lgb
from sklearn.metrics import roc_auc_score
S = f'{WORK_DIR}'
W = f'{S}/top50/reverse'
FR = 0.01
F = ['score', 'nk', 'rk', 'ns1_q', 'ns1_s', 'nrev', 'rmax', 'smax', 's2nd', 'tsr', 'tsort', 'rcomp', 'exact', 'qaddr', 'atsr', 'aovl', 'anum', 'qlen', 'slen', 'india', 'rel']
c = duckdb.connect(); c.execute("SET threads=3; SET memory_limit='2GB'")
vs = c.execute(f"SELECT s1, hash(s1) % 40 < 20 AS h0 FROM '{S}/wf/hunt-odd-one-out-model/valid_s1.parquet'").df()
d = pd.read_parquet(f'{W}/train_revf20.parquet').merge(vs, on='s1', how='left')
d['slice'] = d.h0.notna()
d['india'] = (d.country == 'india').astype(float); d['rel'] = d.score / d.rmax
sl = d[d.slice].copy(); ex = d[~d.slice].copy()
X = sl[F].values.astype(np.float32); y = sl.y.values; h0 = sl.h0.values.astype(bool)
p = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=30, verbose=-1, num_threads=3, seed=1)
oof = np.zeros(len(y))
for k in (True, False):
    tr = h0 != k
    oof[~tr] = lgb.train(p, lgb.Dataset(X[tr], y[tr]), 300).predict(X[~tr])
print('slice pairs', len(y), 'pos', y.sum(), 'AUC', round(roc_auc_score(y, oof), 4))
for fr in (0.005, 0.01, 0.02, 0.04):
    print(fr, 'pos kept', y[oof >= np.quantile(oof, 1 - fr)].sum())
full = lgb.train(p, lgb.Dataset(X, y), 300); full.save_model(f'{W}/k20_stage1.txt')
sl['p1'] = oof
ex['p1'] = full.predict(ex[F].values.astype(np.float32))
tf = np.quantile(ex.p1, 1 - FR)
print('full-model threshold', tf, 'extra pairs', len(ex), 'pos', ex.y.sum(), 'selected pos', ex.y[ex.p1 >= tf].sum())
a = sl[sl.p1 >= np.quantile(oof, 1 - FR)]; b = ex[ex.p1 >= tf]
print('selected slice', len(a), a.groupby('country').size().to_dict(), 'extra', len(b), b.groupby('country').size().to_dict())
a.to_parquet(f'{W}/k20_slice.parquet', index=False); b.to_parquet(f'{W}/k20_extra.parquet', index=False)
pd.concat([a, b])[['rid', 's1', 'q_text', 's_text']].to_parquet(f'{W}/k20_ce_in.parquet', index=False)
open(f'{W}/k20_tf.txt', 'w').write(str(tf))
