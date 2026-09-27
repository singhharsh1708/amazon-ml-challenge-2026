"""Validation: out-of-fold LightGBM stacker over stage-2 and cross-encoder scores, reporting AUC and held-out F.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/valid_band.parquet
    {D}/features/valid.parquet
    {D}/features/valid_predictions.parquet
    {SCR}/runs/new_slice_oof.parquet

Outputs:
    {CE}/valid_stacked.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys
sys.path.insert(0, f'{FINAL_STEPS}/lib')
sys.path.insert(0, f'{WORK_DIR}')
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import *
from eval2 import setup, load, assigned, fscore
CE = f'{WORK_DIR}/ce'
ce_file = sys.argv[1]
c = con('2500MB'); setup(c)
d = c.execute(f"""SELECT b.rid, b.s1, b.p2, e.ce, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, (t.s1 IS NOT NULL)::int AS y, b.s1 % 2 AS fold
    FROM '{CE}/valid_band.parquet' b JOIN '{ce_file}' e USING (rid, s1)
    JOIN (SELECT rid, s1, p, address_missing, source, rk FROM read_parquet('{D}/features/valid.parquet') v JOIN read_parquet('{D}/features/valid_predictions.parquet') USING (rid, s1)) f USING (rid, s1)
    LEFT JOIN truth t ON t.rid = b.rid AND t.s1 = b.s1""").fetchnumpy()
y = d['y']; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print(f"band pairs {len(y):,} pos {y.mean():.3f} | AUC gbm {roc_auc_score(y, d['p2']):.4f} ce {roc_auc_score(y, d['ce']):.4f}")
X = np.column_stack([lg(d['p2']), d['ce'], lg(d['p1']), d['am'], d['source'], d['rk']]).astype(np.float32)
oof = np.zeros(len(y))
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=4, seed=1)
for k in (0, 1):
    tr = d['fold'] != k
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
    oof[~tr] = m.predict(X[~tr])
print(f"stacked AUC {roc_auc_score(y, oof):.4f}")
c.register('st', pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': oof.astype(np.float32)}))
c.execute(f"""CREATE TABLE mix AS SELECT o.rid, o.s1, coalesce(st.ps, o.p2) AS p2 FROM '{SCR}/runs/new_slice_oof.parquet' o LEFT JOIN st USING (rid, s1)""")
load(c, 'base'.replace('base', 'new_slice'))
load(c, 'stk', "mix")
for T in (0.75, 0.8, 0.85, 0.9):
    row = []
    for tag in ('new_slice', 'stk'):
        assigned(c, tag, T, True); f = fscore(c)
        row.append(f"{tag} {f['ALL'][0]:.5f}/{f['ALL'][1]:.5f} (US {f['us'][1]:.5f} IN {f['india'][1]:.5f})")
    print(f"T={T}: " + " | ".join(row), flush=True)
pq.write_table(pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': oof.astype(np.float32)}), f'{CE}/valid_stacked.parquet')
