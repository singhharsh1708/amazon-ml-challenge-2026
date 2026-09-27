"""Validation for step 3: measures held-out F when top-1 pairs with p2 above 0.99 are capped by the ce1+ce2 stacker.

Step context: Step 3 (high-confidence recheck). US/India top-1 pairs with stage-2 p in (0.99, 0.999] are rescored by the ce1+ce2 stacker and p is replaced by least(p, stacked).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/valid_band.parquet
    {D}/features/valid.parquet
    {D}/features/valid_predictions.parquet
    {SCR}/runs/new_slice_oof.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys
sys.path.insert(0, f'{FINAL_STEPS}/lib')
sys.path.insert(0, f'{REPO_DIR}/src')
import numpy as np, pyarrow as pa, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import *
from eval2 import setup, load, fscore
from expected_f import select
CE = f'{WORK_DIR}/ce'
c = con('2000MB'); setup(c)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def band(ces):
    """Load the validation band joined with the given cross-encoder scores."""
    sel = ", ".join(f"e{i}.ce AS ce_{i}" for i in range(len(ces))); join = " ".join(f"JOIN '{CE}/{f}' e{i} USING (rid, s1)" for i, f in enumerate(ces))
    return c.execute(f"""SELECT b.rid, b.s1, b.p2, {sel}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, (t.s1 IS NOT NULL)::int AS y, b.s1 % 2 AS fold
      FROM '{CE}/valid_band.parquet' b {join}
      JOIN (SELECT rid, s1, p, address_missing, source, rk FROM read_parquet('{D}/features/valid.parquet') v JOIN read_parquet('{D}/features/valid_predictions.parquet') USING (rid, s1)) f USING (rid, s1)
      LEFT JOIN truth t ON t.rid = b.rid AND t.s1 = b.s1""").fetchnumpy()
def oof(d, k):
    """Return out-of-fold predictions for the given inputs."""
    X = np.column_stack([lg(d['p2'])] + [d[f'ce_{i}'] for i in range(k)] + [lg(d['p1']), d['am'], d['source'], d['rk']]).astype(np.float32)
    out = np.zeros(len(d['y'])); params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=3, seed=1)
    for f in (0, 1):
        tr = d['fold'] != f
        out[~tr] = lgb.train(params, lgb.Dataset(X[tr], d['y'][tr]), 300).predict(X[~tr])
    return out
d3 = band(['valid_ce1.parquet', 'valid_ce2.parquet', 'valid_kbase.parquet']); o3 = oof(d3, 3)
d2 = band(['valid_ce1.parquet', 'valid_ce2.parquet']); o2 = oof(d2, 2)
hi = (d2['p2'] > 0.99)
print(f"high slice (p2 in (0.99, 0.998]): pairs {hi.sum():,} positives {d2['y'][hi].mean():.4f} AUC p2 {roc_auc_score(d2['y'][hi], d2['p2'][hi]):.4f} ce2 {roc_auc_score(d2['y'][hi], d2['ce_1'][hi]):.4f} stacked2 {roc_auc_score(d2['y'][hi], o2[hi]):.4f}")
c.register('s3', pa.table({'rid': d3['rid'], 's1': d3['s1'], 'ps': o3.astype(np.float32)}))
c.register('s2', pa.table({'rid': d2['rid'], 's1': d2['s1'], 'ps': o2.astype(np.float32)}))
c.execute(f"CREATE TABLE ranked AS SELECT o.rid, o.s1, o.p2, row_number() OVER (PARTITION BY o.rid ORDER BY o.p2 DESC) AS rk FROM '{SCR}/runs/new_slice_oof.parquet' o")
def run(tag, high, scale=1.0):
    """Build the mixed scores for one configuration and return held-out F."""
    extra = f"WHEN r.rk = 1 AND r.p2 > 0.99 THEN coalesce(least(r.p2, s2.ps * {scale} + r.p2 * (1 - {scale})), r.p2)" if high else ""
    c.execute(f"""CREATE OR REPLACE TABLE mix AS SELECT r.rid, r.s1, CASE WHEN r.rk <= 2 AND r.p2 BETWEEN 0.01 AND 0.99 THEN coalesce(s3.ps, r.p2) {extra} ELSE r.p2 END AS p2
        FROM ranked r LEFT JOIN s3 USING (rid, s1) LEFT JOIN s2 USING (rid, s1)""")
    load(c, 'm', 'mix')
    dd = c.execute("SELECT rid, s1, p FROM best_m WHERE veto = 0 AND p >= 0.01").fetchnumpy()
    keep = select(dd['s1'], dd['p'].astype(np.float64), 1.0)
    c.register('kk', pa.table({'rid': dd['rid'][keep], 's1': dd['s1'][keep]}))
    c.execute("CREATE OR REPLACE TEMP TABLE A AS SELECT b.* FROM best_m b JOIN kk USING (rid, s1)")
    c.execute("""DELETE FROM A k WHERE k.p < 0.95 AND k.fne = 0 AND EXISTS (SELECT 1 FROM A x WHERE x.s1 = k.s1 AND x.rid <> k.rid AND x.rid // 10000000000 = k.rid // 10000000000 AND x.fne = 1)""")
    f = fscore(c)
    print(f"{tag:34} {f['ALL'][0]:.5f}/{f['ALL'][1]:.5f} US {f['us'][1]:.5f} IN {f['india'][1]:.5f} assigned {int(f['ALL'][2])}", flush=True)
run('v15 base (no high recheck)', False)
run('high recheck, stacked ce1+ce2', True)
run('high recheck, half weight', True, 0.5)
