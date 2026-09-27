"""Validation: second stacker experiment (feature sets and cross-encoder subsets) on the held-out split.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/valid_band.parquet
    {D}/features/valid.parquet
    {D}/features/valid_predictions.parquet
    {SCR}/runs/new_slice_oof.parquet
    /runs/new_slice_oof.parquet

Outputs:
    {CE}/{tag_out}.parquet
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
ce_files = sys.argv[1].split(',')
tag_out = sys.argv[2] if len(sys.argv) > 2 else 'valid_stacked_multi'
c = con('1500MB'); setup(c)
ce_sel = ", ".join(f"e{i}.ce AS ce_{i}" for i in range(len(ce_files)))
ce_join = " ".join(f"JOIN '{f}' e{i} USING (rid, s1)" for i, f in enumerate(ce_files))
d = c.execute(f"""SELECT b.rid, b.s1, b.p2, {ce_sel}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, (t.s1 IS NOT NULL)::int AS y, b.s1 % 2 AS fold
    FROM '{CE}/valid_band.parquet' b {ce_join}
    JOIN (SELECT rid, s1, p, address_missing, source, rk FROM read_parquet('{D}/features/valid.parquet') v JOIN read_parquet('{D}/features/valid_predictions.parquet') USING (rid, s1)) f USING (rid, s1)
    LEFT JOIN truth t ON t.rid = b.rid AND t.s1 = b.s1""").fetchnumpy()
y = d['y']; lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print(f"band pairs {len(y):,} pos {y.mean():.3f} | AUC gbm {roc_auc_score(y, d['p2']):.4f} " + " ".join(f"ce_{i} {roc_auc_score(y, d[f'ce_{i}']):.4f}" for i in range(len(ce_files))))
X = np.column_stack([lg(d['p2'])] + [d[f'ce_{i}'] for i in range(len(ce_files))] + [lg(d['p1']), d['am'], d['source'], d['rk']]).astype(np.float32)
oof = np.zeros(len(y))
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, num_threads=4, seed=1)
for k in (0, 1):
    tr = d['fold'] != k
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
    oof[~tr] = m.predict(X[~tr])
print(f"stacked AUC {roc_auc_score(y, oof):.4f}")
c.register('st', pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': oof.astype(np.float32)}))
c.execute(f"""CREATE TABLE mix AS SELECT o.rid, o.s1, coalesce(st.ps, o.p2) AS p2 FROM '{SCR}/runs/new_slice_oof.parquet' o LEFT JOIN st USING (rid, s1)""")
load(c, 'new_slice')
load(c, 'stk', "mix")
for T in (0.85,):
    row = []
    for tag in ('new_slice', 'stk'):
        assigned(c, tag, T, True); f = fscore(c)
        row.append(f"{tag} {f['ALL'][0]:.5f}/{f['ALL'][1]:.5f} (US {f['us'][1]:.5f} IN {f['india'][1]:.5f})")
    print(f"T={T}: " + " | ".join(row), flush=True)
pq.write_table(pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': oof.astype(np.float32)}), f'{CE}/{tag_out}.parquet')

import sys as _s
_s.path.insert(0, f'{REPO_DIR}/src')
from expected_f import select
c.execute("CREATE OR REPLACE TABLE ranked AS SELECT o.rid, o.s1, o.p2, row_number() OVER (PARTITION BY o.rid ORDER BY o.p2 DESC) AS rk FROM '" + SCR + "/runs/new_slice_oof.parquet' o")
c.execute("CREATE OR REPLACE TABLE mix2 AS SELECT r.rid, r.s1, CASE WHEN r.rk <= 2 AND r.p2 BETWEEN 0.01 AND 0.99 THEN coalesce(st.ps, r.p2) ELSE r.p2 END AS p2 FROM ranked r LEFT JOIN st USING (rid, s1)")
load(c, 'nb', 'mix2')
dd = c.execute("SELECT rid, s1, p FROM best_nb WHERE veto = 0 AND p >= 0.01").fetchnumpy()
keep = select(dd['s1'], dd['p'].astype(np.float64), 1.0)
c.register('kk', pa.table({'rid': dd['rid'][keep], 's1': dd['s1'][keep]}))
c.execute("CREATE OR REPLACE TEMP TABLE A AS SELECT b.* FROM best_nb b JOIN kk USING (rid, s1)")
c.execute("""DELETE FROM A k WHERE k.p < 0.95 AND k.fne = 0 AND EXISTS (SELECT 1 FROM A x WHERE x.s1 = k.s1 AND x.rid <> k.rid
    AND x.rid // 10000000000 = k.rid // 10000000000 AND x.fne = 1)""")
f = fscore(c)
print(f"V11-STYLE (narrow band + EF w=1 + sibling): {f['ALL'][0]:.5f}/{f['ALL'][1]:.5f} US {f['us'][1]:.5f} IN {f['india'][1]:.5f}", flush=True)
