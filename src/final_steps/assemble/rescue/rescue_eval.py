"""Step 4 input (rescue additions). Experiment/validation script rescue_eval.py for this step.

Step context: Step 4 input (rescue additions). Rescue candidates for unmatched records scored by ce1/ce2 and a LightGBM gate; produces test_rescue_adds.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {H}/valid_s1.parquet
    {H}/truth.parquet
    {R}/valid_v11_assigned.parquet
    {R}/valid_v11_scores.parquet
    $REPO_DIR/data/candidates/train.parquet
    {R}/valid_rescue.parquet
    {R}/valid_rescue_{t}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score
S = f'{WORK_DIR}'
H = f'{S}/wf/hunt-odd-one-out-model'
R = f'{S}/rescue'
W = 1.887
FAMS = ['aexact', 'corenum', 'cat', 'typo', 'anum', 'aalpha']
ce_tags = sys.argv[1].split(',')
c = duckdb.connect(); c.execute(f"SET memory_limit='1500MB'; SET threads=2; SET temp_directory='{R}/tmp_eval'")
c.execute(f"CREATE TABLE vs1 AS SELECT * FROM '{H}/valid_s1.parquet'")
c.execute(f"CREATE TABLE truth AS SELECT * FROM '{H}/truth.parquet'")
c.execute(f"CREATE TABLE A AS SELECT a.rid, a.s1, t.s1 AS ts FROM '{R}/valid_v11_assigned.parquet' a LEFT JOIN truth t USING (rid)")
c.execute(f"CREATE TABLE touch AS SELECT DISTINCT rid FROM '{R}/valid_v11_scores.parquet'")
c.execute(f"CREATE TABLE cand AS SELECT rid, s1 FROM read_parquet('{REPO_DIR}/data/candidates/train.parquet') WHERE rid IN (SELECT rid FROM '{R}/valid_rescue.parquet')")
ce_sel = ", ".join(f"e{i}.ce AS ce{i}" for i in range(len(ce_tags)))
ce_join = " ".join(f"JOIN '{R}/valid_rescue_{t}.parquet' e{i} USING (rid, s1)" for i, t in enumerate(ce_tags))
c.execute(f"""CREATE TABLE rz AS SELECT r.*, {ce_sel}, t.s1 AS ts,
    (NOT r.assigned AND NOT (r.rid NOT IN (SELECT rid FROM touch) AND EXISTS (SELECT 1 FROM cand k WHERE k.rid = r.rid AND k.s1 = t.s1))) AS eligible
    FROM '{R}/valid_rescue.parquet' r {ce_join} LEFT JOIN truth t ON t.rid = r.rid""")
d = c.execute("SELECT * FROM rz WHERE eligible").fetchnumpy()
print(f"eligible rescue pairs {len(d['rid']):,}, true {int(d['label'].sum()):,}")
q = [x.split(' | ', 1) for x in d['q_text']]; s = [x.split(' | ', 1) for x in d['s_text']]
qn, qa = [a[0] for a in q], [a[1] if len(a) > 1 else '' for a in q]
sn, sa = [a[0] for a in s], [a[1] if len(a) > 1 else '' for a in s]
fam = [set(f.split(',')) for f in d['family']]
X = np.column_stack(
    [d[f'ce{i}'] for i in range(len(ce_tags))]
    + [np.array([fz in f for f in fam], np.float32) for fz in FAMS]
    + [np.array([len(f) for f in fam], np.float32),
       np.array([fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.token_set_ratio(a, b) if a else -1 for a, b in zip(qa, sa)], np.float32),
       np.array([float(not a) for a in qa], np.float32)]).astype(np.float32)
y = d['label'].astype(int)
rid = d['rid']
order = np.argsort(rid, kind='stable')
fold = (d['s1'] % 2)
oof = np.zeros(len(y))
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, verbose=-1, num_threads=2, seed=3)
for k in (0, 1):
    tr = fold != k
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
    oof[~tr] = m.predict(X[~tr])
print(f"AUC ce0 {roc_auc_score(y, d['ce0']):.4f} rescue-model {roc_auc_score(y, oof):.4f}")
c.register('rs', pa.table({'rid': rid, 's1': d['s1'], 'ps': oof, 'ts': d['ts']}))
c.execute("CREATE OR REPLACE TABLE rbest AS SELECT rid, arg_max(s1, ps) AS s1, max(ps) AS ps, any_value(ts) AS ts FROM rs GROUP BY rid")
def F(extra):
    """Return held-out precision, recall and F when the given extra pairs are added."""
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM A UNION ALL SELECT rid, s1, ts FROM rbest WHERE {extra}),
      per AS (SELECT v.s1, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) GROUP BY ALL)
      SELECT avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END),
             avg(CASE WHEN n_true=0 AND tp+fpw+fpd=0 THEN 1.0 WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+fpd) END)
      FROM per""").fetchone()
base = F('false'); print(f"v11 base mix {base[0]:.5f} exact {base[1]:.5f}")
for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    n, tp = c.execute(f"SELECT count(*), count(*) FILTER (WHERE ts = s1) FROM rbest WHERE ps >= {t}").fetchone()
    f = F(f"ps >= {t}")
    print(f"t={t}: add {n:,} ({tp:,} true) mix {f[0]:.5f} ({f[0]-base[0]:+.5f}) exact {f[1]:.5f} ({f[1]-base[1]:+.5f})", flush=True)
m = lgb.train(params, lgb.Dataset(X, y), 300)
m.save_model(f'{R}/rescue_model_{"_".join(ce_tags)}.txt')
