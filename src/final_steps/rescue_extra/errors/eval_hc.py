"""Step 6 (name-key rescue). Experiment/validation script eval_hc.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
    {E}/valid_hc_am.parquet
    {E}/valid_hc_ce1.parquet
    {E}/valid_hc_ce2.parquet

Outputs:
    {E}/hc_model.txt
    {E}/oof_hc.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, duckdb, numpy as np, pyarrow as pa, lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb', read_only=True)
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
W = 1.887
d = c.execute(f"""SELECT p.rid, p.s1, p.label, p.jw, p.q_text, p.s_text, e1.ce ce1, e2.ce ce2, t.s1 ts, (hash(p.s1) % 40 < 20)::int fold
  FROM '{E}/valid_hc_am.parquet' p JOIN '{E}/valid_hc_ce1.parquet' e1 USING (rid, s1) JOIN '{E}/valid_hc_ce2.parquet' e2 USING (rid, s1)
  LEFT JOIN truth t ON t.rid = p.rid""").fetchnumpy()
def feats(d):
    """Build the gate features for a frame of candidate pairs."""
    qn = [x.split(' | ', 1)[0] for x in d['q_text']]; sn = [x.split(' | ', 1)[0] for x in d['s_text']]
    rid = d['rid']; ce = (d['ce1'] + d['ce2']) / 2
    import pandas as pd
    df = pd.DataFrame({'rid': rid, 'ce': ce})
    g = df.groupby('rid')['ce']
    mx = g.transform('max').values; cnt = g.transform('count').values
    second = df.assign(r=g.rank(ascending=False, method='first')).query('r == 2').set_index('rid')['ce']
    sec = pd.Series(rid).map(second).fillna(-10).values
    other = np.where(ce >= mx, ce - sec, ce - mx)
    return np.column_stack([d['ce1'], d['ce2'], d['jw'], [fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)],
        [fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], cnt, other]).astype(np.float32)
X = feats(d); y = d['label'].astype(int)
print(f"pairs {len(y)} pos {y.sum()} AUC ce1 {roc_auc_score(y, d['ce1']):.4f} ce2 {roc_auc_score(y, d['ce2']):.4f}")
oof = np.zeros(len(y))
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, verbose=-1, num_threads=3, seed=5)
for k in (0, 1):
    tr = d['fold'] != k
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300)
    oof[~tr] = m.predict(X[~tr])
print(f"model AUC {roc_auc_score(y, oof):.4f}")
full = lgb.train(params, lgb.Dataset(X, y), 300); full.save_model(f'{E}/hc_model.txt')
c.register('np_', pa.table({'rid': d['rid'], 's1': d['s1'], 'ts': d['ts'], 'ps': oof, 'ce': (d['ce1'] + d['ce2']) / 2})); c.execute(f"COPY np_ TO '{E}/oof_hc.parquet' (FORMAT parquet)")
def F(extra, half):
    """Return held-out precision, recall and F when the given extra pairs are added."""
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM fa UNION ALL SELECT rid, s1, ts FROM nb WHERE {extra}),
      per AS (SELECT v.s1, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) WHERE {half} GROUP BY ALL)
      SELECT avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END) FROM per""").fetchone()[0]
halves = {'fit': 'hash(v.s1) % 40 < 20', 'eval': 'hash(v.s1) % 40 >= 20', 'all': 'true'}
for sc, grid in (('ce', [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3]), ('ps', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])):
    c.execute(f"CREATE OR REPLACE TEMP TABLE nb AS SELECT rid, arg_max(s1, {sc}) s1, max({sc}) sc, arg_max(ts, {sc}) ts FROM np_ GROUP BY rid")
    base = {h: F('false', q) for h, q in halves.items()}
    print(sc, 'base', {h: round(b, 5) for h, b in base.items()})
    best = None
    for t in grid:
        n, tp, fpd = c.execute(f"SELECT count(*), count(*) FILTER (WHERE ts=s1), count(*) FILTER (WHERE ts IS NULL) FROM nb WHERE sc >= {t} AND hash(s1) % 20 = 0").fetchone()
        r = {h: F(f'sc >= {t}', q) - base[h] for h, q in halves.items()}
        print(f"  {sc} t={t} slice-add {n} tp {tp} fpd {fpd} | fit {r['fit']:+.5f} eval {r['eval']:+.5f} all {r['all']:+.5f}", flush=True)
        if best is None or r['fit'] > best[1]: best = (t, r['fit'], r['eval'])
    print('  chosen on fit half', best)
    best2 = None
    for t in grid:
        r = F(f'sc >= {t}', halves['eval']) - base['eval']
        if best2 is None or r > best2[1]: best2 = (t, r)
    t2 = best2[0]
    print('  reverse: chosen on eval half', best2, 'scored on fit half', round(F(f'sc >= {t2}', halves['fit']) - base['fit'], 6))
