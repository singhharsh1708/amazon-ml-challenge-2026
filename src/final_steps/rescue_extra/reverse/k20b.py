"""Step 6 (reverse additions). Experiment/validation script k20b.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {H}/valid_s1.parquet
    {H}/truth.parquet
    {R}/valid_v15_assigned.parquet
    {R}/valid_v11_scores.parquet
    $REPO_DIR/data/candidates/train.parquet
    {R}/valid_rescue.parquet
    {R}/valid_rescue_ce1.parquet
    {R}/valid_rescue_ce2.parquet
    {W}/k20_pn.parquet
    {W}/k20_rf.parquet
    {W}/k20_ce2.parquet
    {W}/k20_slice.parquet

Outputs:
    {W}/k20_cand_rids.parquet
    {W}/k20_cr.parquet
    {W}/k20_stage2.txt
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, shutil, os
import duckdb, numpy as np, pandas as pd, pyarrow as pa, lightgbm as lgb
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score
S = f'{WORK_DIR}'
H = f'{S}/wf/hunt-odd-one-out-model'
R = f'{S}/rescue'
W = f'{S}/top50/reverse'
WT = 1.887
MODE = sys.argv[1]
FAMS = ['aexact', 'corenum', 'cat', 'typo', 'anum', 'aalpha']
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp_s2u'")
c.execute(f"CREATE TABLE vs1 AS SELECT *, hash(s1) % 40 < 20 AS h0 FROM '{H}/valid_s1.parquet'")
c.execute(f"CREATE TABLE truth AS SELECT * FROM '{H}/truth.parquet'")
c.execute(f"CREATE TABLE A AS SELECT a.rid, a.s1, t.s1 AS ts FROM '{R}/valid_v15_assigned.parquet' a LEFT JOIN truth t USING (rid)")
c.execute(f"CREATE TABLE touch AS SELECT DISTINCT rid FROM '{R}/valid_v11_scores.parquet'")
c.execute(f"CREATE TABLE cand AS SELECT rid, s1 FROM read_parquet('{REPO_DIR}/data/candidates/train.parquet') WHERE rid IN (SELECT rid FROM '{R}/valid_rescue.parquet')")
c.execute(f"""CREATE TABLE rz AS SELECT r.*, e0.ce AS ce0, e1.ce AS ce1, t.s1 AS ts,
    (NOT r.assigned AND r.rid NOT IN (SELECT rid FROM A) AND NOT (r.rid NOT IN (SELECT rid FROM touch) AND EXISTS (SELECT 1 FROM cand k WHERE k.rid = r.rid AND k.s1 = t.s1))) AS eligible
    FROM '{R}/valid_rescue.parquet' r JOIN '{R}/valid_rescue_ce1.parquet' e0 USING (rid, s1) JOIN '{R}/valid_rescue_ce2.parquet' e1 USING (rid, s1) LEFT JOIN truth t ON t.rid = r.rid""")
d = c.execute("SELECT * FROM rz WHERE eligible").fetchnumpy()
q = [x.split(' | ', 1) for x in d['q_text']]; s = [x.split(' | ', 1) for x in d['s_text']]
qn, qa = [a[0] for a in q], [a[1] if len(a) > 1 else '' for a in q]
sn, sa = [a[0] for a in s], [a[1] if len(a) > 1 else '' for a in s]
fam = [set(f.split(',')) for f in d['family']]
X = np.column_stack([d['ce0'], d['ce1']] + [np.array([fz in f for f in fam], np.float32) for fz in FAMS]
    + [np.array([len(f) for f in fam], np.float32),
       np.array([fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.token_set_ratio(a, b) if a else -1 for a, b in zip(qa, sa)], np.float32),
       np.array([float(not a) for a in qa], np.float32)]).astype(np.float32)
y = d['label'].astype(int); fold = d['s1'] % 2; oof = np.zeros(len(y))
params = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, verbose=-1, num_threads=2, seed=3)
for k in (0, 1):
    tr = fold != k
    oof[~tr] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 300).predict(X[~tr])
c.register('rs', pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': oof, 'ts': d['ts']}))
c.execute("CREATE TABLE radd AS SELECT rid, arg_max(s1, ps) AS s1, any_value(ts) AS ts FROM rs GROUP BY rid HAVING max(ps) >= 0.7")
c.execute("CREATE TABLE B AS SELECT rid, s1, ts FROM A UNION ALL SELECT rid, s1, ts FROM radd")

PN = pd.read_parquet(f'{W}/k20_pn.parquet'); RF = pd.read_parquet(f'{W}/k20_rf.parquet')
def prep(v):
    """Join the reverse-search features onto a frame of candidates."""
    v = v.merge(RF, on='rid', how='left')
    v = v.merge(PN.rename(columns={'nc': 'qn', 'npool': 'npool_q'}), on=['country', 'qn'], how='left')
    v = v.merge(PN.rename(columns={'nc': 'sn', 'npool': 'npool_s'}), on=['country', 'sn'], how='left')
    v[['npool_q', 'npool_s', 'ncand', 'fmax']] = v[['npool_q', 'npool_s', 'ncand', 'fmax']].fillna(0)
    v['india'] = (v.country == 'india').astype(float); v['rel'] = v.score / v.rmax
    v['ssr'] = v.score / v.smax; v['gap'] = v.smax - v.s2nd
    return v
CE = pd.read_parquet(f'{W}/k20_ce2.parquet').rename(columns={'ce': 'c2'})
v = prep(pd.read_parquet(f'{W}/k20_slice.parquet')).merge(CE, on=['rid', 's1'])
v = v.drop(columns=['h0']).merge(c.execute("SELECT s1, h0 FROM vs1").df(), on='s1', how='left')
v = v.merge(c.execute("SELECT rid, s1 AS ts FROM truth").df().drop_duplicates('rid').set_index('rid').loc[lambda z: z.index.isin(v.rid)].reset_index(), on='rid', how='left')
v['elig'] = ~v.rid.isin(c.execute("SELECT rid FROM B").df().rid)
ex = prep(pd.read_parquet(f'{W}/k20_extra.parquet')).merge(CE, on=['rid', 's1'])
F = ['c2', 'npool_q', 'npool_s', 'ncand', 'fmax', 'tsr', 'tsort', 'rcomp', 'exact', 'qaddr', 'atsr', 'aovl', 'anum', 'ns1_q', 'ns1_s', 'score', 'rel', 'ssr', 'gap', 'nrev', 'rk', 'qlen', 'slen', 'india']
print('slice pairs', len(v), 'pos', v.y.sum(), 'eligible pos', v.y[v.elig].sum(), 'extra', len(ex), 'pos', ex.y.sum())
X = v[F].values.astype(np.float32); y = v.y.values; h0 = v.h0.values.astype(bool)
XE = ex[F].values.astype(np.float32); YE = ex.y.values
p2 = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=3, seed=5)
NR = 300
oof = np.zeros(len(y)); models = {}
for k in (True, False):
    tr = h0 != k
    models[k] = lgb.train(p2, lgb.Dataset(np.vstack([X[tr], XE]), np.concatenate([y[tr], YE])), NR)
    oof[~tr] = models[k].predict(X[~tr])
print('AUC c2', round(roc_auc_score(y, v.c2), 4), 'stack', round(roc_auc_score(y, oof), 4))
v['p'] = oof
ve = v[v.elig].sort_values('p', ascending=False)
cr = ve.drop_duplicates('rid')[['rid', 's1', 'p', 'ts', 'h0']]
cr = cr[cr.p >= 0.15]
print('candidate records', len(cr))
if MODE == 'dump':
    cr[['rid']].to_parquet(f'{W}/k20_cand_rids.parquet', index=False)
    sys.exit()
ns = prep(pd.read_parquet(f'{W}/k20_comp.parquet')).merge(pd.read_parquet(f'{W}/k20_comp_ce2.parquet').rename(columns={'ce': 'c2'}), on=['rid', 's1'])
ns = ns[ns.rid.isin(cr.rid)].merge(cr[['rid', 'h0']], on='rid')
print('competitor pairs', len(ns), 'true', ns.y.sum())
ns['p'] = 0.0
for k in (True, False):
    m = ns.h0.values.astype(bool) == k
    if m.any():
        ns.loc[m, 'p'] = models[k].predict(ns.loc[m, F].values.astype(np.float32))
cr = cr.merge(ns.groupby('rid').p.max().rename('nsmax'), on='rid', how='left')
sec = ve.groupby('rid').p.apply(lambda z: z.iloc[1] if len(z) > 1 else -1.0).rename('sec')
cr = cr.merge(sec, on='rid', how='left')
cr['nsmax'] = np.maximum(cr.nsmax.fillna(-1), cr.sec.fillna(-1))
c.register('crr', cr)
def Fs(extra):
    """Return held-out F statistics when the given extra reverse pairs are added."""
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM B UNION ALL SELECT rid, s1, ts FROM crr WHERE {extra}),
      per AS (SELECT v.s1, v.h0, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) GROUP BY ALL),
      sc AS (SELECT h0, CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{WT}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{WT}*fpd) END AS f,
        CASE WHEN n_true=0 AND tp+fpw+fpd=0 THEN 1.0 WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+fpd) END AS fe FROM per)
      SELECT avg(f), avg(f) FILTER (WHERE h0), avg(f) FILTER (WHERE NOT h0), avg(fe) FROM sc""").fetchone()
base = Fs('false')
print(f"base (v15+rescue) mix {base[0]:.5f} h0 {base[1]:.5f} h1 {base[2]:.5f} exact {base[3]:.5f}")
for pol in ('true', 'nsmax < 0.1', 'nsmax < 0.2', 'nsmax < 0.3', 'p - nsmax >= 0.4'):
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        cond = f"p >= {t} AND {pol}"
        n, tp, fpd = c.execute(f"SELECT count(*), count(*) FILTER (WHERE ts = s1), count(*) FILTER (WHERE ts IS NULL) FROM crr WHERE {cond}").fetchone()
        f = Fs(cond)
        print(f"{pol} t={t}: add {n} (true {tp}, fpd {fpd}) mix {f[0]-base[0]:+.6f} h0 {f[1]-base[1]:+.6f} h1 {f[2]-base[2]:+.6f} exact {f[3]-base[3]:+.6f}", flush=True)
cr.to_parquet(f'{W}/k20_cr.parquet', index=False)
m = lgb.train(p2, lgb.Dataset(np.vstack([X, XE]), np.concatenate([y, YE])), NR); m.save_model(f'{W}/k20_stage2.txt')
c.close(); shutil.rmtree(f'{W}/tmp_s2u', ignore_errors=True)
