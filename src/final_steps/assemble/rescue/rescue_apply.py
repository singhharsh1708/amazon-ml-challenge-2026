"""Scores test rescue candidates with the rescue gate and writes test_rescue_adds.parquet.

Step context: Step 4 input (rescue additions). Rescue candidates for unmatched records scored by ce1/ce2 and a LightGBM gate; produces test_rescue_adds.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/rescue
    {R}/rescue_model_ce1_ce2.txt
    {R}/test_rescue.parquet
    {R}/test_rescue_ce1.parquet
    {R}/test_rescue_ce2.parquet

Outputs:
    {R}/test_rescue_adds.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from rapidfuzz import fuzz
R = f'{WORK_DIR}/rescue'
FAMS = ['aexact', 'corenum', 'cat', 'typo', 'anum', 'aalpha']
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
m = lgb.Booster(model_file=f'{R}/rescue_model_ce1_ce2.txt')
c = duckdb.connect(); c.execute("SET memory_limit='1500MB'; SET threads=2")
d = c.execute(f"""SELECT r.rid, r.s1, r.family, r.q_text, r.s_text, r.country, e0.ce AS ce0, e1.ce AS ce1
    FROM '{R}/test_rescue.parquet' r JOIN '{R}/test_rescue_ce1.parquet' e0 USING (rid, s1) JOIN '{R}/test_rescue_ce2.parquet' e1 USING (rid, s1)
    WHERE r.country <> 'france'""").fetchnumpy()
q = [x.split(' | ', 1) for x in d['q_text']]; s = [x.split(' | ', 1) for x in d['s_text']]
qn, qa = [a[0] for a in q], [a[1] if len(a) > 1 else '' for a in q]
sn, sa = [a[0] for a in s], [a[1] if len(a) > 1 else '' for a in s]
fam = [set(f.split(',')) for f in d['family']]
X = np.column_stack(
    [d['ce0'], d['ce1']]
    + [np.array([fz in f for f in fam], np.float32) for fz in FAMS]
    + [np.array([len(f) for f in fam], np.float32),
       np.array([fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], np.float32),
       np.array([fuzz.token_set_ratio(a, b) if a else -1 for a, b in zip(qa, sa)], np.float32),
       np.array([float(not a) for a in qa], np.float32)]).astype(np.float32)
ps = m.predict(X)
c.register('rs', pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': ps, 'country': d['country']}))
c.execute(f"COPY (SELECT rid, arg_max(s1, ps) AS s1 FROM rs GROUP BY rid HAVING max(ps) >= {T}) TO '{R}/test_rescue_adds.parquet' (FORMAT parquet)")
print(f"test rescue pairs scored {len(ps):,} (US/India); accepted at {T}:", c.execute(f"SELECT count(*) FROM '{R}/test_rescue_adds.parquet'").fetchone()[0],
      "| by country:", c.execute(f"SELECT country, count(*) FROM (SELECT rid, arg_max(country, ps) AS country FROM rs GROUP BY rid HAVING max(ps) >= {T}) GROUP BY 1").fetchall())
