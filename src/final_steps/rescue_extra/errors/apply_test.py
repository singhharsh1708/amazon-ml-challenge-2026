"""Scores test rescue candidates with a trained rescue gate and writes the test addition set.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Command-line arguments used: argv[1], argv[2], argv[3].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
    {E}/test_{tag}.parquet
    {E}/test_{tag}_ce1.parquet
    {E}/test_{tag}_ce2.parquet
    {E}/test_additions_{tag}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
sys.path.insert(0, f'{FINAL_STEPS}/rescue_extra/errors')
from nm1b_lib import feats
E = f'{WORK_DIR}/top50/errors'
tag, model, T = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = duckdb.connect()
d = c.execute(f"""SELECT p.rid, p.s1, p.jw, p.q_text, p.s_text, p.country, e1.ce ce1, e2.ce ce2 FROM '{E}/test_{tag}.parquet' p
  JOIN '{E}/test_{tag}_ce1.parquet' e1 USING (rid, s1) JOIN '{E}/test_{tag}_ce2.parquet' e2 USING (rid, s1)""").fetchnumpy()
m = lgb.Booster(model_file=f'{E}/{model}')
ps = m.predict(feats(d))
c.register('t', pa.table({'rid': d['rid'], 's1': d['s1'], 'ps': ps, 'q_text': d['q_text'], 's_text': d['s_text'], 'country': d['country']}))
c.execute(f"""CREATE TABLE add AS SELECT rid, arg_max(s1, ps) s1, max(ps) ps, arg_max(q_text, ps) q_text, arg_max(s_text, ps) s_text, arg_max(country, ps) country FROM t GROUP BY rid HAVING max(ps) >= {T}""")
c.execute(f"""COPY (SELECT rid, s1, 'S' || (rid // 10000000000) || '-' || lpad((rid % 10000000000)::varchar, 9, '0') AS rid_id,
    'S1-' || lpad((s1 % 10000000000)::varchar, 9, '0') AS s1_id, ps, country, q_text, s_text FROM add) TO '{E}/test_additions_{tag}.parquet' (FORMAT parquet)""")
print(c.execute("SELECT count(*), count(DISTINCT s1), round(avg(ps),3) FROM add").fetchone())
print(c.execute(f"SELECT country, count(*) FROM '{E}/test_additions_{tag}.parquet' GROUP BY 1").fetchall())
print(c.execute(f"SELECT rid_id, s1_id, round(ps,3), q_text, s_text FROM '{E}/test_additions_{tag}.parquet' USING SAMPLE 12").df().to_string())
