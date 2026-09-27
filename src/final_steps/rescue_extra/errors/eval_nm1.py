"""Step 6 (name-key rescue). Experiment/validation script eval_nm1.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
    {E}/valid_nm1_am.parquet
    {E}/valid_nm1_ce1.parquet
    {E}/valid_nm1_ce2.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, duckdb, numpy as np
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb', read_only=True)
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
W = 1.887
c.execute(f"""CREATE TEMP TABLE np AS SELECT p.rid, p.s1, p.label, p.jw, e1.ce ce1, e2.ce ce2, t.s1 ts FROM '{E}/valid_nm1_am.parquet' p
  JOIN '{E}/valid_nm1_ce1.parquet' e1 USING (rid, s1) JOIN '{E}/valid_nm1_ce2.parquet' e2 USING (rid, s1) LEFT JOIN truth t ON t.rid = p.rid""")
print(c.execute("SELECT min(ce1), max(ce1), min(ce2), max(ce2), count(*), sum(label) FROM np").fetchall())
score = sys.argv[1] if len(sys.argv) > 1 else '(ce1+ce2)/2'
c.execute(f"CREATE TEMP TABLE nb AS SELECT rid, arg_max(s1, sc) s1, max(sc) sc, arg_max(ts, sc) ts, arg_max(label, sc) AS lab, count(*) nc FROM (SELECT *, {score} sc FROM np) GROUP BY rid")
def F(extra, half):
    """Return held-out precision, recall and F when the given extra pairs are added."""
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM fa UNION ALL SELECT rid, s1, ts FROM nb WHERE {extra}),
      per AS (SELECT v.s1, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) WHERE {half} GROUP BY ALL)
      SELECT avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END) FROM per""").fetchone()[0]
halves = {'fit': 'hash(v.s1) % 40 < 20', 'eval': 'hash(v.s1) % 40 >= 20', 'all': 'true'}
base = {h: F('false', q) for h, q in halves.items()}
print('base', {h: round(b, 5) for h, b in base.items()})
qs = [float(x) for x in np.quantile(c.execute("SELECT sc FROM nb").fetchnumpy()['sc'], [0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98])]
best = None
for t in qs:
    n, tp, fpd = c.execute(f"SELECT count(*), count(*) FILTER (WHERE ts=s1), count(*) FILTER (WHERE ts IS NULL) FROM nb WHERE sc >= {t}").fetchone()
    r = {h: F(f'sc >= {t}', q) - base[h] for h, q in halves.items()}
    print(f"t={t:.4f} add {n} tp {tp} fpd {fpd} | fit {r['fit']:+.5f} eval {r['eval']:+.5f} all {r['all']:+.5f}", flush=True)
    if best is None or r['fit'] > best[1]: best = (t, r['fit'], r['eval'])
print('chosen on fit half', best)
