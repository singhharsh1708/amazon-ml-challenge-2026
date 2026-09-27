"""Step 6 (name-key rescue). Experiment/validation script combo.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
    {E}/oof_{t}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, duckdb
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb', read_only=True)
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
W = 1.887
halves = {'fit': 'hash(v.s1) % 40 < 20', 'eval': 'hash(v.s1) % 40 >= 20', 'all': 'true'}
def F(adds, half):
    """Return held-out F when the given rescue families are added."""
    u = " UNION ALL ".join(f"SELECT rid, arg_max(s1, ps) s1, arg_max(ts, ps) ts FROM '{E}/oof_{t}.parquet' GROUP BY rid HAVING max(ps) >= {th}" for t, th in adds) or "SELECT NULL::bigint rid, NULL::bigint s1, NULL::bigint ts WHERE false"
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM fa UNION ALL SELECT * FROM ({u})),
      per AS (SELECT v.s1, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) WHERE {half} GROUP BY ALL)
      SELECT avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END) FROM per""").fetchone()[0]
base = {h: F([], q) for h, q in halves.items()}
print('base', {h: round(v, 6) for h, v in base.items()})
configs = {'A nm1c@.85': [('nm1c', 0.85)], 'B nm1d@.7': [('nm1d', 0.7)], 'C nm1c@.85+hc@.7': [('nm1c', 0.85), ('hc', 0.7)],
           'D nm1d@.7+hc@.7': [('nm1d', 0.7), ('hc', 0.7)], 'E nm1d@.85': [('nm1d', 0.85)], 'F nm1d@.85+hc@.8': [('nm1d', 0.85), ('hc', 0.8)]}
for k, adds in configs.items():
    r = {h: F(adds, q) - base[h] for h, q in halves.items()}
    print(f"{k:22} fit {r['fit']:+.6f} eval {r['eval']:+.6f} all {r['all']:+.6f} -> all {base['all'] + r['all']:.5f}", flush=True)
