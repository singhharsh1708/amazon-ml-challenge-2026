"""Step 6 (name-key rescue). Experiment/validation script combo2.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
    {E}/oof_{t}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb', read_only=True)
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
W = 1.887
def F(adds, grp):
    """Return held-out precision, recall and F when the given extra pairs are added."""
    u = " UNION ALL ".join(f"SELECT rid, arg_max(s1, ps) s1, arg_max(ts, ps) ts FROM '{E}/oof_{t}.parquet' GROUP BY rid HAVING max(ps) >= {th}" for t, th in adds) or "SELECT NULL::bigint rid, NULL::bigint s1, NULL::bigint ts WHERE false"
    return dict(c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM fa UNION ALL SELECT * FROM ({u})),
      per AS (SELECT v.s1, v.country, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) GROUP BY ALL)
      SELECT {grp} g, avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END) FROM per GROUP BY 1 ORDER BY 1""").fetchall())
for grp in ["country", "hash(s1 * 7 + 3) % 4"]:
    b = F([], grp); a = F([('nm1c', 0.85)], grp); d = F([('nm1d', 0.7), ('hc', 0.7)], grp)
    for g in b:
        print(f"{grp:22} {g}: A {a[g]-b[g]:+.6f} D {d[g]-b[g]:+.6f} D-A {d[g]-a[g]:+.6f}")
