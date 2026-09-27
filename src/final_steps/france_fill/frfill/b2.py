"""Step 7 (France empty-S1 fill). Experiment/validation script b2.py for this step.

Step context: Step 7 (France empty-S1 fill). Candidate pool for France source-1 records that ended with no match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/fill/frfill
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
W = f"{WORK_DIR}/fill/frfill"
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill'")
c.execute("""create or replace table f as select *,
  (qbn = lower(qbn) and regexp_matches(qbn, '[a-z]')) lc, contains(qbn, '.') dot,
  lower(trim(qbn)) = lower(trim(sbn)) exact,
  try_cast(regexp_extract(coalesce(qba,''), '(\\d+)', 1) as bigint) qnum, try_cast(regexp_extract(coalesce(sba,''), '(\\d+)', 1) as bigint) snum
  from pr""")
c.execute("create or replace view g as select *, (qnum - snum) in (1,2,3,4,5,7,9,11,13,21) up from f where v in ('same','same_noise') and ac in ('same','same_ns','same_num_wtypo','missing','s_missing') and (nadd = '' or true)")
print("verdict-kept by ac", c.execute("select ac, count(*) from f where v in ('same','same_noise') group by 1 order by 2 desc").fetchall())
for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    for cmp in ("samename", "any"):
        cond = "p_samename < 0.5*p" if cmp == "samename" else "coalesce(p_other,0) < 0.5*p"
        r = c.execute(f"""select count(*), sum(exact::int), sum(lc::int), sum(dot::int),
          sum(case when not exact then lc::int end), sum((not exact)::int),
          sum(coalesce(up,false)::int), count(ce), sum((ce>=0)::int)
          from g where p >= {th} and {cond}""").fetchone()
        n, ex, lc, dot, lcne, ne, up, nce, cepos = r
        pct = lambda a, b: round(100*a/b, 2) if b else None
        print(th, cmp, "n", n, "exact", ex, "lc%", pct(lc, n), "dot%", pct(dot, n), "lc_nonexact", lcne, "/", ne, pct(lcne or 0, ne), "up%", pct(up, n), "ce_cov", nce, "ce>=0%", pct(cepos, nce))
print(c.execute("select v, nc, count(*) from g where p>=0.5 group by all order by 3 desc").fetchall())
print(c.execute("select nadd, nrm, count(*) from g where p>=0.5 and v='same_noise' group by all order by 3 desc limit 20").fetchall())
