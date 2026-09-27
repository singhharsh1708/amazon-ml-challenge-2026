"""Step 7 (France empty-S1 fill). Experiment/validation script b3.py for this step.

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
SUF = "['fils','cie','groupe','developpement','associes','france','et','compagnie','services','frs','freres','st','and']"
c.execute(f"""create or replace table g2 as select *, (qnum - snum) in (1,2,3,4,5,7,9,11,13,21) up,
  list_bool_and(list_transform(list_filter(string_split(nrm,' '), x -> x <> ''), x -> list_contains({SUF}, x))) rm_ok
  from f where v in ('same','same_noise') and ac in ('same','same_ns','same_num_wtypo','missing','s_missing') and p_samename < 0.5*p""")
c.execute("create or replace view g3 as select * from g2 where coalesce(rm_ok, true)")
pct = lambda a, b: round(100*a/b, 2) if b else None
for th in (0.5, 0.6, 0.7, 0.8):
    n, ex, lc, dot, lcne, ne, up, nce, cepos, miss = c.execute(f"""select count(*), sum(exact::int), sum(lc::int), sum(dot::int),
      sum(case when not exact then lc::int end), sum((not exact)::int), sum(coalesce(up,false)::int), count(ce), sum((ce>=0)::int), sum((ac='missing')::int)
      from g3 where p >= {th}""").fetchone()
    print(th, "n", n, "exact", ex, "addr_missing", miss, "lc%", pct(lc, n), "dot%", pct(dot, n), "lc_nonexact", lcne, "/", ne, pct(lcne or 0, ne), "up%", pct(up, n), "ce_cov", nce, "ce>=0%", pct(cepos, nce))
print(c.execute("select nc, ac, count(*), sum((ce>=0)::int), count(ce) from g3 where p>=0.5 group by all order by 3 desc").fetchall())
print("dropped swaps", c.execute("select count(*) from g2 where p>=0.5 and not coalesce(rm_ok,true)").fetchone())
print("--- lowercase / ce<0 rows p>=0.5")
for r in c.execute("select round(p,3), round(ce,2), lc, nc, ac, qbn, '|', qba, '||', sbn, '|', sba from g3 where p>=0.5 and (lc or ce<0) order by p desc").fetchall(): print(r)
