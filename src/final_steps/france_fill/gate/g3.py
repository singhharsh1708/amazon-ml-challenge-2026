"""Step 7 gate. Experiment/validation script g3.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {S}/fr_rerun/work/new/scores.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.dont_write_bytecode = True
import duckdb
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
c = duckdb.connect(f"{G}/tmp/g.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
c.execute("create or replace table fr as select eid, bn, ba, lower(strip_accents(regexp_replace(bn, '[^A-Za-zÀ-ÿ0-9 ]', '', 'g'))) nk from raw where country='France'")
c.execute("create or replace table frs1 as select f.*, (f.eid in (select s1 from emp)) is_emp from fr f where eid < 20000000000")
c.execute("create or replace table frrec as select f.*, (f.eid in (select rid from asg)) is_asg from fr f where eid >= 20000000000")
print("French records: assigned vs unassigned; lowercase share, dot share")
for r in c.execute("""select is_asg, count(*), round(100*avg((bn = lower(bn) and regexp_matches(bn, '[a-z]'))::int),2), round(100*avg((bn like '%.%')::int),2),
   round(100*avg((coalesce(trim(ba),'') = '')::int),2) from frrec group by 1""").fetchall(): print("  ", r)
c.execute("create or replace table vpf as select v.*, " + "(cast(substr(rs, 2, 1) as bigint) * 10000000000 + cast(substr(rs, 4) as bigint)) rid, (cast(substr(s1s, 2, 1) as bigint) * 10000000000 + cast(substr(s1s, 4) as bigint)) s1 from vp v")
print("v20 French accepted pairs where rec address missing and ci-exact name: lowercase share, dot share")
print("  ", c.execute("""select count(*), round(100*avg((q.bn = lower(q.bn) and regexp_matches(q.bn, '[a-z]'))::int),2), round(100*avg((q.bn like '%.%')::int),2)
   from vpf v join frrec q on q.eid = v.rid join frs1 s on s.eid = v.s1 where coalesce(trim(q.ba),'') = '' and q.nk = s.nk""").fetchone())
print("\nshipped p>=0.8 pairs: score columns, same-name French S1 counts")
for r in c.execute(f"""select a.s1, a.rid, round(sc.p,3), round(sc.p_guard,3), round(sc.p_s2,3), round(sc.ps,3), round(sc.p1,3), sc.fne,
   (select count(*) from frs1 x where x.nk = s.nk) n_same_s1, (select count(*) from frs1 x where x.nk = s.nk and x.is_emp) n_same_s1_empty,
   (select count(*) from frrec y where y.nk = q.nk) n_same_rec, (select count(*) from frrec y where y.nk = q.nk and not y.is_asg) n_same_rec_unasg,
   s.bn, q.bn
   from a_p08 a join '{S}/fr_rerun/work/new/scores.parquet' sc using (rid, s1) join frs1 s on s.eid = a.s1 join frrec q on q.eid = a.rid order by sc.p desc""").fetchall(): print("  ", r)
print("\nall scored candidates in scores.parquet for the 27 shipped S1 (p>=0.3), with assignment state")
for r in c.execute(f"""select sc.s1, sc.rid, round(sc.p,3), (sc.rid in (select rid from a_p08)) shipped, (sc.rid in (select rid from asg)) asg_v20, q.bn, q.ba
   from '{S}/fr_rerun/work/new/scores.parquet' sc join frrec q on q.eid = sc.rid where sc.s1 in (select s1 from a_p08) and sc.p >= 0.3 and not (sc.rid in (select rid from a_p08)) order by sc.s1, sc.p desc""").fetchall(): print("  ", r)
