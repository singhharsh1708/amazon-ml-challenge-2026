"""Step 7 gate. Experiment/validation script g7.py for this step.

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
c.execute("create or replace table u1 as select nk, any_value(eid) s1, any_value(is_emp) emp from frs1 group by nk having count(*) = 1")
c.execute("create or replace table vs1 as select distinct s1 from vpf")
c.execute("""create or replace table mr as select q.eid rid, q.is_asg, u.s1, u.emp from frrec q join u1 u using (nk) where coalesce(trim(q.ba),'') = '' """)
print("missing-address records whose exact name is a unique French S1 name: by record assigned x S1 empty")
for r in c.execute("select is_asg, emp, count(*) from mr group by all order by all").fetchall(): print("  ", r)
print("of assigned ones: assigned to that same S1")
print("  ", c.execute("select count(*), sum((v.s1 = mr.s1)::int) from mr join vpf v using (rid) where mr.is_asg").fetchone())
c.execute(f"create or replace table mrp as select mr.*, sc.p, sc.p_guard, least(sc.p, sc.p_guard) eff from mr left join '{S}/fr_rerun/work/new/scores.parquet' sc using (rid, s1)")
print("unassigned, by S1 empty: n, scored, p>=0.8, eff>=0.8, eff>=0.7, median p")
for r in c.execute("select emp, count(*), count(p), sum((p>=0.8)::int), sum((eff>=0.8)::int), sum((eff>=0.7)::int), round(median(p),3) from mrp where not is_asg group by 1").fetchall(): print("  ", r)
print("assigned to that S1: p distribution")
print("  ", c.execute("select count(*), count(p), round(quantile_cont(p, 0.1),3), round(median(p),3), round(avg((eff<0.85)::int)*100,2) from mrp join vpf v using (rid) where mrp.is_asg and v.s1 = mrp.s1").fetchone())
