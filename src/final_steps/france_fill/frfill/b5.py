"""Step 7 (France empty-S1 fill). Experiment/validation script b5.py for this step.

Step context: Step 7 (France empty-S1 fill). Candidate pool for France source-1 records that ended with no match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {S}/push/gate/additions_trim.parquet
    {S}/push/frce/removals.parquet
    {S}/fill/additions.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
S = f"{WORK_DIR}"
W = f"{S}/fill/frfill"
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill'")
c.execute("create or replace table add8 as select * from g4 where p >= 0.8")
print("n, distinct rid, distinct s1", c.execute("select count(*), count(distinct rid), count(distinct s1) from add8").fetchone())
print("rid in v20", c.execute("select count(*) from add8 where rid in (select rid from asg)").fetchone())
print("s1 empty french", c.execute("select count(*) from add8 where s1 in (select s1 from femp)").fetchone())
print("rid source", c.execute("select rid // 10000000000, count(*) from add8 group by 1").fetchall())
print("p_other>=0.5p (any competitor)", c.execute("select count(*) from add8 where coalesce(p_other,0) >= 0.5*p").fetchone())
print("ac", c.execute("select ac, count(*) from add8 group by 1").fetchall(), "v", c.execute("select v, count(*) from add8 group by 1").fetchall())
print("ce cov / >=0 / min", c.execute("select count(ce), sum((ce>=0)::int), min(ce), median(ce) from add8").fetchone())
print("in rescue additions or removals", c.execute(f"select count(*) from add8 where rid in (select rid from '{S}/push/gate/additions_trim.parquet') or rid in (select rid from '{S}/push/frce/removals.parquet')").fetchone())
print("not in sample of 30:")
for r in c.execute("select round(p,2), round(ce,1), nsame_s1, qbn, '|', coalesce(qba,'-'), '||', sbn, '|', sba from add8 where rid not in (select rid from add8 order by hash(rid * 7 + 911) limit 30)").fetchall(): print(r)
c.execute(f"copy (select rid, s1::bigint s1 from add8 order by rid) to '{S}/fill/additions.parquet' (format parquet)")
print("written", c.execute(f"select count(*), count(distinct rid), typeof(any_value(s1)) from '{S}/fill/additions.parquet'").fetchone())
