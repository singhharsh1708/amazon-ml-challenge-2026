"""Step 5 (France push). Experiment/validation script s2_filter.py for this step.

Step context: Step 5 (France push). France rescue additions for records left without a match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frrescue
    {R}/test_rescue.parquet
    {R}/test_rescue_ce1.parquet
    {R}/test_rescue_ce2.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
W = f"{WORK_DIR}/push/frrescue"
SP = f"{WORK_DIR}"
c = duckdb.connect(f"{W}/work.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp'")
R = f"{SP}/rescue"
c.execute(f"""create or replace table pool0 as
  select t.rid, t.s1, t.family, t.q_text, t.s_text, a.ce as ce1, b.ce as ce2, (a.ce + b.ce) / 2 as ce
  from '{R}/test_rescue.parquet' t
  join '{R}/test_rescue_ce1.parquet' a using (rid, s1)
  join '{R}/test_rescue_ce2.parquet' b using (rid, s1)
  where t.country = 'france'""")
print("french rescue pairs", c.execute("select count(*), count(distinct rid) from pool0").fetchone())
print("dup keys", c.execute("select count(*) from (select rid, s1 from pool0 group by all having count(*) > 1)").fetchone())
print("s1 in fr_new S1", c.execute("select count(*) filter (where s1 in (select eid from nm where eid//10000000000=1)), count(*) filter (where rid in (select eid from nm where eid//10000000000>1)) from pool0").fetchone())
c.execute("create or replace table pool1 as select * from pool0 where rid not in (select rid from v17)")
print("unassigned in v17", c.execute("select count(*), count(distinct rid) from pool1").fetchone())
c.execute("create or replace table pool2 as select * from pool1 where ce >= 0.95")
print("ce>=0.95", c.execute("select count(*), count(distinct rid), count(distinct s1) from pool2").fetchone())
print("families", c.execute("select family, count(*) from pool2 group by 1 order by 2 desc").fetchall())
print("ce>=0.95 by s1 v17-empty", c.execute("select s.empty, count(*), count(distinct p.rid) from pool2 p join v17s1 s using (s1) group by 1").fetchall())
