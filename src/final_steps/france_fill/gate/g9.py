"""Step 7 gate. Experiment/validation script g9.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    /t2.parquet
    {G}/additions_strict.parquet
    {S}/fill/additions.parquet
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
drop = {30629911482: "Lille Groupe <- Lille Compagnie: type-word swap, guard 0.595",
        30784614876: "Lille Club SAS <- Lille Club SCI: non-empty S1 Lille Club SARL at same 12 Rue Lyderic, no frCE",
        30527450118: "Sapeurs Danse <- Sapeurs Danse no-address: 4 other Sapeurs Danse S1 (2 empty), guard 0.42",
        30354939977: "Lille Club <- Lille Club 27 Jean Bart: empty S1 Lille Club SAS at same address"}
c.execute("create or replace table t2 as select * from '" + G + "/t2.parquet'")
c.execute(f"create or replace table strict as select rid, s1::bigint s1 from t2 where in08 and ce >= 0 and rid not in ({','.join(map(str, drop))}) order by rid")
c.execute(f"copy strict to '{G}/additions_strict.parquet' (format parquet)")
f = f"{G}/additions_strict.parquet"
print("strict rows, rid, s1", c.execute(f"select count(*), count(distinct rid), count(distinct s1), typeof(any_value(s1)), typeof(any_value(rid)) from '{f}'").fetchone())
print("subset of shipped", c.execute(f"select count(*) from '{f}' a join '{S}/fill/additions.parquet' b using (rid, s1)").fetchone())
print("rid in v20", c.execute(f"select count(*) from '{f}' where rid in (select rid from asg)").fetchone(), "s1 empty in v20", c.execute(f"select count(*) from '{f}' where s1 in (select s1 from emp)").fetchone())
print("French both sides", c.execute(f"select count(*) from '{f}' a join raw q on q.eid=a.rid join raw s on s.eid=a.s1 where q.country='France' and s.country='France'").fetchone())
print("frCE >=0 / scored, min, median; eff min", c.execute(f"select sum((ce>=0)::int), count(ce), round(min(ce),2), round(median(ce),2), round(min(eff),3), round(min(p),3) from t2 where rid in (select rid from '{f}')").fetchone())
print("source split", c.execute(f"select rid // 10000000000, count(*) from '{f}' group by 1 order by 1").fetchall())
print("dropped from shipped 31:")
for r in c.execute(f"select rid, s1, round(ce,2) from t2 where in08 and rid not in (select rid from '{f}') order by rid").fetchall(): print("  ", r, drop.get(r[0], "frCE < 0"))
print("fingerprints strict:", c.execute(f"""select count(*), sum((r.bn = lower(r.bn) and regexp_matches(r.bn,'[a-z]'))::int), sum((r.bn = lower(r.bn) and regexp_matches(r.bn,'[a-z]') and lower(r.bn) <> lower(s.bn))::int), sum((r.bn like '%.%')::int), sum((coalesce(trim(r.ba),'')='')::int)
   from '{f}' a join raw r on r.eid=a.rid join raw s on s.eid=a.s1""").fetchone())
