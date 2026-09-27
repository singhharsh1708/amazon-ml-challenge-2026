"""Step 7 gate. Experiment/validation script g5.py for this step.

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
c = duckdb.connect(f"{G}/tmp/g.db", read_only=True)
c.execute(f"SET threads=3; SET memory_limit='2GB'")
q = f"""select sc.s1, s.bn, sc.rid, q.bn, q.ba, round(sc.p,3), (sc.rid in (select rid from asg)) asg
 from '{S}/fr_rerun/work/new/scores.parquet' sc join frs1 s on s.eid=sc.s1 join frrec q on q.eid=sc.rid where sc.s1 in (?, ?) or sc.rid in (?, ?, ?) order by sc.s1, sc.p desc"""
for r in c.execute(q, [10921554309, 10218608617, 30354939977, 20898121723, 30585844099]).fetchall(): print(r)
print("Lille Club SAS 12 Lyderic group:")
for r in c.execute(q, [10621985812, 10621985812, 30784614876, 30784614876, 30784614876]).fetchall(): print(r)
print("Chasse Club Tourcoing:")
for r in c.execute("select eid, bn, ba, is_emp from frs1 where lower(bn) like 'chasse club%' and lower(ba) like '%tourcoing%'").fetchall(): print(r)
for r in c.execute("select eid, bn, ba, (eid in (select rid from asg)) from frrec where lower(bn) like 'chasse club%' and lower(ba) like '%tourcoing%'").fetchall(): print(r)
