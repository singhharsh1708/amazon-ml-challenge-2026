"""Step 7 (France empty-S1 fill). Experiment/validation script b4.py for this step.

Step context: Step 7 (France empty-S1 fill). Candidate pool for France source-1 records that ended with no match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/fill/frfill
    /tmp/g4.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
W = f"{WORK_DIR}/fill/frfill"
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill'")
c.execute("create or replace table nmc as select name_full, count(*) k from nm where eid < 20000000000 group by 1")
c.execute("""create or replace table g4 as select g.*, coalesce(k.k, 0) nsame_s1, (select count(*) from nm n where n.name_full = s.name_full and n.eid >= 20000000000) nsame_rec
   from g2 g join nm s on s.eid = g.s1 left join nmc k on k.name_full = s.name_full where coalesce(rm_ok, true) and p >= 0.5""")
print("nsame_s1 dist", c.execute("select nsame_s1, count(*), sum((ce>=0)::int), count(ce) from g4 group by 1 order by 1").fetchall())
print("nsame_rec dist", c.execute("select least(nsame_rec,5), count(*), sum((ce>=0)::int), count(ce) from g4 group by 1 order by 1").fetchall())
c.execute("copy (select * from g4) to '" + W + "/tmp/g4.parquet'")
for th in (0.5, 0.6, 0.7, 0.8):
    print(f"==== band p>={th}")
    rows = c.execute(f"select round(p,2), round(ce,1), nsame_s1, nsame_rec, qbn, '|', coalesce(qba,'-'), '||', sbn, '|', sba from g4 where p >= {th} order by hash(rid * 7 + 911) limit 30").fetchall()
    for i, r in enumerate(rows): print(i+1, *r)
