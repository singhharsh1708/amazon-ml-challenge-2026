"""Step 7 gate. Experiment/validation script g6.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {S}/fr_rerun/work/new/scores.parquet
    {S}/fill/frfill/additions_p07_alt.parquet
    {S}/fill/additions.parquet
    {S}/fill/frfill/pool_p05.parquet
    {G}/t2.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
sys.path.insert(0, f"{REPO_DIR}/src"); sys.path.insert(0, f"{FINAL_STEPS}/france_rerun/src")
import duckdb, numpy as np
from decoy_veto import build_vocab, load_decoy_words, veto_flags
c = duckdb.connect()
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
ND = f"{S}/fr_rerun/data/fr_new"
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
c.execute(f"create table names as select {ID} eid, country, name_full from read_parquet('{ND}/test_s*.parquet')")
c.execute(f"""create table g as select rid, arg_max(s1, (p_guard, -s1)) filter (where p_guard is not null) g_s1, max(p_guard) g_p
   from '{S}/fr_rerun/work/new/scores.parquet' where rid in (select rid from '{S}/fill/frfill/additions_p07_alt.parquet') group by rid""")
c.execute(f"""create table t as select a.rid, a.s1, sc.p, sc.p_guard, g.g_s1, g.g_p, (g.g_s1 = a.s1) gagree,
   case when g.g_s1 = a.s1 then least(sc.p, g.g_p) else 0 end eff, pl.ce, (a.rid in (select rid from '{S}/fill/additions.parquet')) in08,
   q.country, q.name_full qn, s.name_full sn
   from '{S}/fill/frfill/additions_p07_alt.parquet' a join '{S}/fr_rerun/work/new/scores.parquet' sc using (rid, s1) join g using (rid)
   left join '{S}/fill/frfill/pool_p05.parquet' pl using (rid, s1) join names q on q.eid = a.rid join names s on s.eid = a.s1""")
d = c.execute("select rid, s1, p, country, qn, sn from t").fetchnumpy()
s1n = c.execute("select country, name_full from names where eid // 10000000000 = 1").fetchnumpy()
dh, sh = veto_flags(d["country"], d["qn"], d["sn"], d["p"], build_vocab(s1n["country"], s1n["name_full"]), load_decoy_words(), set())
import pyarrow as pa
c.register("vf", pa.table({"rid": d["rid"], "decoy_hit": dh, "swap_hit": sh}))
c.execute("create table t2 as select * from t join vf using (rid)")
c.execute(f"copy t2 to '{G}/t2.parquet' (format parquet)")
print("band, n, guard argmax agrees, eff>=0.85, eff>=0.7, eff>=0.5, decoy_hit, swap_hit")
for r in c.execute("select in08, count(*), sum(gagree::int), sum((eff>=0.85)::int), sum((eff>=0.7)::int), sum((eff>=0.5)::int), sum(decoy_hit::int), sum(swap_hit::int) from t2 group by 1").fetchall(): print("  ", r)
print("shipped rows sorted by eff:")
for r in c.execute("select round(p,3), round(p_guard,3), round(g_p,3), gagree, round(eff,3), round(ce,2), decoy_hit, swap_hit, sn, '<-', qn from t2 where in08 order by eff desc").fetchall(): print("  ", r)
