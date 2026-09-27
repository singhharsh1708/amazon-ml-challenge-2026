"""Step 5 (France push). Experiment/validation script a3_bins.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frce
    {W}/sc_a.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.dont_write_bytecode = True
import duckdb
W = f"{WORK_DIR}/push/frce"
sys.path.insert(0, W)
from vd import review_address, review_key
DEL = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
c = duckdb.connect(); c.execute("SET threads=3; SET memory_limit='2GB'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
c.execute(f"create table fr as select * from '{W}/sc_a.parquet'")
B = "case when ce <= -4 then 'a<=-4' when ce <= -2 then 'b(-4,-2]' when ce < 0 then 'c(-2,0)' when ce < 2 then 'd[0,2)' when ce < 4 then 'e[2,4)' else 'f>=4' end"
print("v17 diff_* pairs by verdict x frCE bin: n, lc%, dot%")
for r in c.execute(f"select v.v, {B} b, count(*), round(100*avg(rw.lc),2), round(100*avg(rw.dot),2) from w.v17v v join fr using (rid, s1) join w.rw rw on rw.eid=v.rid where v.v like 'diff%' group by all order by 1, 2").fetchall(): print("  ", r)
print("band p>=0.85 unassigned by verdict x frCE bin: n, lc%, dot%")
for r in c.execute(f"select v.v, {B} b, count(*), round(100*avg(rw.lc),2), round(100*avg(rw.dot),2) from w.bandv v join fr using (rid, s1) join w.rw rw on rw.eid=v.rid where v.p >= 0.85 and not v.rid_asg and v.v in ('same','same_noise','diff_typeswap','diff_addr','unsure_addr') group by all order by 1, 2").fetchall(): print("  ", r)
rows = c.execute(f"""select {B} b, {review_key('q.address')}, {review_key('s.address')}, exists (select 1 from w.nm s2 where s2.eid < 20000000000 and s2.eid <> v.s1 and s2.name_full = q.name_full and s2.address = q.address) sib
  from w.v17v v join fr using (rid, s1) join w.nm q on q.eid=v.rid join w.nm s on s.eid=v.s1 where v.v = 'diff_addr'""").fetchall()
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0, 0, 0])
for b, qa, sa, sib in rows:
    d = review_address(qa, sa)[1]; a = agg[b]; a[0] += 1; a[4] += int(sib)
    if d is not None:
        a[1] += 1; a[2] += int(d > 0 and d in DEL); a[3] += int(d < 0)
print("v17 diff_addr by frCE bin: n, with_delta, pos_offset, negative, exact-S1-sibling-exists")
for b in sorted(agg): print("  ", b, agg[b])
