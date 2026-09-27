"""Step 5 (France push). Experiment/validation script a4_sib.py for this step.

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
from collections import Counter
W = f"{WORK_DIR}/push/frce"
sys.path.insert(0, W)
from vd import review_address
DEL = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
c = duckdb.connect(); c.execute("SET threads=3; SET memory_limit='2GB'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
c.execute(f"create table fr as select * from '{W}/sc_a.parquet'")
c.execute("""create table s1n as select eid, name_full, try_cast(regexp_extract(ak, '\\b([0-9]+)\\b', 1) as int) num,
  regexp_replace(ak, '\\b[0-9]+\\b', '', 'g') st from w.nm where eid < 20000000000""")
rows = c.execute("""select v.rid, v.s1, q.ak, s.ak, fr.ce,
  exists (select 1 from s1n z where z.eid <> v.s1 and z.name_full = s.name_full
     and z.num = try_cast(regexp_extract(q.ak, '\\b([0-9]+)\\b', 1) as int)) sib
  from w.v17v v join fr using (rid, s1) join w.nm q on q.eid = v.rid join w.nm s on s.eid = v.s1
  where v.v = 'diff_addr' and fr.ce <= -2""").fetchall()
neg = Counter(); sib = Counter(); tot = Counter()
for rid, s1, qa, sa, ce, sb in rows:
    d = review_address(qa, sa)[1]
    g = "none" if d is None else ("pos_set" if d > 0 and d in DEL else ("pos_other" if d > 0 else ("neg_set" if -d in DEL else "neg_other")))
    tot[g] += 1; sib[g] += int(sb)
print("v17 diff_addr frCE<=-2: delta group -> n, S1-sibling-with-same-name-at-q-number")
for g in tot: print("  ", g, tot[g], sib[g])
