"""Step 5 (France push). Experiment/validation script a2_base.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frce
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
for name, q in (("v17 diff_addr", "select rid, s1, v from w.v17v where v = 'diff_addr'"), ("v17 unsure_num", "select rid, s1, v from w.v17v where v = 'unsure_num'"),
                ("band unasg p>=.85 diff_addr", "select rid, s1, v from w.bandv where p >= 0.85 and not rid_asg and v = 'diff_addr'"),
                ("band unasg p>=.85 unsure_num", "select rid, s1, v from w.bandv where p >= 0.85 and not rid_asg and v = 'unsure_num'")):
    rows = c.execute(f"select {review_key('q.address')}, {review_key('s.address')}, rw.lc from ({q}) x join w.nm q on q.eid=x.rid join w.nm s on s.eid=x.s1 join w.rw rw on rw.eid=x.rid").fetchall()
    dd = [review_address(a, b)[1] for a, b, _ in rows]; dd = [x for x in dd if x is not None]
    pos = sum(1 for x in dd if x > 0 and x in DEL); neg = sum(1 for x in dd if x < 0)
    print(f"{name}: n={len(rows)} lc={100*sum(r[2] for r in rows)/max(1,len(rows)):.2f}% with delta={len(dd)} pos-offset={100*pos/max(1,len(dd)):.1f}% neg={100*neg/max(1,len(dd)):.1f}%")
