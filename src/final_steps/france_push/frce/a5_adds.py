"""Step 5 (France push). Experiment/validation script a5_adds.py for this step.

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
from vd import review_name
SUF = {"fils", "associes", "cie", "services", "developpement", "groupe", "france", "frs", "freres", "et", "and", "compagnie"}
c = duckdb.connect(); c.execute("SET threads=3; SET memory_limit='2GB'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
c.execute(f"create table fr as select rid, s1, any_value(ce) ce from read_parquet(['{W}/sc_a.parquet']) group by all")
rows = c.execute("""select x.rid, x.s1, x.v, x.ce, q.name_full, s.name_full, rw.lc from (
   select rid, arg_max(s1, ce) s1, max(ce) ce, arg_max(v, ce) v from w.bandv b join fr f using (rid, s1) where p >= 0.85 and ce >= 0 and not rid_asg group by rid) x
  join w.nm q on q.eid = x.rid join w.nm s on s.eid = x.s1 join w.rw rw on rw.eid = x.rid""").fetchall()
g = Counter(); l = Counter()
for rid, s1, v, ce, qn, sn, lc in rows:
    nc, nadd, nrm = review_name(qn, sn)
    aw = [w for w in nadd.split() if w]
    k = (v, "add_suffix_only" if aw and all(w in SUF for w in aw) else ("no_added_word" if not aw else "other_added"))
    g[k] += 1; l[k] += lc
print("additions (rule as specified) by verdict x added-word type: n, lc%")
for k in sorted(g): print("  ", k, g[k], round(100 * l[k] / g[k], 2))
