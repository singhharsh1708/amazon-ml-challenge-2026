"""Step 5 gate. Experiment/validation script g3_look.py for this step.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Command-line arguments used: argv[1], argv[2], argv[3].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import duckdb
G = f"{WORK_DIR}/push/gate"
SP = f"{WORK_DIR}"
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
tab, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = c.execute(f"""select x.rid, x.s1, q.name, q.address, s.name, s.address, fce.ce
  from {tab} x join raw q on q.eid = x.rid join raw s on s.eid = x.s1 left join fce using (rid, s1)
  order by hash(x.rid, {seed}) limit {n}""").fetchall()
for i, (rid, s1, qn, qa, sn, sa, ce) in enumerate(rows, 1):
    print(f"[{i}] rid={rid} s1={s1} frCE={ce}")
    print(f"   Q : {qn} | {qa}")
    print(f"   S1: {sn} | {sa}")
    sib = c.execute(f"select q.name, q.address from v17 v join raw q on q.eid = v.rid where v.s1 = {s1} and v.rid <> {rid} limit 4").fetchall()
    for a, b in sib:
        print(f"   v17 other: {a} | {b}")
