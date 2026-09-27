"""Step 5 gate. Experiment/validation script g4_fp.py for this step.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.dont_write_bytecode = True
import duckdb
G = f"{WORK_DIR}/push/gate"
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
exec(open(f"{G}/g2_check.py").read().split("print(\"== fingerprints")[0].split("def fpr")[1].join(["def fpr", ""]) if False else "")
def fpr(label, q):
    """Print the rate of each risk flag in a query result."""
    r = c.execute(f"""with x as ({q}) select count(*) n,
      round(100 * avg(qf.lc::int), 2), round(100 * avg((qf.lc and qf.mw)::int), 2),
      round(100 * avg((qf.lc and qf.mw and not qf.dom)::int), 2), round(100 * avg(qf.dot::int), 2),
      round(100 * avg((qf.dot and not qf.dom)::int), 2), round(100 * avg(qf.dom::int), 2),
      count(*) filter (where qf.hn is not null and sf.hn is not null),
      count(*) filter (where qf.hn <> sf.hn),
      count(*) filter (where qf.hn - sf.hn in (1,2,3,4,5,7,9,11,13,21)),
      count(*) filter (where sf.hn - qf.hn in (1,2,3,4,5,7,9,11,13,21))
      from x join fp qf on qf.eid = x.rid join fp sf on sf.eid = x.s1""").fetchone()
    n, lc, lcmw, lcmwnd, dot, dotnd, dom, whn, hd, up, dn = r
    if not n:
        print(f"{label:44s} n=0"); return
    print(f"{label:44s} n={n:7d} lc={lc:5.2f}% lc_mw={lcmw:5.2f}% lc_mw_nodom={lcmwnd:5.2f}% dot={dot:5.2f}% dot_nodom={dotnd:5.2f}% dom={dom:5.2f}% hn_both={whn} hn_diff={hd} up_off={up} down_off={dn}")
for label, q in [
    ("R lev<=2", "select rid, s1 from rw where lev <= 2"),
    ("R lev>2", "select rid, s1 from rw where lev > 2"),
    ("A all", "select rid, s1 from A"),
    ("A non-domain", "select A.rid, A.s1 from A join fp on fp.eid = A.rid where not fp.dom and fp.name not like '#%'"),
    ("A frCE<0", "select rid, s1 from A join fce using (rid, s1) where ce < 0"),
    ("v17 French same, non-domain", "select v.rid, v.s1 from v17v v join fp on fp.eid = v.rid where v = 'same' and not fp.dom"),
]:
    fpr(label, q)
print("A dom/hashtag/plain split", c.execute("""select case when fp.dom then 'domain' when fp.name like '#%' or fp.name like '@%' then 'handle' when fp.name not like '% %' then 'one_token' else 'multi_word' end k,
  count(*), sum(fp.lc::int) from A join fp on fp.eid = A.rid group by 1 order by 2 desc""").fetchall())
print("v17 French Q-name split", c.execute("""select case when fp.dom then 'domain' when fp.name like '#%' or fp.name like '@%' then 'handle' when fp.name not like '% %' then 'one_token' else 'multi_word' end k,
  count(*) from v17fr v join fp on fp.eid = v.rid group by 1 order by 2 desc""").fetchall())
