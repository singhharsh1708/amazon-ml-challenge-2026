"""Step 5 gate. Experiment/validation script g2_check.py for this step.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
    {SP}/push/frrescue/additions.parquet
    {SP}/push/frce/removals.parquet
    {SP}/push/frce/removals_asspec.parquet
    {SP}/push/frce/sc_a.parquet
    {SP}/push/frce/sc_b.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
import duckdb
import pyarrow as pa
G = f"{WORK_DIR}/push/gate"
SP = f"{WORK_DIR}"
sys.path.insert(0, f"{FINAL_STEPS}/france_push/frce")
from vd import review_name, review_address, review_key, review_verdict
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
c.execute(f"create or replace table A as select * from '{SP}/push/frrescue/additions.parquet'")
c.execute(f"create or replace table R as select * from '{SP}/push/frce/removals.parquet'")
c.execute(f"create or replace table R0 as select * from '{SP}/push/frce/removals_asspec.parquet'")
c.execute(f"create or replace table fce as select rid, s1, max(ce) ce from (select * from '{SP}/push/frce/sc_a.parquet' union all select * from '{SP}/push/frce/sc_b.parquet') group by all")
print("schemas", [c.execute(f"describe {x}").fetchall() for x in ("A", "R", "R0")])
for x in ("A", "R", "R0"):
    print(f"== {x}", c.execute(f"""select count(*), count(distinct rid), count(distinct (rid, s1)), count(distinct s1),
      count(*) filter (where rid is null or s1 is null) from {x}""").fetchone())
    print("  rid src/country", c.execute(f"select q.src, q.country, count(*) from {x} left join raw q on q.eid = {x}.rid group by all").fetchall())
    print("  s1 src/country", c.execute(f"select s.src, s.country, count(*) from {x} left join raw s on s.eid = {x}.s1 group by all").fetchall())
    print("  pair in v17", c.execute(f"select count(*) from {x} join v17 using (rid, s1)").fetchone(),
          "rid assigned in v17", c.execute(f"select count(*) from {x} where rid in (select rid from v17)").fetchone(),
          "pair in v17 candidate file", c.execute(f"select count(*) from {x} join cand using (rid, s1)").fetchone())
print("A rid overlap R rid", c.execute("select count(*) from A where rid in (select rid from R)").fetchone())
print("R subset of R0", c.execute("select count(*) from R join R0 using (rid, s1)").fetchone())
c.execute("create or replace table nv as select * from v17 where (rid, s1) not in (select (rid, s1) from R) union all select rid, s1 from A")
print("new file pairs, distinct rid", c.execute("select count(*), count(distinct rid) from nv").fetchone())
fr_s1 = "(select eid from raw where src = 1 and country = 'France')"
for nm_, t in (("v17", "v17"), ("v17-R+A", "nv"), ("v17-R", "(select * from v17 where (rid, s1) not in (select (rid, s1) from R))"), ("v17+A", "(select * from v17 union all select rid, s1 from A)")):
    print(f"empty French S1 in {nm_}", c.execute(f"select count(*) filter (where eid not in (select s1 from {t})), count(*) from {fr_s1} x").fetchone())
print("A: target S1 empty in v17", c.execute("select count(distinct s1) from A where s1 not in (select s1 from v17)").fetchone())
print("R: S1 emptied by R", c.execute("select count(distinct s1) from R where s1 not in (select s1 from v17 where (rid, s1) not in (select (rid, s1) from R))").fetchone())
c.execute("""create or replace table fp as select eid, name, address,
  regexp_matches(name, '[a-z]') and name = lower(name) lc, name like '% %' mw,
  regexp_matches(lower(name), '(\\.(com|fr|net|org|eu|biz|info)\\b)|^#|@|www') dom, name like '%.%' dot,
  try_cast(nullif(regexp_extract(address, '\\b([0-9]+)\\b', 1), '') as int) hn from raw where country = 'France'""")
def fpr(label, q):
    """Print the rate of each risk flag in a query result."""
    r = c.execute(f"""with x as ({q}) select count(*) n,
      round(100 * avg(qf.lc::int), 2) lc, round(100 * avg((qf.lc and qf.mw)::int), 2) lc_mw,
      round(100 * avg((qf.lc and qf.mw and not qf.dom)::int), 2) lc_mw_nodom, round(100 * avg(qf.dot::int), 2) dot,
      round(100 * avg((qf.dot and not qf.dom)::int), 2) dot_nodom, round(100 * avg(qf.dom::int), 2) dom,
      count(*) filter (where qf.hn is not null and sf.hn is not null) with_hn,
      count(*) filter (where qf.hn <> sf.hn) hn_diff,
      count(*) filter (where qf.hn - sf.hn in (1,2,3,4,5,7,9,11,13,21)) up_off,
      count(*) filter (where sf.hn - qf.hn in (1,2,3,4,5,7,9,11,13,21)) down_off
      from x join fp qf on qf.eid = x.rid join fp sf on sf.eid = x.s1""").fetchone()
    n, lc, lcmw, lcmwnd, dot, dotnd, dom, whn, hd, up, dn = r
    print(f"{label:48s} n={n:7d} lc={lc:5.2f}% lc_mw={lcmw:5.2f}% lc_mw_nodom={lcmwnd:5.2f}% dot={dot:5.2f}% dot_nodom={dotnd:5.2f}% dom={dom:5.2f}% hn_both={whn} hn_diff={hd} up_off={up} ({100*up/max(1,hd):.1f}% of diff) down_off={dn}")
print("== fingerprints (Q = S2/S3 raw name, house number = first number in raw address)")
fpr("v17 French all", "select rid, s1 from v17fr")
for v in ("same", "same_noise", "diff_typeswap", "unsure_disjoint", "diff_addr", "unsure_addword"):
    fpr(f"v17 French {v}", f"select rid, s1 from v17v where v = '{v}'")
fpr("R (frce removals)", "select rid, s1 from R")
fpr("R0 (as spec)", "select rid, s1 from R0")
fpr("R0 minus R", "select rid, s1 from R0 where (rid, s1) not in (select (rid, s1) from R)")
fpr("R0 minus R, diff_typeswap", "select rid, s1 from R0 join v17v using (rid, s1) where v = 'diff_typeswap' and (rid, s1) not in (select (rid, s1) from R)")
fpr("R0 minus R, diff_addr", "select rid, s1 from R0 join v17v using (rid, s1) where v = 'diff_addr'")
fpr("v17 diff_typeswap not in R", "select rid, s1 from v17v where v = 'diff_typeswap' and (rid, s1) not in (select (rid, s1) from R)")
fpr("v17 diff_typeswap frCE > -2", "select rid, s1 from v17v join fce using (rid, s1) where v = 'diff_typeswap' and ce > -2")
fpr("v17 diff_typeswap no frCE score", "select rid, s1 from v17v where v = 'diff_typeswap' and (rid, s1) not in (select (rid, s1) from fce)")
fpr("A (rescue additions)", "select rid, s1 from A")
print("R verdicts", c.execute("select v, count(*) from R left join v17v using (rid, s1) group by 1").fetchall())
print("R frCE", c.execute("select count(ce), min(ce), max(ce), median(ce) from R left join fce using (rid, s1)").fetchone())
print("v17 diff_typeswap frCE coverage", c.execute("select count(*), count(ce), count(*) filter (where ce <= -4), count(*) filter (where ce > -4 and ce <= -2), count(*) filter (where ce > -2) from v17v left join fce using (rid, s1) where v = 'diff_typeswap'").fetchone())
print("A frCE", c.execute("select count(ce), count(*) filter (where ce >= 0), count(*) filter (where ce >= 2), median(ce) from A left join fce using (rid, s1)").fetchone())
