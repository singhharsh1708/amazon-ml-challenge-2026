"""Writes the France push removal set.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frce
    $REPO_DIR/student_resource/dataset/test
    {W}/sc_a.parquet
    {R}/test_source1.tsv
    {R}/test_source2.tsv
    {R}/test_source3.tsv
    {W}/labels60.parquet
    {W}/removals.parquet
    {W}/removals_asspec.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.dont_write_bytecode = True
import duckdb
W = f"{WORK_DIR}/push/frce"
R = f"{REPO_DIR}/student_resource/dataset/test"
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
c = duckdb.connect(); c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill4'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
c.execute(f"create table fr as select rid, s1, any_value(ce) ce from read_parquet(['{W}/sc_a.parquet']) group by all")
c.execute(f"""create table raw as select {ID} eid, business_name bn, business_address ba,
  (business_name = lower(business_name) and regexp_matches(business_name, '[a-z]'))::int lc, contains(business_name, '.')::int dot
  from read_csv(['{R}/test_source1.tsv','{R}/test_source2.tsv','{R}/test_source3.tsv'], delim='\\t', header=true, all_varchar=true, quote='', escape='')
  where lower(country)='france'""")
c.execute("create table spec as select v.rid, v.s1, v.v, f.ce from w.v17v v join fr f using (rid, s1) where f.ce <= -2 and v.v like 'diff%'")
c.execute("create table ref as select * from spec where v = 'diff_typeswap' and ce <= -4")
c.execute("create table adds_spec as select rid, arg_max(s1, ce) s1, max(ce) ce, arg_max(v, ce) v from w.bandv b join fr f using (rid, s1) where p >= 0.85 and ce >= 0 and not rid_asg group by rid")
c.execute("create table adds_ref as select * from adds_spec where v in ('same', 'same_noise')")
for t in ("spec", "ref", "adds_spec", "adds_ref"):
    n, lc, dot, dv, sv = c.execute(f"select count(*), avg(r.lc), avg(r.dot), avg((x.v like 'diff%')::int), avg((x.v in ('same','same_noise'))::int) from {t} x join raw r on r.eid=x.rid").fetchone()
    dup = c.execute(f"select count(*) - count(distinct rid) from {t}").fetchone()[0]
    print(f"{t}: n={n} lc={100*lc:.2f}% dot={100*dot:.2f}% diff_*={100*dv:.1f}% same/same_noise={100*sv:.1f}% dup_rids={dup}")
lab = c.execute(f"select l.lab, count(*) from '{W}/labels60.parquet' l join ref using (rid, s1) group by 1").fetchall()
print("hand labels inside refined removals:", lab)
print("hand labels inside spec removals:", c.execute(f"select l.lab, count(*) from '{W}/labels60.parquet' l join spec using (rid, s1) group by 1").fetchall())
print("hand labels inside refined additions:", c.execute(f"select l.lab, count(*) from '{W}/labels60.parquet' l join adds_ref using (rid, s1) group by 1").fetchall())
print("hand-labelled pairs in v17:", c.execute(f"select l.lab, count(*) from '{W}/labels60.parquet' l join w.v17fr using (rid, s1) group by 1").fetchall())
print("refined removals: 20 random examples")
for r in c.execute("""select x.v, round(x.ce, 2), q.bn, q.ba, s.bn, s.ba from ref x join raw q on q.eid = x.rid join raw s on s.eid = x.s1 order by hash(x.rid, 99) limit 20""").fetchall():
    print(f"  [{r[0]} ce={r[1]}] Q: {r[2]} | {r[3]}  ||  S1: {r[4]} | {r[5]}")
if "--write" in sys.argv:
    c.execute(f"copy (select rid, s1 from ref order by rid) to '{W}/removals.parquet'")
    c.execute(f"copy (select rid, s1 from spec order by rid) to '{W}/removals_asspec.parquet'")
    print("wrote removals.parquet", c.execute(f"select count(*) from '{W}/removals.parquet'").fetchone(), "removals_asspec.parquet", c.execute(f"select count(*) from '{W}/removals_asspec.parquet'").fetchone())
    print("check removals subset of v17:", c.execute(f"select count(*) from '{W}/removals.parquet' x anti join w.v17fr using (rid, s1)").fetchone())
