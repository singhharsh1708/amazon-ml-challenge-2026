"""Step 7 gate. Experiment/validation script g1.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/student_resource/dataset/test
    $REPO_DIR/output/v20/matching_results_v20.tsv
    {R}/test_source1.tsv
    {R}/test_source2.tsv
    {R}/test_source3.tsv
    {S}/fill/additions.parquet
    {S}/fill/frfill/additions_p07_alt.parquet
    {S}/push/gate/additions_trim.parquet
    {S}/push/frce/removals.parquet
    {G}/j_p07.parquet
    {G}/j_p08.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, re
sys.dont_write_bytecode = True
import duckdb
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
R = f"{REPO_DIR}/student_resource/dataset/test"
c = duckdb.connect(f"{G}/tmp/g.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
ID = "(cast(substr(X, 2, 1) as bigint) * 10000000000 + cast(substr(X, 4) as bigint))"
c.execute(f"create or replace table v20 as select * from read_csv('{REPO_DIR}/output/v20/matching_results_v20.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')")
print("v20 rows, distinct s1", c.execute("select count(*), count(distinct source1_entity_id) from v20").fetchone())
c.execute(f"create or replace table vp as select source1_entity_id s1s, trim(x) rs from (select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) x from v20 where coalesce(matched_entity_ids,'') <> '')")
c.execute(f"create or replace table asg as select distinct {ID.replace('X','rs')} rid from vp")
print("assigned pairs, distinct records", c.execute("select count(*), count(distinct rs) from vp").fetchone(), c.execute("select count(*) from asg").fetchone())
c.execute(f"create or replace table emp as select {ID.replace('X','source1_entity_id')} s1 from v20 where coalesce(matched_entity_ids,'') = ''")
c.execute(f"""create or replace table raw as select {ID.replace('X','entity_id')} eid, entity_id, business_name bn, business_address ba, country from (
  select * from read_csv('{R}/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
  union all select * from read_csv('{R}/test_source2.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
  union all select * from read_csv('{R}/test_source3.tsv', delim='\t', header=true, all_varchar=true, quote='', escape=''))""")
print("raw rows, distinct eid, countries", c.execute("select count(*), count(distinct eid) from raw").fetchone(), c.execute("select country, count(*) from raw group by 1 order by 2 desc").fetchall())
print("french S1 total, french S1 empty in v20", c.execute("select count(*) from raw where eid < 20000000000 and lower(country) like 'fr%'").fetchone(), c.execute("select count(*) from emp join raw r on r.eid = emp.s1 where lower(r.country) like 'fr%'").fetchone())
for name, f in (("p08", f"{S}/fill/additions.parquet"), ("p07", f"{S}/fill/frfill/additions_p07_alt.parquet")):
    c.execute(f"create or replace table a_{name} as select * from '{f}'")
    t = f"a_{name}"
    print("==", name, f)
    print(" rows, distinct rid, distinct s1, null", c.execute(f"select count(*), count(distinct rid), count(distinct s1), sum((rid is null or s1 is null)::int) from {t}").fetchone())
    print(" s1 are source1 ids, rids are S2/S3", c.execute(f"select sum((s1 // 10000000000 = 1)::int), sum((rid // 10000000000 in (2,3))::int) from {t}").fetchone())
    print(" rid already assigned in v20", c.execute(f"select count(*) from {t} where rid in (select rid from asg)").fetchone())
    print(" s1 present in v20 file / empty in v20", c.execute(f"select count(*) from {t} a join (select {ID.replace('X','source1_entity_id')} s1 from v20) v using (s1)").fetchone(), c.execute(f"select count(*) from {t} where s1 in (select s1 from emp)").fetchone())
    print(" s1 country / rid country", c.execute(f"select r.country, count(*) from {t} a join raw r on r.eid = a.s1 group by 1").fetchall(), c.execute(f"select r.country, count(*) from {t} a join raw r on r.eid = a.rid group by 1").fetchall())
    print(" rid found in raw / s1 found in raw", c.execute(f"select count(*) from {t} a join raw r on r.eid = a.rid").fetchone(), c.execute(f"select count(*) from {t} a join raw r on r.eid = a.s1").fetchone())
    print(" rid source split", c.execute(f"select rid // 10000000000, count(*) from {t} group by 1 order by 1").fetchall())
    print(" rid in rescue additions / removals", c.execute(f"select count(*) from {t} where rid in (select rid from '{S}/push/gate/additions_trim.parquet')").fetchone(), c.execute(f"select count(*) from {t} where rid in (select rid from '{S}/push/frce/removals.parquet')").fetchone())
    c.execute(f"create or replace table j_{name} as select a.rid, a.s1, q.entity_id qid, q.bn qn, q.ba qa, s.entity_id sid, s.bn sn, s.ba sa from {t} a join raw q on q.eid = a.rid join raw s on s.eid = a.s1")
c.execute(f"copy (select * from j_p07) to '{G}/j_p07.parquet' (format parquet)")
c.execute(f"copy (select * from j_p08) to '{G}/j_p08.parquet' (format parquet)")
print("p08 subset of p07", c.execute("select count(*) from a_p08 a join a_p07 b using (rid, s1)").fetchone())
