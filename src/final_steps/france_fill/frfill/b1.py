"""Step 7 (France empty-S1 fill). Experiment/validation script b1.py for this step.

Step context: Step 7 (France empty-S1 fill). Candidate pool for France source-1 records that ended with no match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/student_resource/dataset/test
    $REPO_DIR/output/v20/matching_results_v20.tsv
    {S}/fr_rerun/work/new/scores.parquet
    {S}/push/frce/sc_a.parquet
    {S}/push/frce/sc_b.parquet
    {S}/push/frce/sc_h.parquet
    {R}/test_source{s}.tsv
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
import duckdb, pyarrow as pa
S = f"{WORK_DIR}"
W = f"{S}/fill/frfill"
sys.path.insert(0, f"{FINAL_STEPS}/france_push/frce")
from vd import review_name, review_address, review_key, review_verdict
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
R = f"{REPO_DIR}/student_resource/dataset/test"
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill'; SET preserve_insertion_order=false")
c.execute(f"create or replace table v20 as select * from read_csv('{REPO_DIR}/output/v20/matching_results_v20.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')")
c.execute(f"create or replace table asg as select distinct {ID.replace('entity_id','x')} as rid from (select unnest(string_split(matched_entity_ids, ',')) x from v20 where matched_entity_ids is not null and matched_entity_ids <> '')")
c.execute(f"create or replace table nm as select {ID} eid, name_full, {review_key('address')} ak, address_missing from read_parquet('{S}/fr_rerun/data/fr_new/test_s*.parquet')")
c.execute(f"create or replace table fs1 as select eid from nm where eid < 20000000000")
c.execute(f"create or replace table emp as select {ID.replace('entity_id','source1_entity_id')} s1 from v20 where matched_entity_ids is null or matched_entity_ids = ''")
c.execute("create or replace table femp as select s1 from emp where s1 in (select eid from fs1)")
print("v20 rows", c.execute("select count(*) from v20").fetchone(), "assigned records", c.execute("select count(*) from asg").fetchone())
print("french S1", c.execute("select count(*) from fs1").fetchone(), "french empty", c.execute("select count(*) from femp").fetchone())
c.execute(f"create or replace table sc as select rid, s1, p from '{S}/fr_rerun/work/new/scores.parquet'")
c.execute("create or replace table cand as select s.* from sc s where s.s1 in (select s1 from femp) and s.rid not in (select rid from asg) and s.p >= 0.3")
print("cand pairs p>=0.3", c.execute("select count(*), count(distinct rid), count(distinct s1) from cand").fetchone())
c.execute("create or replace table best as select rid, arg_max(s1, p) s1, max(p) p from cand group by rid")
c.execute("""create or replace table comp as select b.rid, b.s1, b.p,
   max(case when o.s1 <> b.s1 then o.p end) p_other
   from best b join sc o on o.rid = b.rid group by all""")
c.execute("""create or replace table comp2 as select b.rid, b.s1, max(o.p) p_samename from best b join sc o on o.rid = b.rid and o.s1 <> b.s1
   join nm n1 on n1.eid = b.s1 join nm n2 on n2.eid = o.s1 where n1.name_full = n2.name_full and n1.ak <> n2.ak group by all""")
c.execute("create or replace table b2 as select c.*, coalesce(c2.p_samename, 0) p_samename from comp c left join comp2 c2 using (rid, s1)")
t = c.execute("select b.rid, b.s1, q.name_full qn, s.name_full sn, q.ak qa, s.ak sa from b2 b join nm q on q.eid = b.rid join nm s on s.eid = b.s1").to_arrow_table()
cols = [t.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")]
rows = [review_name(a, b) + review_address(x, y) for a, b, x, y in zip(*cols)]
o = pa.table({"rid": t.column("rid"), "s1": t.column("s1"), "nc": [r[0] for r in rows], "nadd": [r[1] for r in rows], "nrm": [r[2] for r in rows],
   "ac": [r[3] for r in rows], "delta": pa.array([r[4] for r in rows], pa.int64()), "v": [review_verdict(*r) for r in rows]})
c.register("o", o)
c.execute(f"create or replace table ce as select rid, s1, max(ce) ce from read_parquet(['{S}/push/frce/sc_a.parquet','{S}/push/frce/sc_b.parquet','{S}/push/frce/sc_h.parquet']) group by all")
for s in (1, 2, 3):
    c.execute(f"create or replace table raw{s} as select {ID} eid, business_name bn, business_address ba from read_csv('{R}/test_source{s}.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where country ilike 'france'")
c.execute("create or replace table raw as select * from raw1 union all select * from raw2 union all select * from raw3")
c.execute("""create or replace table pr as select b.*, o.nc, o.nadd, o.nrm, o.ac, o.delta, o.v, ce.ce,
   rq.bn qbn, rq.ba qba, rs.bn sbn, rs.ba sba
   from b2 b join o using (rid, s1) left join ce using (rid, s1) left join raw rq on rq.eid = b.rid left join raw rs on rs.eid = b.s1""")
print("pr", c.execute("select count(*), count(qbn), count(sbn), count(ce) from pr").fetchone())
print(c.execute("select v, count(*), round(avg(p),3) from pr group by v order by 2 desc").fetchall())
