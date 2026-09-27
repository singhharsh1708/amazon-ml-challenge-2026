"""Writes the final France audit sets (judge_keep_removals, judge_keep_adds, judge_n2_adds).

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {raw}/test/test_source1.tsv
    {J}/removal_typeswap.parquet
    {J}/fr_top.parquet
    {J}/frc.parquet
    {J}/judge_keep_adds.parquet
    {J}/judge_maybe_adds.parquet
    {J}/judge_keep_removals.parquet

Outputs:
    {J}/props_cls.parquet
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
TN="['fils','cie','groupe','services','developpement','associes','france','frs','freres','st']"
c.execute(f"""create temp table p as select p.*, list_filter(string_split(coalesce(p.nadd,''),' '), w -> w<>'') aw from '{J}/props_cls.parquet' p""")
c.execute(f"""create temp table keep_add as
  select rid, s1, 'A1A2_noise' k from p where rl in ('A1','A2') and ((len(aw)>0 and list_has_all({TN}, aw)) or (nadd='compagnie' and nrm='cie')) and not (n_at_addr>1 and p<0.1)
  union all select rid, s1, 'R5A5' from p where rl in ('R5','A5') and p>=0.1
  union all select rid, s1, 'A3' from p where rl='A3'
  union all select rid, s1, 'N1' from p where rl='N1' and p>=0.5""")
c.execute(f"""create temp table maybe_add as
  select rid, s1, 'N2' k from p where rl='N2' and p>=0.5
  union all select rid, s1, 'A4' from p where rl='A4' and p>=0.85
  union all select rid, s1, 'A3b' from p where rl='A3b'""")
c.execute(f"""create temp table rq as select {ID.format(c='entity_id')} eid, lower(business_name) qn from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table rs as select {ID.format(c='entity_id')} eid, lower(business_name) sn from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table rem as
  select rid, s1, 'R1_typeswap' k from '{J}/removal_typeswap.parquet'
  union all select rid, s1, 'R1_compagnie' from '{J}/fr_top.parquet' where asg_this and ag='same' and nc='swap1' and nadd='compagnie' and nrm<>'cie'
  union all select f.rid, f.s1, 'R2_dupfrance' from '{J}/frc.parquet' f join rq on rq.eid=f.rid join rs on rs.eid=f.s1
     where f.asg_this and f.nc='eq' and f.delta in (1,2,3,4,5,7,9,11,13,21) and len(regexp_extract_all(rq.qn,'france')) > len(regexp_extract_all(rs.sn,'france')) and len(regexp_extract_all(rs.sn,'france'))>=1""")
c.execute(f"copy (select * from keep_add) to '{J}/judge_keep_adds.parquet' (format parquet)")
c.execute(f"copy (select * from maybe_add) to '{J}/judge_maybe_adds.parquet' (format parquet)")
c.execute(f"copy (select * from rem) to '{J}/judge_keep_removals.parquet' (format parquet)")
show(c, "select k, count(*) n, count(distinct rid) nr from keep_add group by 1 union all select k, count(*), count(distinct rid) from maybe_add group by 1 union all select k, count(*), count(distinct rid) from rem group by 1 order by 1")
show(c, "select count(*) overlap_add_rem from keep_add a join rem r using (rid)")
tot=1434993; base=837127
na=c.execute("select count(distinct rid) from keep_add").fetchone()[0]; nr=c.execute("select count(distinct rid) from rem").fetchone()[0]
print("France matches per record: v11", round(base/tot,4), "after keep adds+removals", round((base+na-nr)/tot,4), "matches per S1", round((base+na-nr)/259452,3))
