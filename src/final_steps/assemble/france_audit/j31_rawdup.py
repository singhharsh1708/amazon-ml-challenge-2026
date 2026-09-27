"""Step 4 input (France audit). Experiment/validation script j31_rawdup.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {raw}/test/test_source1.tsv
    {J}/frc.parquet
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
SET="(1,2,3,4,5,7,9,11,13,21)"
c.execute(f"""create temp table rq as select {ID.format(c='entity_id')} eid, lower(business_name) qn from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table rs as select {ID.format(c='entity_id')} eid, lower(business_name) sn from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table x as select f.rid, f.s1, f.asg_this, f.ac, f.delta, f.p, w.w,
   len(regexp_extract_all(rq.qn, w.w)) cq, len(regexp_extract_all(rs.sn, w.w)) cs
   from '{J}/frc.parquet' f join rq on rq.eid=f.rid join rs on rs.eid=f.s1, (select unnest(['france','groupe','développement','developpement']) w) w
   where f.nc='eq'""")
show(c, f"""select w, case when ac like 'same%' then 'same' when delta>0 and delta in {SET} then 'pos_set' when ac like 'numchg%' then 'numchg_other' else ac end ag, asg_this,
   count(*) n from x where cq>cs and cs>=1 group by all order by all""")
