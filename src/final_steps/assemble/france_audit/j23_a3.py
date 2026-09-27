"""Step 4 input (France audit). Experiment/validation script j23_a3.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {raw}/test/test_source1.tsv
    {J}/props_cls.parquet
    {J}/test_fp.parquet
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
c.execute(f"""create temp table rq as select {ID.format(c='entity_id')} eid, business_name qn, business_address qa from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table rs as select {ID.format(c='entity_id')} eid, business_name sn, business_address sa from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table a3 as select p.*, rq.qn, rq.qa, rs.sn, rs.sa, fp.ncase, fp.ndot from '{J}/props_cls.parquet' p join rq on rq.eid=p.rid join rs on rs.eid=p.s1 join '{J}/test_fp.parquet' fp on fp.eid=p.rid where p.rl in ('A3','A3b')""")
show(c, "select rl, nc, (n_at_addr=1) sole, count(*) n, round(avg((ncase='L')::int),3) low, round(avg(ndot::int),3) ndot, round(median(p),3) mp, round(median(p_guard),3) mpg from a3 group by all order by all")
show(c, "select rl, nc, round(p,3) p, round(p_guard,3) pg, n_at_addr na, qn, qa, sn, sa from a3 where nc='eq' using sample 14 rows (reservoir, 11)", 20)
show(c, "select rl, nc, round(p,3) p, round(p_guard,3) pg, qn, sn from a3 where nc in ('typo','disjoint','empty') using sample 12 rows (reservoir, 5)", 20)
