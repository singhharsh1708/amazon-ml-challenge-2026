"""Step 4 input (France audit). Experiment/validation script j18_r5raw.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {raw}/test/test_source1.tsv
    {D}/oov_incand.parquet
    {J}/test_fp.parquet
    {J}/r5_raw.parquet
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
c.execute(f"""create temp table rq as select {ID.format(c='entity_id')} eid, business_name qn, business_address qa from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table rs as select {ID.format(c='entity_id')} eid, business_name sn, business_address sa from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
c.execute(f"""create temp table r5 as select o.rid, o.s1, o.pa, o.g_p, o.is_acr, rq.qn, rq.qa, rs.sn, rs.sa, fp.src, fp.ncase from '{D}/oov_incand.parquet' o join rq on rq.eid=o.rid join rs on rs.eid=o.s1 join '{J}/test_fp.parquet' fp on fp.eid=o.rid""")
c.execute(f"copy r5 to '{J}/r5_raw.parquet' (format parquet)")
show(c, """select case when regexp_matches(qn, '[.]|www|\\.com') then 'web' when is_acr then 'acr' when regexp_matches(qn, '^[A-Za-z]+$') then 'oneword' else 'multi' end kind, count(*) n, round(avg((src=2)::int),3) s2, round(avg((ncase='L')::int),3) low, round(median(pa),3) mp from r5 group by 1 order by 1""")
show(c, "select round(pa,3) p, round(g_p,3) pg, qn, qa, sn, sa from r5 using sample 25 rows (reservoir, 7)", 25)
