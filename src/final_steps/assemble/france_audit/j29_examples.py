"""Step 4 input (France audit). Experiment/validation script j29_examples.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {D}/proposed_france_adds.parquet
    {D}/d1.parquet
    {raw}/test/test_source1.tsv
    {J}/frc.parquet
"""
from q import *
import pandas as pd
pd.set_option('display.width', 250)
c=con()
raw=R+'/student_resource/dataset'
ex = {"R5":"S3-88125439 S3-697505358 S3-364024758 S3-490270138 S2-759503517","A3":"S3-851207720 S3-712772378 S2-817702836 S3-319675654 S2-770111939",
"N1":"S3-524859336 S3-789515434 S3-591939774 S3-452086378 S3-129714124","A4":"S3-161147031 S2-713793280 S3-84934311 S2-735420935 S3-999641184",
"A1":"S2-806519199 S3-213144413 S3-942939838 S2-566618602 S3-262611452","A2":"S2-503135025 S2-898428025 S3-869319656 S2-838083081 S3-307729461",
"N2":"S3-959401782 S3-331275073 S3-773100621 S2-381155352 S2-25440417","H1":"S2-187309278 S2-871489807 S3-532746878 S2-441726708 S3-150789611",
"D1":"S2-881933500 S2-755445040 S3-226405406 S3-423244337 S2-135811228"}
rows=[(k,e) for k,v in ex.items() for e in v.split()]
c.execute("create temp table e(rl varchar, eid_s varchar)"); c.executemany("insert into e values (?,?)", rows)
c.execute(f"""create temp table rq as select entity_id, {ID.format(c='entity_id')} eid, business_name qn, business_address qa from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='') where entity_id in (select eid_s from e)""")
c.execute(f"""create temp table pa as select a.rid, a.s1, a.rl from '{D}/proposed_france_adds.parquet' a union all select rid, s1, 'D1' from '{D}/d1.parquet'""")
c.execute(f"""create temp table rs as select {ID.format(c='entity_id')} eid, business_name sn, business_address sa from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france'""")
df=c.execute(f"""select e.rl, e.eid_s, pa.rl prl, round(f.p,3) p, round(f.p_guard,3) pg, f.nc, f.ac, f.nadd, rq.qn, rq.qa, rs.sn, rs.sa from e join rq on rq.entity_id=e.eid_s
  left join pa on pa.rid=rq.eid left join rs on rs.eid=pa.s1 left join '{J}/frc.parquet' f on f.rid=pa.rid and f.s1=pa.s1 order by e.rl""").fetchdf()
for _, r in df.iterrows():
    print(f"{r.rl}|{r.prl}|{r.eid_s}|p={r.p} pg={r.pg}|{r.nc}/{r.ac}/{r.nadd}| Q: {r.qn} @ {r.qa} || S: {r.sn} @ {r.sa}")
