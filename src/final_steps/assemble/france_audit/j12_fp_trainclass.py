"""Step 4 input (France audit). Experiment/validation script j12_fp_trainclass.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/train/train_source2.tsv
    {raw}/train/train_source3.tsv
    {J}/train_fp.parquet
    {J}/vpc.parquet
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
FP = """
  case when business_name = upper(business_name) then 'U' when business_name = lower(business_name) then 'L' else 'M' end ncase,
  regexp_matches(coalesce(business_address,''), '^[0-9]') adig,
  regexp_matches(business_name, '[.]') ndot"""
c.execute(f"""copy (select {ID.format(c='entity_id')} eid, cast(substr(entity_id,2,1) as int) src, {FP}
  from read_csv(['{raw}/train/train_source2.tsv','{raw}/train/train_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')) to '{J}/train_fp.parquet' (format parquet)""")
AG="case when ac in ('same','same_ns','same_num_wtypo') then 'same' when ac in ('numchg1','numchg1_wtypo') then 'numchg1' else ac end"
show(c, f"""select {AG} ag, nc, is_true, count(*) n, round(avg((ncase='L')::int),3) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),3) ndot, round(avg((src=2)::int),3) s2, round(avg(adig::int),3) adig
  from '{J}/vpc.parquet' v join '{J}/train_fp.parquet' f on f.eid=v.rid where v.rk=1 group by all having count(*)>=1000 order by ag, nc, is_true""", 100)
