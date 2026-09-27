"""Step 4 input (France audit). Experiment/validation script j10_fp_train.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/train/train_source2.tsv
    {raw}/train/train_source3.tsv
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
FP = """
  case when business_name = upper(business_name) then 'U' when business_name = lower(business_name) then 'L' else 'M' end ncase,
  (business_address is null or business_address='' ) amiss,
  regexp_matches(coalesce(business_address,''), '^[0-9]') adig,
  regexp_matches(coalesce(business_address,''), '[()]') apar,
  len(string_split(coalesce(business_address,''), ',')) aparts,
  regexp_matches(business_name, '[.]') ndot,
  regexp_matches(business_name, '  ') n2sp,
  regexp_matches(coalesce(business_address,''), '  ') a2sp"""
c.execute(f"""create temp table tr as select {ID.format(c='entity_id')} eid, cast(substr(entity_id,2,1) as int) src, lower(country) country, {FP}
  from read_csv(['{raw}/train/train_source2.tsv','{raw}/train/train_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')""")
c.execute(f"create temp table t2 as select tr.*, (t.rid is not null) is_m from tr left join '{S}/wf/hunt-odd-one-out-model/truth.parquet' t on t.rid=tr.eid")
show(c, """select country, is_m, count(*) n, round(avg((src=2)::int),3) s2, round(avg((ncase='U')::int),3) up, round(avg((ncase='L')::int),3) low,
  round(avg(amiss::int),3) amiss, round(avg(adig::int),3) adig, round(avg(apar::int),3) apar, round(avg(aparts),3) aparts, round(avg(ndot::int),3) ndot, round(avg(n2sp::int),4) n2sp, round(avg(a2sp::int),4) a2sp
  from t2 group by all order by all""")
