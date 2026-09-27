"""Step 4 input (France audit). Experiment/validation script j11_fp_test.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {J}/test_fp.parquet
    {raw}/test/test_source1.tsv
    {J}/fr_s1_fp.parquet
    {R}/output/v11/matching_results_v11_ce_ef.tsv
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
FP = """
  case when business_name = upper(business_name) then 'U' when business_name = lower(business_name) then 'L' else 'M' end ncase,
  (business_address is null or business_address='' ) amiss,
  regexp_matches(coalesce(business_address,''), '^[0-9]') adig,
  len(string_split(coalesce(business_address,''), ',')) aparts,
  regexp_matches(business_name, '[.]') ndot,
  regexp_matches(business_name, '  ') n2sp,
  case when regexp_matches(lower(coalesce(business_address,'')), 'hauts-de-france|nouvelle-aquitaine|pays de la loire') then 'reg'
       when regexp_matches(lower(coalesce(business_address,'')), '(^|, )(nord|gironde|loire-atlantique|pas-de-calais)(,|$)') then 'dep' else 'none' end rform"""
c.execute(f"""copy (select {ID.format(c='entity_id')} eid, cast(substr(entity_id,2,1) as int) src, lower(country) country, {FP}
  from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')) to '{J}/test_fp.parquet' (format parquet)""")
c.execute(f"""copy (select {ID.format(c='entity_id')} eid, {FP}
  from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where lower(country)='france') to '{J}/fr_s1_fp.parquet' (format parquet)""")
c.execute(f"""create temp table a as select {ID.format(c='m')} rid from (select unnest(string_split(matched_entity_ids, ',')) m from read_csv('{R}/output/v11/matching_results_v11_ce_ef.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where coalesce(matched_entity_ids,'')<>'') where m<>''""")
show(c, f"""select country, (a.rid is not null) asg, count(*) n, round(avg((src=2)::int),3) s2, round(avg((ncase='U')::int),3) up, round(avg((ncase='L')::int),3) low,
  round(avg(amiss::int),3) amiss, round(avg(adig::int),3) adig, round(avg(aparts),3) aparts, round(avg(ndot::int),3) ndot, round(avg(n2sp::int),4) n2sp,
  round(avg((rform='reg')::int),3) reg, round(avg((rform='dep')::int),3) dep
  from '{J}/test_fp.parquet' f left join a on a.rid=f.eid group by all order by all""")
show(c, f"""select round(avg((ncase='U')::int),3) up, round(avg((ncase='L')::int),3) low, round(avg(adig::int),3) adig, round(avg(ndot::int),3) ndot, round(avg((rform='reg')::int),3) reg, round(avg((rform='dep')::int),3) dep, round(avg(amiss::int),3) amiss from '{J}/fr_s1_fp.parquet'""")
