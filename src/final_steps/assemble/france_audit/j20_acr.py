"""Step 4 input (France audit). Experiment/validation script j20_acr.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/output/v11/matching_results_v11_ce_ef.tsv
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {raw}/train/train_source2.tsv
    {raw}/train/train_source3.tsv
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
c.execute(f"""create temp table a as select {ID.format(c='m')} rid from (select unnest(string_split(matched_entity_ids, ',')) m from read_csv('{R}/output/v11/matching_results_v11_ce_ef.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where coalesce(matched_entity_ids,'')<>'') where m<>''""")
print("test S2/S3 records whose raw name is 2-5 uppercase letters (acronym-like), by country/source")
show(c, f"""with r as (select {ID.format(c='entity_id')} eid, cast(substr(entity_id,2,1) as int) src, lower(country) country, business_name from read_csv(['{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')
   where regexp_full_match(business_name, '[A-Z]{{2,5}}'))
  select country, src, count(*) n, round(avg((a.rid is not null)::int),3) asg from r left join a on a.rid=r.eid group by all order by all""")
print("train S2/S3 acronym-like records: truth by source")
show(c, f"""with r as (select {ID.format(c='entity_id')} eid, cast(substr(entity_id,2,1) as int) src, lower(country) country from read_csv(['{raw}/train/train_source2.tsv','{raw}/train/train_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')
   where regexp_full_match(business_name, '[A-Z]{{2,5}}'))
  select country, src, count(*) n, round(avg((t.rid is not null)::int),3) has_truth from r left join '{S}/wf/hunt-odd-one-out-model/truth.parquet' t on t.rid=r.eid group by all order by all""")
