"""Step 4 input (France audit). Experiment/validation script j01_calib.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source1.tsv
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
    {R}/output/v11/matching_results_v11_ce_ef.tsv
    {raw}/train/train_source1.tsv
    {raw}/train/train_source2.tsv
    {raw}/train/train_source3.tsv
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
show(c, f"""
with s as (select country, 1 src from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 o as (select country from read_csv('{raw}/test/test_source2.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
       union all select country from read_csv('{raw}/test/test_source3.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 sc as (select lower(country) country, count(*) n_s1 from s group by 1),
 oc as (select lower(country) country, count(*) n_rec from o group by 1),
 m as (select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) x from read_csv('{R}/output/v11/matching_results_v11_ce_ef.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='') where coalesce(matched_entity_ids,'')<>''),
 s1c as (select entity_id, lower(country) country from read_csv('{raw}/test/test_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 mc as (select s1c.country, count(*) n_m from m join s1c on s1c.entity_id=m.source1_entity_id where x<>'' group by 1)
select sc.country, n_s1, n_rec, n_m, round(n_rec/n_s1,3) rec_per_s1, round(n_m/n_s1,3) m_per_s1, round(n_m/n_rec,4) m_per_rec from sc join oc using(country) join mc using(country) order by 1""")
show(c, f"""
with s as (select lower(country) country from read_csv('{raw}/train/train_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 o as (select lower(country) country from read_csv('{raw}/train/train_source2.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
       union all select lower(country) from read_csv('{raw}/train/train_source3.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 s1c as (select {ID.format(c='entity_id')} eid, lower(country) country from read_csv('{raw}/train/train_source1.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')),
 t as (select s1c.country, count(*) n_t, count(distinct rid) n_trid from '{S}/wf/hunt-odd-one-out-model/truth.parquet' x join s1c on s1c.eid=x.s1 group by 1),
 sc as (select country, count(*) n_s1 from s group by 1), oc as (select country, count(*) n_rec from o group by 1)
select country, n_s1, n_rec, n_t, n_trid, round(n_rec/n_s1,3) rec_per_s1, round(n_t/n_s1,3) t_per_s1, round(n_t/n_rec,4) t_per_rec from sc join oc using(country) join t using(country) order by 1""")
