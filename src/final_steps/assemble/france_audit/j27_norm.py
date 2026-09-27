"""Step 4 input (France audit). Experiment/validation script j27_norm.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {raw}/test/test_source1.tsv
    {raw}/test/test_source2.tsv
    {raw}/test/test_source3.tsv
"""
from q import *
c=con()
raw=R+'/student_resource/dataset'
c.execute(f"""create temp table r as select entity_id, cast(substr(entity_id,2,1) as int) src, lower(country) country, business_name n, coalesce(business_address,'') a
  from read_csv(['{raw}/test/test_source1.tsv','{raw}/test/test_source2.tsv','{raw}/test/test_source3.tsv'], delim='\t', header=true, all_varchar=true, quote='', escape='')""")
print("region form of address, France, by source (non-empty addresses)")
show(c, """select src>1 is_copy, case when regexp_matches(lower(a), 'hauts-de-france|nouvelle-aquitaine|pays de la loire') then 'reg'
   when regexp_matches(lower(a), '(^|,\\s*)(nord|gironde|loire-atlantique|pas-de-calais)\\s*(,|$)') then 'dep' else 'none' end rf, count(*) n
   from r where country='france' and a<>'' group by all order by all""")
print("N-degree prefix by country/source (test raw)")
show(c, """select country, src>1 is_copy, count(*) filter (where regexp_matches(a, '\\b[Nn]\\s*[°º]')) n_deg, count(*) filter (where regexp_matches(a, '\\b[Nn]\\s*[°º]\\d')) glued from r group by all order by all""")
print("normalized ndeg tokens (test norm) France S2/S3")
show(c, f"""select count(*) filter (where regexp_matches(address, '\\bndeg\\b')) spaced, count(*) filter (where regexp_matches(address, '\\bndeg[0-9]')) glued from read_parquet('{R}/data/norm/test_s[23].parquet') where country='france'""")
print("glued number+suffix (bis/ter/letter except e) France raw, by source")
show(c, """select src>1 is_copy, count(*) filter (where regexp_matches(a, '\\b\\d+([Bb][Ii][Ss]|[Tt][Ee][Rr]|[A-Da-dF-Zf-z])\\b')) glued from r where country='france' group by 1 order by 1""")
print("S.A.S without final dot and its normalization")
show(c, f"""with x as (select r.entity_id, n.name_full from r join read_parquet('{R}/data/norm/test_s*.parquet') n using (entity_id) where r.country='france' and regexp_matches(r.n, 'S\\.A\\.S(\\b|$)') and not regexp_matches(r.n, 'S\\.A\\.S\\.'))
  select count(*) n, count(*) filter (where regexp_matches(name_full, '\\bsas\\b')) has_sas, count(*) filter (where regexp_matches(name_full, '\\bs a s\\b')) spaced from x""")
