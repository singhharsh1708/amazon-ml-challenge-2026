"""Step 4 input (France audit). Experiment/validation script j19_syll.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/r5_raw.parquet
    {J}/test_fp.parquet
    {J}/vpc.parquet
"""
from q import *
c=con()
SY="(xylo|orbi|vera|zeta|arc|veo|iri|quo|delta|vantage|kelo|cira|kor|flux|umbra|riza|pyra|lyra|zeph|jax|lum|onyx|avi|mira|sol|nyla|ect|ox|ria|vo|ia|ra|ira|elo|bi|ix|ta)"
PAT=f"^{SY}{{2,5}}$"
c.execute(f"""create temp table fr as select r.qn, fp.src from read_parquet('{J}/r5_raw.parquet') r join '{J}/test_fp.parquet' fp on fp.eid=r.rid""")
show(c, f"""select regexp_full_match(lower(qn), '{PAT}') syl, count(*) from fr where regexp_full_match(qn,'[A-Za-z]+') group by 1""")
print("TRAIN valid pairs: single-token names matching syllable pattern")
show(c, f"""select v.country, v.ac in ('same','same_ns') same_addr, v.rk=1 top, count(*) n, round(avg(is_true::int),3) ptrue, round(avg((p2>=0.85)::int),3) p85
  from '{J}/vpc.parquet' v where regexp_full_match(v.qc, '{PAT}') group by all order by all""")
print("TRAIN: records (valid slice) with syllable names: matched to any S1 in truth?")
show(c, f"""with r as (select distinct rid, qc, country from '{J}/vpc.parquet' where regexp_full_match(qc, '{PAT}'))
  select country, count(*) n, round(avg((t.rid is not null)::int),3) has_truth from r left join '{S}/wf/hunt-odd-one-out-model/truth.parquet' t using (rid) group by 1""")
show(c, f"""select qc, sc, qk, sk, is_true, round(p2,3) p from '{J}/vpc.parquet' where regexp_full_match(qc, '{PAT}') and rk=1 using sample 10 rows (reservoir, 3)""")
