"""Step 4 input (France audit). Experiment/validation script j09_r5train.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/vpc.parquet
    {J}/train_s1n.parquet
    {J}/train_s1_vocab.parquet

Outputs:
    {J}/vp.parquet
"""
from q import *
c=con()
DROP="['sa','sas','sasu','sarl','eurl','sci','snc','gmbh','inc','llc','ltd','pvt','private','limited','llp','lp','corp','co','company','plc','pllc','pc','incorporated','corporation','de','du','des','la','le','les','et','and','the','of','d','l']"
BAND="case when {p}>=0.85 then 'a>=.85' when {p}>=0.5 then 'b.5-.85' when {p}>=0.1 then 'c.1-.5' else 'd<.1' end"
c.execute(f"""create temp table v as select v.*, n.n_at_addr, n.n_name from '{J}/vpc.parquet' v join '{J}/train_s1n.parquet' n on n.eid=v.s1
  where v.ac in ('same','same_ns') and v.nc in ('disjoint','acr') and regexp_matches(v.qk,'[0-9]')""")
c.execute(f"""create temp table oov as select rid, s1, bool_and(t.t is null) all_oov from (select v.rid, v.s1, v.country, unnest(list_filter(string_split(vv.qf,' '), x -> x<>'' and not list_contains({DROP}, x))) w from v join '{J}/vp.parquet' vv using (rid, s1)) x
   left join '{J}/train_s1_vocab.parquet' t on t.country=x.country and t.t=x.w group by 1,2""")
print("TRAIN: same address (with a number), record name disjoint/acronym; by sole-S1-at-address, all-OOV, p2 band")
show(c, f"""select v.nc, (n_at_addr=1) sole, coalesce(o.all_oov,false) all_oov, {BAND.format(p='p2')} band, count(*) n, round(avg(is_true::int),3) ptrue
  from v left join oov o using (rid, s1) group by all order by all""", 80)
print("TRAIN: same, per record: does the record have a different true S1?")
show(c, f"""select v.nc, (n_at_addr=1) sole, coalesce(o.all_oov,false) all_oov, count(*) n, round(avg(is_true::int),3) ptrue,
   round(avg((not is_true and exists(select 1 from '{S}/wf/hunt-odd-one-out-model/truth.parquet' t where t.rid=v.rid))::int),3) true_elsewhere
  from v left join oov o using (rid, s1) group by all order by all""", 80)
