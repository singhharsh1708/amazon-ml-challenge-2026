"""Step 4 input (France audit). Experiment/validation script j17_r5fp.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
    {J}/fr_s1n.parquet
    {J}/fr_s1_vocab.parquet
    {J}/vpc.parquet
    {J}/train_s1n.parquet
    {J}/train_s1_vocab.parquet
    {J}/train_fp.parquet

Outputs:
    {J}/vp.parquet
"""
from q import *
c=con()
DROP="['sa','sas','sasu','sarl','eurl','sci','snc','gmbh','inc','llc','ltd','pvt','private','limited','llp','lp','corp','co','company','plc','pllc','pc','incorporated','corporation','de','du','des','la','le','les','et','and','the','of','d','l']"
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),4) ndot, round(avg((src=2)::int),3) s2"
c.execute(f"""create temp table x as select t.*, n.n_at_addr from '{J}/fr_top.parquet' t join '{J}/fr_s1n.parquet' n on n.eid=t.s1 where t.ag='same' and t.nc='disjoint' and regexp_matches(t.qk,'[0-9]')""")
c.execute(f"""create temp table nmq as select {ID.format(c='entity_id')} eid, name_full from read_parquet('{R}/data/norm/test_s[23].parquet') where country='france'""")
c.execute(f"""create temp table oov as select rid, bool_and(v.t is null) all_oov from (select x.rid, unnest(list_filter(string_split(q.name_full,' '), w -> w<>'' and not list_contains({DROP}, w))) w from x join nmq q on q.eid=x.rid) y
  left join '{J}/fr_s1_vocab.parquet' v on v.t=y.w group by 1""")
print("FRANCE same address, disjoint name: by sole S1, all-OOV, assigned")
show(c, f"""select (n_at_addr=1) sole, coalesce(all_oov,false) all_oov, asg_this, {FPC}, round(median(p),3) med_p from x left join oov using (rid) group by all order by all""")
print("FRANCE sole+OOV by p band")
show(c, f"""select asg_this, case when p>=0.85 then 'a' when p>=0.5 then 'b' when p>=0.1 then 'c' else 'd' end band, {FPC} from x left join oov using (rid) where n_at_addr=1 and all_oov group by all order by all""")
print("TRAIN same address disjoint sole OOV: fingerprint by truth")
c.execute(f"""create temp table v as select v.*, n.n_at_addr from '{J}/vpc.parquet' v join '{J}/train_s1n.parquet' n on n.eid=v.s1
  where v.ac in ('same','same_ns') and v.nc='disjoint' and regexp_matches(v.qk,'[0-9]')""")
c.execute(f"""create temp table oov2 as select rid, s1, bool_and(t.t is null) all_oov from (select v.rid, v.s1, v.country, unnest(list_filter(string_split(vv.qf,' '), x -> x<>'' and not list_contains({DROP}, x))) w from v join '{J}/vp.parquet' vv using (rid, s1)) x
   left join '{J}/train_s1_vocab.parquet' t on t.country=x.country and t.t=x.w group by 1,2""")
show(c, f"""select (n_at_addr=1) sole, coalesce(all_oov,false) all_oov, is_true, {FPC} from v left join oov2 using (rid, s1) join '{J}/train_fp.parquet' f on f.eid=v.rid group by all order by all""")
