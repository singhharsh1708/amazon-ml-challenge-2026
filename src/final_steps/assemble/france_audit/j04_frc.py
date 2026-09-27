"""Step 4 input (France audit). Experiment/validation script j04_frc.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/output/v11/matching_results_v11_ce_ef.tsv
    {J}/fr_asg.parquet
    {J}/fr_s1n.parquet
    {J}/fr_s1_vocab.parquet
    {S}/v11/pred_v11.parquet
    {J}/frc.parquet
"""
from q import *
from jrun import run, AK
c=con()
c.execute(f"""create temp table nm as select {ID.format(c='entity_id')} eid, country, name_full, name_core, address from read_parquet('{R}/data/norm/test_s*.parquet') where country='france'""")
c.execute(f"""create temp table asg as select {ID.format(c='m')} rid, {ID.format(c='source1_entity_id')} s1 from (
  select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{R}/output/v11/matching_results_v11_ce_ef.tsv', delim='\t', header=true, all_varchar=true, quote='', escape='')
  where coalesce(matched_entity_ids,'')<>'') where m<>''""")
c.execute(f"copy (select a.* from asg a join nm on nm.eid=a.rid) to '{J}/fr_asg.parquet' (format parquet)")
c.execute(f"""create temp table s1k as select eid, name_core, {AK('address', True)} ak from nm where eid//10000000000=1""")
c.execute(f"copy (select eid, ak, count(*) over (partition by ak) n_at_addr, count(*) over (partition by name_core) n_name from s1k) to '{J}/fr_s1n.parquet' (format parquet)")
c.execute(f"copy (select distinct unnest(string_split(name_full,' ')) t from nm where eid//10000000000=1) to '{J}/fr_s1_vocab.parquet' (format parquet)")
run(c, f"""select p.rid, p.s1, p.p, p.p_guard, (a.s1 is not null) asg_this, q.qc, q.sc, q.qf, q.sf, q.qk, q.sk from '{S}/v11/pred_v11.parquet' p
  join (select q.eid rid, s.eid s1, q.name_core qc, s.name_core sc, q.name_full qf, s.name_full sf, {AK('q.address', True)} qk, {AK('s.address', True)} sk
        from nm q, nm s where false) q on true
  left join fr_asg a on false""" if False else f"""
  with pp as (select p.* from '{S}/v11/pred_v11.parquet' p where p.rid in (select eid from nm))
  select pp.rid, pp.s1, pp.p, pp.p_guard, (a.s1 is not null) asg_this, q.name_core qc, s.name_core sc, q.name_full qf, s.name_full sf,
    {AK('q.address', True)} qk, {AK('s.address', True)} sk
  from pp join nm q on q.eid=pp.rid join nm s on s.eid=pp.s1 left join '{J}/fr_asg.parquet' a on a.rid=pp.rid and a.s1=pp.s1""", f"{J}/frc.parquet")
show(c, f"select count(*), count(distinct rid), sum(asg_this::int) from '{J}/frc.parquet'")
show(c, f"select count(*), count(distinct rid) from '{J}/fr_asg.parquet'")
