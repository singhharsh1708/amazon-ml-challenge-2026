"""Step 4 input (France audit). Experiment/validation script j02_build_valid.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {S}/rescue/valid_v11_scores.parquet
    {S}/rescue/valid_v11_assigned.parquet
    {J}/vp.parquet
"""
from q import *
c=con()
c.execute(f"""create temp table vs as select rid, s1, p2, row_number() over (partition by rid order by p2 desc, s1) rk from '{S}/rescue/valid_v11_scores.parquet'""")
c.execute(f"""create temp table ids as select distinct rid eid from vs union select distinct s1 from vs""")
c.execute(f"""create temp table nm as select {ID.format(c='entity_id')} eid, country, name_full, name_core, address from read_parquet('{R}/data/norm/train_s*.parquet') where {ID.format(c='entity_id')} in (select eid from ids)""")
print(c.execute("select count(*) from nm").fetchone())
c.execute(f"""copy (select v.rid, v.s1, v.p2, v.rk, (t.rid is not null) is_true, a.s1 asg, q.country,
  q.name_full qf, q.name_core qc, q.address qa, s.name_full sf, s.name_core sc, s.address sa
  from vs v join nm q on q.eid=v.rid join nm s on s.eid=v.s1
  left join '{S}/wf/hunt-odd-one-out-model/truth.parquet' t on t.rid=v.rid and t.s1=v.s1
  left join '{S}/rescue/valid_v11_assigned.parquet' a on a.rid=v.rid) to '{J}/vp.parquet' (format parquet)""")
show(c, f"""select country, count(*) n, count(distinct rid) nrid, sum(is_true::int) ntrue, count(*) filter (where rk=1) top, avg(is_true::int) filter (where rk=1) ptrue_top,
  count(distinct rid) filter (where asg is not null) n_asg from '{J}/vp.parquet' group by 1""")
show(c, f"""with tr as (select t.rid, t.s1 from '{S}/wf/hunt-odd-one-out-model/truth.parquet' t where t.rid in (select distinct rid from '{S}/rescue/valid_v11_scores.parquet'))
 select count(*) n_truth_valid, count(*) filter (where (rid,s1) in (select (rid,s1) from '{S}/rescue/valid_v11_assigned.parquet')) tp from tr""")
