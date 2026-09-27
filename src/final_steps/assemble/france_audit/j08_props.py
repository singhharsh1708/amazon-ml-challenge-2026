"""Step 4 input (France audit). Experiment/validation script j08_props.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_asg.parquet
    {D}/proposed_france_adds.parquet
    {J}/frc.parquet
    {J}/fr_s1n.parquet
    {J}/props_cls.parquet
"""
from q import *
c=con()
c.execute(f"""create temp table pa as select a.rid, a.s1, a.rl, f.nc, f.nadd, f.nrm, f.ac, f.delta, f.p, f.p_guard, f.asg_this, n.n_at_addr, n.n_name,
  (select count(*) from '{J}/fr_asg.parquet' x where x.rid=a.rid) rid_asg
  from '{D}/proposed_france_adds.parquet' a left join '{J}/frc.parquet' f on f.rid=a.rid and f.s1=a.s1 left join '{J}/fr_s1n.parquet' n on n.eid=a.s1""")
c.execute(f"copy pa to '{J}/props_cls.parquet' (format parquet)")
show(c, """select rl, count(*) n, count(nc) in_frc, sum(rid_asg) already_asg, count(distinct s1) ns1, round(median(p),3) med_p, round(avg((p>=0.85)::int),3) p85, round(avg((p_guard>=0.85)::int),3) pg85, round(avg((p>=0.5)::int),3) p50,
   round(avg((n_at_addr=1)::int),3) sole_addr from pa group by 1 order by 1""")
show(c, """select rl, ac, nc, count(*) n from pa group by all qualify row_number() over (partition by rl order by count(*) desc)<=6 order by rl, n desc""", 80)
