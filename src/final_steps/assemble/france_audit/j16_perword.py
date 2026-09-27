"""Step 4 input (France audit). Experiment/validation script j16_perword.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
"""
from q import *
c=con()
c.execute(f"""create temp table nc1 as select nadd w, count(*) n_numchg from '{J}/fr_top.parquet' where ag='numchg1' and nc='swap1' group by 1""")
c.execute(f"""create temp table sa as select nadd w, count(*) n, count(*) filter (where asg_this) n_asg,
  avg((ncase='L')::int) filter (where asg_this) low_a, avg(ndot::int) filter (where asg_this) dot_a,
  avg((ncase='L')::int) filter (where not asg_this) low_u, avg(ndot::int) filter (where not asg_this) dot_u,
  avg((ncase='L')::int) low_all, avg(ndot::int) dot_all
  from '{J}/fr_top.parquet' where ag='same' and nc='swap1' group by 1""")
show(c, f"""select sa.w, n, n_asg, coalesce(n_numchg,0) n_numchg, round(low_a,4) low_a, round(dot_a,4) dot_a, round(low_u,4) low_u, round(dot_u,4) dot_u,
  round(least(greatest((0.035-low_all)/(0.035-0.0033),0),1),2) ftrue_low, round(least(greatest((0.054-dot_all)/(0.054-0.005),0),1),2) ftrue_dot
  from sa left join nc1 using (w) where n>=150 order by n desc""", 90)
