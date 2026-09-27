"""Step 4 input (France audit). Experiment/validation script j05_xtab.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/vpc.parquet
    {J}/frc.parquet
"""
from q import *
c=con()
AG="case when ac in ('same','same_ns') then 'same' when ac in ('numchg1','numchg1_wtypo') then 'numchg1' else ac end"
print("TRAIN valid (US/India), all candidate pairs: name class x address class")
show(c, f"""select {AG} ag, nc, count(*) n, round(avg(is_true::int),3) ptrue, round(avg((p2>=0.85)::int),3) p85,
  count(*) filter (where p2<0.5) n_lo, round(avg(is_true::int) filter (where p2<0.5),3) ptrue_lo,
  count(*) filter (where p2>=0.5 and p2<0.85) n_mid, round(avg(is_true::int) filter (where p2>=0.5 and p2<0.85),3) ptrue_mid
  from '{J}/vpc.parquet' group by all having count(*)>=300 order by ag, n desc""", 200)
print("FRANCE test, all candidate pairs: name class x address class")
show(c, f"""select {AG} ag, nc, count(*) n, round(avg(asg_this::int),3) asg_rate, round(avg((p>=0.85)::int),3) p85, round(avg((p_guard>=0.85)::int),3) pg85
  from '{J}/frc.parquet' group by all having count(*)>=300 order by ag, n desc""", 200)
