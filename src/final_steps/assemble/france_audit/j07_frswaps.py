"""Step 4 input (France audit). Experiment/validation script j07_frswaps.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/frc.parquet
"""
from q import *
c=con()
print("FRANCE same-address single-word swaps: top added words")
show(c, f"""select nadd, count(*) n, count(distinct rid) nr, round(avg(asg_this::int),3) asg, round(avg((p>=0.85)::int),3) p85, round(avg((p_guard>=0.85)::int),3) pg85 from '{J}/frc.parquet'
  where ac in ('same','same_ns') and nc='swap1' group by all order by n desc""", 45)
print("FRANCE numchg1 single-word swaps: top added words")
show(c, f"""select nadd, count(*) n, round(avg(asg_this::int),3) asg from '{J}/frc.parquet'
  where ac='numchg1' and nc='swap1' group by all order by n desc""", 30)
print("FRANCE same-address add: top added words")
show(c, f"""select nadd, count(*) n, round(avg(asg_this::int),3) asg, round(avg((p>=0.85)::int),3) p85 from '{J}/frc.parquet'
  where ac in ('same','same_ns') and nc='add' group by all order by n desc""", 25)
print("FRANCE numchg1 add: top added words")
show(c, f"""select nadd, count(*) n, round(avg(asg_this::int),3) asg from '{J}/frc.parquet'
  where ac='numchg1' and nc='add' group by all order by n desc""", 25)
