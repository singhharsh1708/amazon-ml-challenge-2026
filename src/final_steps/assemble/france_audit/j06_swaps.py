"""Step 4 input (France audit). Experiment/validation script j06_swaps.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/vpc.parquet
"""
from q import *
c=con()
print("TRAIN same-address single-word swaps: top added words")
show(c, f"""select country, nadd, count(*) n, round(avg(is_true::int),3) ptrue, round(avg((p2>=0.85)::int),3) p85 from '{J}/vpc.parquet'
  where ac in ('same','same_ns') and nc='swap1' group by all order by n desc""", 40)
print("TRAIN same-address single-word swaps: top removed words")
show(c, f"""select country, nrm, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet'
  where ac in ('same','same_ns') and nc='swap1' group by all order by n desc""", 25)
print("TRAIN numchg1 swap1: top added words")
show(c, f"""select country, nadd, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet'
  where ac in ('numchg1') and nc='swap1' group by all order by n desc""", 25)
print("TRAIN same-address add: top added words")
show(c, f"""select country, nadd, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet'
  where ac in ('same','same_ns') and nc='add' group by all order by n desc""", 25)
