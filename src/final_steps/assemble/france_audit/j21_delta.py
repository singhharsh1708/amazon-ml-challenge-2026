"""Step 4 input (France audit). Experiment/validation script j21_delta.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/vpc.parquet
    {J}/frc.parquet
"""
from q import *
c=con()
SET="(1,2,3,4,5,7,9,11,13,21)"
DG=f"case when delta>0 and delta in {SET} then 'pos_set' when delta>0 then 'pos_other' when -delta in {SET} then 'neg_set' else 'neg_other' end"
print("TRAIN numchg1 (same street words, one number each), name eq/typo/spaceless: ptrue by delta group")
show(c, f"""select country, nc, {DG} dg, count(*) n, round(avg(is_true::int),3) ptrue, round(avg(is_true::int) filter (where p2<0.5),3) ptrue_lo, round(avg((p2>=0.85)::int),3) p85
  from '{J}/vpc.parquet' where ac in ('numchg1','numchg1_wtypo') and nc in ('eq','typo','spaceless') group by all order by all""", 60)
print("TRAIN delta histogram (eq, top pair), false vs true")
show(c, f"""select delta, count(*) filter (where not is_true) n_false, count(*) filter (where is_true) n_true from '{J}/vpc.parquet' where ac='numchg1' and nc='eq' and abs(delta)<=25 group by 1 order by 1""", 60)
print("FRANCE delta histogram (eq, all pairs), assigned vs not")
show(c, f"""select delta, count(*) filter (where not asg_this) n_unasg, count(*) filter (where asg_this) n_asg from '{J}/frc.parquet' where ac='numchg1' and nc='eq' and abs(delta)<=25 group by 1 order by 1""", 60)
