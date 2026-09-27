"""Step 4 input (France audit). Experiment/validation script j14_fp_rules.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {D}/proposed_france_adds.parquet
    {J}/test_fp.parquet
    {J}/frc.parquet
    {D}/d1.parquet
"""
from q import *
c=con()
c.execute(f"""create temp table pr as select a.rid, a.s1, a.rl, fp.ncase, fp.ndot, fp.src, f.p, f.p_guard, f.nc, f.ac from '{D}/proposed_france_adds.parquet' a
  join '{J}/test_fp.parquet' fp on fp.eid=a.rid left join '{J}/frc.parquet' f on f.rid=a.rid and f.s1=a.s1""")
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),4) ndot, round(avg((src=2)::int),3) s2"
print("Proposed adds: fingerprint per rule")
show(c, f"select rl, {FPC} from pr group by 1 order by 1")
print("Proposed adds: per rule and p band")
show(c, f"select rl, case when p>=0.85 then 'a>=.85' when p>=0.5 then 'b.5-.85' when p>=0.1 then 'c.1-.5' else 'd<.1' end band, {FPC} from pr group by all order by all", 60)
print("Proposed adds: per rule and my name class")
show(c, f"select rl, nc, {FPC} from pr group by all having count(*)>=100 order by all", 60)
print("D1 removals")
show(c, f"select 'D1' rl, {FPC} from '{D}/d1.parquet' d join '{J}/test_fp.parquet' fp on fp.eid=d.rid")
