"""Step 4 input (France audit). Experiment/validation script j13_fp_fr.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/frc.parquet
    {J}/test_fp.parquet
    {D}/proposed_france_adds.parquet
    {J}/fr_top.parquet
"""
from q import *
c=con()
NOISE="['fils','cie','associes','services','service','compagnie','frs','st','societe','centre','groupe','france','developpement']"
PURE="['participations','holding','distribution','international']"
c.execute(f"""create temp table top as select f.* from '{J}/frc.parquet' f qualify row_number() over (partition by rid order by asg_this desc, p desc, s1)=1""")
c.execute(f"""create temp table t as select top.*, fp.ncase, fp.ndot, fp.src, fp.adig, fp.rform, pr.rl,
  case when ac in ('same','same_ns','same_num_wtypo') then 'same' when ac in ('numchg1','numchg1_wtypo') then 'numchg1' else ac end ag,
  case when nc in ('swap1','add') and list_contains({NOISE}, nadd) then 'noise' when nc in ('swap1','add') and list_contains({PURE}, nadd) then 'pure' when nc in ('swap1','add') then 'other' else '' end wgrp
  from top join '{J}/test_fp.parquet' fp on fp.eid=top.rid left join '{D}/proposed_france_adds.parquet' pr on pr.rid=top.rid and pr.s1=top.s1""")
c.execute(f"copy t to '{J}/fr_top.parquet' (format parquet)")
show(c, """select ag, nc, wgrp, asg_this asg, count(*) n, round(avg((ncase='L')::int),4) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),3) ndot, round(avg((src=2)::int),3) s2, round(avg(adig::int),3) adig
  from t where ag in ('same','numchg1','missing','numdrop') group by all having count(*)>=800 order by ag, nc, wgrp, asg""", 100)
