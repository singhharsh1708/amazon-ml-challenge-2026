"""Step 4 input (France audit). Experiment/validation script j28_misc.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
    {J}/vpc.parquet
"""
from q import *
c=con()
print("France: region form of copy address by class (known decoy vs known true)")
show(c, f"""select case when ag='numchg1' and nc='add' and wgrp='pure' then 'decoy: numchg+pure word'
   when ag='numchg1' and nc='swap1' and wgrp='other' then 'decoy: numchg+type swap'
   when ag='same' and nc='eq' and asg_this then 'true: same eq assigned'
   when ag='same' and nc='swap1' and wgrp='noise' and asg_this then 'true: noise swap assigned'
   when ag='same' and nc='swap1' and wgrp='other' and asg_this then 'removal cand: type swap assigned' end grp,
   count(*) n, round(avg((rform='reg')::int),3) reg, round(avg((rform='dep')::int),3) dep, round(avg((rform='none')::int),3) none_
   from '{J}/fr_top.parquet' where qk<>'' group by 1 having grp is not null order by 1""")
print("France: assignment rate of same-address noise swaps / acronyms / made-up by region form")
show(c, f"""select case when nc='swap1' and wgrp='noise' then 'noise swap' when nc='acr' then 'acronym' when nc='disjoint' then 'disjoint' when nc='eq' then 'eq' end k, rform, count(*) n, round(avg(asg_this::int),3) asg
   from '{J}/fr_top.parquet' where ag='same' and qk<>'' group by all having k is not null order by all""")
print("TRAIN: name + one added word (group/holdings/enterprises/associates/partners/company/services), by address class")
show(c, f"""select nadd, case when ac in ('same','same_ns','same_num_wtypo') then 'same' when ac in ('numchg1','numchg1_wtypo','numchg') then 'numchg' else ac end ag, count(*) n, round(avg(is_true::int),4) ptrue
   from '{J}/vpc.parquet' where nc='add' and nadd in ('group','holdings','enterprises','associates','partners','company','services','center') group by all order by nadd, n desc""", 50)
