"""Step 4 input (France audit). Experiment/validation script j15_typeswap_asg.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
"""
from q import *
c=con()
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),4) ndot"
B="case when p>=0.99 then 'a>=.99' when p>=0.95 then 'b.95-.99' when p>=0.85 then 'c.85-.95' when p>=0.5 then 'd.5-.85' else 'e<.5' end"
print("same-address single-word swap, non-noise word, by assigned x p band")
show(c, f"select asg_this, {B} band, {FPC} from '{J}/fr_top.parquet' where ag='same' and nc='swap1' and wgrp='other' group by all order by all")
print("same-address noise swaps, by assigned x p band")
show(c, f"select asg_this, {B} band, {FPC} from '{J}/fr_top.parquet' where ag='same' and nc='swap1' and wgrp='noise' group by all order by all")
print("same-address type-word swaps assigned, top added words")
show(c, f"select nadd, {FPC}, round(avg(p),3) mp from '{J}/fr_top.parquet' where ag='same' and nc='swap1' and wgrp='other' and asg_this group by all order by n desc", 25)
print("same-address type-word swaps assigned: by address raw form of record")
show(c, f"select rform, {FPC} from '{J}/fr_top.parquet' where ag='same' and nc='swap1' and wgrp='other' and asg_this group by all order by all")
