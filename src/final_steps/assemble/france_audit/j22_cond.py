"""Step 4 input (France audit). Experiment/validation script j22_cond.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/vpc.parquet
    {J}/train_s1n.parquet
"""
from q import *
c=con()
SET="(1,2,3,4,5,7,9,11,13,21)"
DG=f"case when delta>0 and delta in {SET} then 'pos_set' when delta>0 then 'pos_other' when -delta in {SET} then 'neg_set' else 'neg_other' end"
B="case when p2>=0.85 then 'a>=.85' when p2>=0.5 then 'b.5-.85' when p2>=0.1 then 'c.1-.5' else 'd<.1' end"
print("TRAIN numchg1 eq: ptrue by delta group x p2 band (D1/N1/N2 analogs)")
show(c, f"""select {DG} dg, {B} band, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet' where ac in ('numchg1','numchg1_wtypo') and nc='eq' group by all order by all""", 40)
print("TRAIN A4 analog: address missing, name eq, by S1 name uniqueness x p2 band")
show(c, f"""select (n.n_name=1) uniq, {B} band, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet' v join '{J}/train_s1n.parquet' n on n.eid=v.s1
  where v.ac='missing' and v.nc='eq' group by all order by all""", 40)
print("TRAIN A3b analog: same address, benign name (eq/typo/rm/spaceless), sole S1 at address, by p2 band")
show(c, f"""select (n.n_at_addr=1) sole, nc, {B} band, count(*) n, round(avg(is_true::int),3) ptrue from '{J}/vpc.parquet' v join '{J}/train_s1n.parquet' n on n.eid=v.s1
  where v.ac in ('same','same_ns','same_num_wtypo') and v.nc in ('eq','typo','rm','spaceless') and p2<0.85 group by all order by all""", 40)
