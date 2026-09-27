"""Step 4 input (France audit). Experiment/validation script j03_vpc.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/data/norm/train_s1.parquet
    {J}/train_s1_vocab.parquet
    {J}/train_s1n.parquet
    {J}/vp.parquet
    {J}/vpc.parquet
"""
from q import *
from jrun import run, AK
c=con()
c.execute(f"""create temp table s1k as select {ID.format(c='entity_id')} eid, country, name_core, {AK('address', False)} ak from '{R}/data/norm/train_s1.parquet'""")
c.execute("create temp table s1n as select eid, count(*) over (partition by country, ak) n_at_addr, count(*) over (partition by country, name_core) n_name from s1k")
c.execute(f"""create temp table vocab as select distinct country, unnest(string_split(name_full,' ')) t from '{R}/data/norm/train_s1.parquet'""")
c.execute(f"copy vocab to '{J}/train_s1_vocab.parquet' (format parquet)")
c.execute(f"copy s1n to '{J}/train_s1n.parquet' (format parquet)")
run(c, f"""select v.rid, v.s1, v.p2, v.rk, v.is_true, v.country, v.qf, v.sf, v.qc, v.sc, {AK('v.qa', False)} qk, {AK('v.sa', False)} sk
   from '{J}/vp.parquet' v""", f"{J}/vpc.parquet")
