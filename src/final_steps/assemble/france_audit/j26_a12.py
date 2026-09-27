"""Step 4 input (France audit). Experiment/validation script j26_a12.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Outputs:
    {J}/props_cls.parquet
    {J}/test_fp.parquet
"""
from q import *
c=con()
TN="['fils','cie','groupe','services','developpement','associes','france','frs','freres','st']"
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg(ndot::int),4) ndot, round(median(p),3) med_p"
c.execute(f"""create temp table x as select p.*, fp.ncase, fp.ndot, list_filter(string_split(p.nadd,' '), w -> w<>'') aw from '{J}/props_cls.parquet' p join '{J}/test_fp.parquet' fp on fp.eid=p.rid where p.rl in ('A1','A2')""")
show(c, f"""select rl, case when len(aw)>0 and list_has_all({TN}, aw) then 'true_noise' when nadd='compagnie' and nrm='cie' then 'cie_abbrev' else 'other:'||coalesce(nadd,'') end grp, {FPC}, sum((n_at_addr=1)::int) sole
  from x group by all order by rl, n desc""", 40)
show(c, f"""select case when len(aw)>0 and list_has_all({TN}, aw) then 'true_noise' else 'other' end grp, (n_at_addr=1) sole, case when p>=0.5 then 'p>=.5' when p>=0.1 then '.1-.5' else '<.1' end band, {FPC} from x group by all order by all""")
