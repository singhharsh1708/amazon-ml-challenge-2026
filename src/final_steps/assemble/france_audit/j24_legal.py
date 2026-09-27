"""Step 4 input (France audit). Experiment/validation script j24_legal.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
"""
from q import *
c=con()
LEG="['sa','sas','sasu','sarl','eurl','sci','snc','ei','eirl']"
c.execute(f"""create temp table nm as select {ID.format(c='entity_id')} eid, name_full from read_parquet('{R}/data/norm/test_s*.parquet') where country='france'""")
c.execute(f"""create temp table t as select t.*, list_sort(list_distinct(list_filter(string_split(q.name_full,' '), x -> list_contains({LEG}, x)))) ql,
   list_sort(list_distinct(list_filter(string_split(s.name_full,' '), x -> list_contains({LEG}, x)))) sl
   from '{J}/fr_top.parquet' t join nm q on q.eid=t.rid join nm s on s.eid=t.s1 where t.ag in ('same','numchg1') and t.nc='eq'""")
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg((ncase='U')::int),3) up, round(avg(ndot::int),4) ndot, round(avg((src=2)::int),3) s2"
show(c, f"""select ag, asg_this, case when ql=sl then 'same_legal' when len(ql)=0 then 'q_nolegal' when len(sl)=0 then 's_nolegal' else 'legal_changed' end lg, {FPC} from t group by all order by all""")
