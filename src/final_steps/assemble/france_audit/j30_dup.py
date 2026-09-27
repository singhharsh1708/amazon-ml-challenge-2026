"""Step 4 input (France audit). Experiment/validation script j30_dup.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/frc.parquet
    {J}/test_fp.parquet
"""
from q import *
from collections import Counter
c=con()
SET={1,2,3,4,5,7,9,11,13,21}
DW={'france','groupe','developpement','holding','participations','distribution','international'}
c.execute(f"""create temp table nm as select {ID.format(c='entity_id')} eid, name_full from read_parquet('{R}/data/norm/test_s*.parquet') where country='france'""")
d=c.execute(f"""select f.rid, f.s1, f.p, f.asg_this, f.ac, f.delta, q.name_full qf, s.name_full sf, fp.ncase from '{J}/frc.parquet' f join nm q on q.eid=f.rid join nm s on s.eid=f.s1
   join '{J}/test_fp.parquet' fp on fp.eid=f.rid where f.nc='eq' and f.ac in ('numchg1','numchg1_wtypo','same','same_ns','same_num_wtypo')""").fetchdf()
def extra(q, s):
    """Return the words of q that are not in s."""
    cq, cs = Counter(q.split()), Counter(s.split())
    ex = cq - cs
    return ' '.join(sorted(ex.elements()))
d['extra'] = [extra(a, b) for a, b in zip(d.qf, d.sf)]
d['dupdw'] = d.extra.isin(DW)
d['samea'] = d.ac.str.startswith('same')
d['dg'] = ['same' if s else ('pos_set' if (x is not None and x > 0 and x in SET) else 'other') for s, x in zip(d.samea, d.delta)]
g = d[d.dupdw].groupby(['dg','asg_this']).agg(n=('rid','size'), low=('ncase', lambda x: round((x=='L').mean(),4))).reset_index()
print("pairs whose name equals S1 plus a DUPLICATED decoy/dual word (set-equal), by address group and assigned"); print(g.to_string())
print(d[d.dupdw].groupby(['dg','extra']).size().unstack(fill_value=0).to_string())
