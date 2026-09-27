"""Step 4 input (France audit). Experiment/validation script j25_removal.py for this step.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {J}/fr_top.parquet
    {J}/fr_typewords.parquet
    {J}/fr_s1n.parquet
    {J}/removal_typeswap.parquet
"""
from q import *
c=con()
TRUE_NOISE="['fils','cie','groupe','services','developpement','associes','france','frs','freres','st','cb','compagnie']"
FPC="count(*) n, round(avg((ncase='L')::int),4) low, round(avg(ndot::int),4) ndot"
c.execute(f"""create temp table tv as select nadd w, count(*) n_numchg from '{J}/fr_top.parquet' where ag='numchg1' and nc='swap1' and not list_contains({TRUE_NOISE}, nadd)
   and regexp_full_match(nadd, '[a-z]{{3,}}') group by 1 having count(*)>=150""")
c.execute(f"copy tv to '{J}/fr_typewords.parquet' (format parquet)")
print("type vocabulary size", c.execute("select count(*), sum(n_numchg) from tv").fetchone())
c.execute(f"""create temp table rm as select t.*, n.n_at_addr from '{J}/fr_top.parquet' t join '{J}/fr_s1n.parquet' n on n.eid=t.s1
   where t.asg_this and t.ag='same' and t.nc='swap1' and t.nadd in (select w from tv)""")
print("REMOVAL candidates: assigned, same address, single-word swap, added word in type vocabulary")
show(c, f"select {FPC}, count(distinct s1) ns1, round(avg((n_at_addr=1)::int),3) sole from rm")
show(c, f"select case when p>=0.99 then 'a>=.99' when p>=0.95 then 'b.95-.99' else 'c<.95' end band, {FPC} from rm group by 1 order by 1")
show(c, f"select (nrm in (select w from tv)) rm_is_type, {FPC} from rm group by 1 order by 1")
show(c, f"select rform, {FPC} from rm group by 1 order by 1")
print("control: assigned same-address noise swaps")
show(c, f"select {FPC} from '{J}/fr_top.parquet' where asg_this and ag='same' and nc='swap1' and list_contains({TRUE_NOISE}, nadd)")
print("assigned same-address swaps with words in neither list")
show(c, f"select {FPC} from '{J}/fr_top.parquet' where asg_this and ag='same' and nc='swap1' and not list_contains({TRUE_NOISE}, nadd) and nadd not in (select w from tv)")
show(c, f"select nadd, nrm, count(*) n from '{J}/fr_top.parquet' where asg_this and ag='same' and nc='swap1' and not list_contains({TRUE_NOISE}, nadd) and nadd not in (select w from tv) group by all order by n desc", 15)
print("compagnie swaps by removed word")
show(c, f"select asg_this, nrm='cie' rm_cie, {FPC} from '{J}/fr_top.parquet' where ag='same' and nc='swap1' and nadd='compagnie' group by all order by all")
print("assigned type swaps in other address classes")
show(c, f"select ag, {FPC} from '{J}/fr_top.parquet' where asg_this and nc='swap1' and nadd in (select w from tv) group by 1 order by 2 desc")
print("assigned multi-word diffs (nc=other/add) containing a type word, same address")
show(c, f"select nc, {FPC} from '{J}/fr_top.parquet' where asg_this and ag='same' and nc in ('other','add') and list_has_any(string_split(nadd,' '), (select list(w) from tv)) group by 1")
c.execute(f"copy (select rid, s1, p, nadd, nrm, n_at_addr from rm) to '{J}/removal_typeswap.parquet' (format parquet)")
