"""Final gate: writes removals_trim.parquet and additions_trim.parquet.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
    $WORK_DIR/fr_rerun
    {F}/chk/labels.csv
    {F}/chk/cls.parquet
    {G}/additions_trim.parquet
    {G}/removals_trim.parquet
    {G}/{f}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
G = f"{WORK_DIR}/push/gate"
F = f"{WORK_DIR}/fr_rerun"
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
STOP = "('', 'sarl','sas','sasu','eurl','sa','sci','ei','de','du','des','la','le','the','et')"
TOK = "string_split(regexp_replace(lower(strip_accents({x})), '[^a-z0-9 ]', ' ', 'g'), ' ')"
c.execute(f"""create or replace table A_trim as select rid, s1 from ak
  where not ((ce < 0 and kind in ('multi_word', 'one_token'))
    or (kind = 'multi_word' and len(list_filter(list_intersect({TOK.format(x='qn')}, {TOK.format(x='sn')}), x -> x not in {STOP})) = 0))""")
c.execute("""create or replace table R_trim as select rid, s1 from rw
  where lev > 2 and not ((nadd like 'etablissement%' and nrm = 'ets') or (nrm like 'etablissement%' and nadd = 'ets')
    or (nadd = 'saint' and nrm = 'st') or (nadd = 'st' and nrm = 'saint'))""")
for x in ("A_trim", "R_trim"):
    print(x, c.execute(f"select count(*), count(distinct rid), count(distinct s1) from {x}").fetchone())
print("A_trim: pairs in v17", c.execute("select count(*) from A_trim join v17 using (rid, s1)").fetchone(),
      "rids assigned in v17", c.execute("select count(*) from A_trim semi join v17 using (rid)").fetchone(),
      "non-French rid or s1", c.execute("select count(*) from A_trim a join raw q on q.eid = a.rid join raw s on s.eid = a.s1 where q.country <> 'France' or s.country <> 'France' or s.src <> 1 or q.src = 1").fetchone(),
      "in v17 candidate file", c.execute("select count(*) from A_trim join cand using (rid, s1)").fetchone())
print("R_trim: pairs in v17 French", c.execute("select count(*) from R_trim join v17fr using (rid, s1)").fetchone(),
      "non-French", c.execute("select count(*) from R_trim a join raw q on q.eid = a.rid join raw s on s.eid = a.s1 where q.country <> 'France' or s.country <> 'France' or s.src <> 1").fetchone(),
      "subset of R", c.execute("select count(*) from R_trim join R using (rid, s1)").fetchone(),
      "A_trim subset of A", c.execute("select count(*) from A_trim join A using (rid, s1)").fetchone())
c.execute("create or replace table nv2 as select rid, s1 from (select rid, s1 from v17 anti join R_trim using (rid, s1)) union all select rid, s1 from A_trim")
print("v17 - R_trim + A_trim: pairs, distinct rid", c.execute("select count(*), count(distinct rid) from nv2").fetchone())
print("  non-French pairs changed", c.execute("select count(*) from (select rid, s1 from v17 join raw s on s.eid = v17.s1 where s.country <> 'France') x anti join nv2 using (rid, s1)").fetchone())
print("  French pairs", c.execute("select count(*) from nv2 join raw s on s.eid = nv2.s1 where s.country = 'France'").fetchone())
print("  empty French S1", c.execute("select count(*) filter (where eid not in (select s1 from nv2)), count(*) from raw where src = 1 and country = 'France'").fetchone())
c.execute(f"""create or replace table labp as select l.lab, x.rid, x.s1 from read_csv('{F}/chk/labels.csv') l
  join '{F}/chk/cls.parquet' x on x.grp = l.grp and x.rid = cast(substr(l.eid, 2, 1) as bigint) * 10000000000 + cast(substr(l.eid, 4) as bigint)""")
print("hand labels matched", c.execute("select lab, count(*) from labp group by 1").fetchall())
print("hand labels in v17 by lab", c.execute("select lab, count(*) from labp join v17 using (rid, s1) group by 1").fetchall())
for x in ("R", "R_trim", "A", "A_trim"):
    print(f"hand labels hit by {x}", c.execute(f"select lab, count(*) from labp join {x} using (rid, s1) group by 1").fetchall(),
          "same rid hit", c.execute(f"select lab, count(*) from labp join {x} using (rid) group by 1").fetchall())
def fpr(label, q):
    """Print the rate of each risk flag in a query result."""
    r = c.execute(f"""with x as ({q}) select count(*) n,
      round(100 * avg((qf.lc and qf.mw and not qf.dom)::int), 2), round(100 * avg((qf.dot and not qf.dom)::int), 2),
      count(*) filter (where qf.hn <> sf.hn),
      count(*) filter (where qf.hn - sf.hn in (1,2,3,4,5,7,9,11,13,21)),
      count(*) filter (where sf.hn - qf.hn in (1,2,3,4,5,7,9,11,13,21))
      from x join fp qf on qf.eid = x.rid join fp sf on sf.eid = x.s1""").fetchone()
    print(f"{label:34s} n={r[0]:6d} lowercase(multi-word, non-domain)={r[1]:5.2f}% dots(non-domain)={r[2]:5.2f}% hn_diff={r[3]} up_offset={r[4]} down_offset={r[5]}")
fpr("R_trim", "select rid, s1 from R_trim")
fpr("R minus R_trim", "select rid, s1 from R anti join R_trim using (rid, s1)")
fpr("A_trim", "select rid, s1 from A_trim")
fpr("A minus A_trim", "select rid, s1 from A anti join A_trim using (rid, s1)")
c.execute(f"copy (select rid::bigint rid, s1::bigint s1 from A_trim order by rid) to '{G}/additions_trim.parquet' (format parquet)")
c.execute(f"copy (select rid::bigint rid, s1::bigint s1 from R_trim order by rid) to '{G}/removals_trim.parquet' (format parquet)")
for f in ("additions_trim", "removals_trim"):
    print(f, c.execute(f"select count(*), count(distinct rid) from '{G}/{f}.parquet'").fetchone(), c.execute(f"describe select * from '{G}/{f}.parquet'").fetchall())
