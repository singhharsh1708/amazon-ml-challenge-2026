"""Step 5 gate. Experiment/validation script g5_a.py for this step.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
G = f"{WORK_DIR}/push/gate"
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
LEG = "(sarl|sasu|sas|eurl|sci|snc|sa|ei|scop|scm)"
c.execute(f"""create or replace table fs1 as select eid, name, address,
  regexp_replace(lower(strip_accents(name)), '[^a-z0-9]', '', 'g') cat,
  regexp_replace(regexp_replace(lower(strip_accents(name)), '\\b{LEG}\\b', '', 'g'), '[^a-z0-9]', '', 'g') catnl,
  try_cast(nullif(regexp_extract(address, '\\b([0-9]+)\\b', 1), '') as int) hn
  from raw where src = 1 and country = 'France'""")
c.execute(f"""create or replace table aa as select A.rid, A.s1, q.name qn, q.address qa, s.name sn, s.address sa, fce.ce,
  regexp_replace(regexp_replace(lower(strip_accents(q.name)), '\\.(com|fr|net|org)$', ''), '[^a-z0-9]', '', 'g') stem,
  f.cat, f.catnl, f.hn,
  try_cast(nullif(regexp_extract(q.address, '\\b([0-9]+)\\b', 1), '') as int) qhn
  from A join raw q on q.eid = A.rid join raw s on s.eid = A.s1 join fs1 f on f.eid = A.s1 left join fce using (rid, s1)""")
print("stem == S1 cat or cat-without-legal", c.execute("""select count(*), count(*) filter (where stem = cat or stem = catnl),
  count(*) filter (where stem <> cat and stem <> catnl and (contains(cat, stem) or contains(stem, catnl))) from aa""").fetchone())
c.execute("""create or replace table comp as select a.rid, a.s1, list(distinct o.eid) as oth from aa a join fs1 o
  on o.eid <> a.s1 and (o.cat = a.stem or o.catnl = a.stem) where length(a.stem) >= 4 group by all""")
print("additions with ANOTHER French S1 whose name concatenation equals the Q stem", c.execute("select count(*) from comp").fetchone())
print("  of which that other S1 has same first house number", c.execute("""select count(distinct c.rid) from comp c join aa a using (rid, s1), unnest(c.oth) u(o) join fs1 f on f.eid = u.o where f.hn = a.qhn""").fetchone())
for r in c.execute("""select a.qn, a.qa, a.sn, a.sa, a.ce, f.name, f.address from comp c join aa a using (rid, s1), unnest(c.oth) u(o) join fs1 f on f.eid = u.o order by hash(a.rid, 3) limit 12""").fetchall():
    print("  Q:", r[0], "|", r[1], "\n   chosen S1:", r[2], "|", r[3], "frCE", r[4], "\n   other S1:", r[5], "|", r[6])
print("stem not matching chosen S1 (domain/hashtag only), sample:")
for r in c.execute("""select a.qn, a.qa, a.sn, a.sa, a.ce from aa a join fp on fp.eid = a.rid where fp.dom and a.stem <> a.cat and a.stem <> a.catnl order by hash(a.rid, 5) limit 15""").fetchall():
    print("  ", r)
print("count domain not matching", c.execute("select count(*) from aa a join fp on fp.eid=a.rid where fp.dom and a.stem <> a.cat and a.stem <> a.catnl").fetchone())
