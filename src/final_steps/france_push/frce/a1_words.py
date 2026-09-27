"""Step 5 (France push). Experiment/validation script a1_words.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/student_resource/dataset/test
    {R}/test_source1.tsv
    {R}/test_source2.tsv
    {R}/test_source3.tsv
    {F}/work/new/pred2.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys, time
import duckdb
SP = f"{WORK_DIR}"
F = f"{SP}/fr_rerun"
W = f"{SP}/push/frce"
R = f"{REPO_DIR}/student_resource/dataset/test"
sys.path.insert(0, f"{REPO_DIR}/src"); sys.path.insert(0, f"{FINAL_STEPS}/france_rerun/src")
from france_rules import review_key, REVIEW_STOP
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
t0 = time.time()
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp/spill'")
stop = "[" + ",".join(f"'{w}'" for w in sorted(REVIEW_STOP)) + "]"
c.execute(f"""create or replace table nm as select {ID} eid, name_full, coalesce(address,'') address, {review_key('address')} ak,
  list_distinct(list_filter(string_split(name_full, ' '), x -> len(x) > 1 and not list_contains({stop}, x))) tk,
  list_filter(string_split(name_full, ' '), x -> list_contains(['sa','sas','sasu','sarl','eurl','sci','snc','ei','eirl'], x)) lg
  from read_parquet('{F}/data/fr_new/test_s*.parquet')""")
c.execute(f"""create or replace table rw as select {ID} eid, business_name bn,
  (business_name = lower(business_name) and regexp_matches(business_name, '[a-z]'))::int lc, contains(business_name, '.')::int dot
  from read_csv(['{R}/test_source1.tsv','{R}/test_source2.tsv','{R}/test_source3.tsv'], delim='\\t', header=true, all_varchar=true, quote='', escape='')
  where lower(country)='france'""")
print(c.execute("select count(*), avg(lc), avg(dot) from rw where eid >= 20000000000").fetchall(), f"{time.time()-t0:.0f}s", flush=True)
c.execute(f"""create or replace table pr as
  select p.rid, p.s1, p.p, p.p_guard, p.p1,
    list_filter(q.tk, x -> not list_contains(s.tk, x)) ad, list_filter(s.tk, x -> not list_contains(q.tk, x)) rm,
    q.ak = s.ak sa, regexp_replace(q.ak, '[0-9]+', '#', 'g') = regexp_replace(s.ak, '[0-9]+', '#', 'g') sa_nonum,
    q.lg ql, s.lg sl, w.lc, w.dot
  from read_parquet('{F}/work/new/pred2.parquet') p join nm q on q.eid = p.rid join nm s on s.eid = p.s1 join rw w on w.eid = p.rid""")
print("pairs", c.execute("select count(*) from pr").fetchone(), f"{time.time()-t0:.0f}s", flush=True)
print("== added one word, same address: per word")
c.execute("""create or replace table addw as select ad[1] w, count(distinct rid) n, avg(lc) lc, avg(dot) dot, avg(p) p
  from (select distinct rid, s1, ad, lc, dot, p from pr where sa and len(rm)=0 and len(ad)=1) group by 1""")
for r in c.execute("select * from addw where n >= 40 order by n desc limit 80").fetchall():
    print("  ", r[0], r[1], round(r[2]*100, 2), round(r[3]*100, 2), round(r[4], 3))
print("== swap one word (same address modulo number): per added word")
c.execute("""create or replace table swp as select ad[1] w, count(distinct rid) n, avg(lc) lc, avg(dot) dot, avg(p) p
  from (select distinct rid, s1, ad, lc, dot, p from pr where sa_nonum and len(rm)=1 and len(ad)=1) group by 1""")
for r in c.execute("select * from swp where n >= 150 and regexp_full_match(w, '[a-z]{3,}') order by n desc limit 80").fetchall():
    print("  ", r[0], r[1], round(r[2]*100, 2), round(r[3]*100, 2), round(r[4], 3))
print("== legal-form change, else equal, same address")
print(c.execute("""select count(distinct rid), avg(lc), avg(dot), avg(p) from pr where sa and len(ad)=0 and len(rm)=0
  and len(ql)=1 and len(sl)=1 and ql[1] <> sl[1]""").fetchall())
print("== legal-form dropped (q none, s one), else equal, same address")
print(c.execute("""select count(distinct rid), avg(lc), avg(dot), avg(p) from pr where sa and len(ad)=0 and len(rm)=0
  and len(ql)=0 and len(sl)=1""").fetchall())
print("== exact same tokens and address (baseline)")
print(c.execute("""select count(distinct rid), avg(lc), avg(dot), avg(p) from pr where sa and len(ad)=0 and len(rm)=0 and ql = sl""").fetchall())
print("== dropped one word, same address")
print(c.execute("""select count(distinct rid), avg(lc), avg(dot), avg(p) from pr where sa and len(ad)=0 and len(rm)=1""").fetchall())
print(f"{time.time()-t0:.0f}s")
