"""Step 5 (France push). Experiment/validation script p1_prep.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/output/v17/matching_results_v17.tsv
    {F}/out/france_assign_partial.parquet
    {F}/work/new/stacked.parquet
    {F}/work/new/pred2.parquet
    {SP}/rescue/test_rescue.parquet
    {W}/labels60.parquet
    {W}/score_a.parquet
    {W}/frce_hold.parquet
    {W}/score_h.parquet
    {W}/score_b.parquet
    {W}/score_{x}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time
sys.dont_write_bytecode = True
import numpy as np, pyarrow as pa, duckdb
SP = f"{WORK_DIR}"
F = f"{SP}/fr_rerun"; W = f"{SP}/push/frce"
sys.path.insert(0, W)
from vd import verdicts, review_key
V17 = f"{REPO_DIR}/output/v17/matching_results_v17.tsv"
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
t0 = time.time()
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp/spill'")
print(c.execute(f"describe select * from read_csv('{V17}', delim='\\t', header=true, all_varchar=true, quote='', escape='') ").fetchall())
c.execute(f"""create or replace table v17 as select distinct {ID.format(c='m')} rid, {ID.format(c='source1_entity_id')} s1 from (
  select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) m from read_csv('{V17}', delim='\\t', header=true, all_varchar=true, quote='', escape='')
  where coalesce(matched_entity_ids, '') <> '') where m <> ''""")
c.execute("create or replace table v17fr as select v.* from v17 v where v.rid in (select eid from nm where eid >= 20000000000)")
print("v17 all/fr pairs", c.execute("select (select count(*) from v17), count(*), count(distinct rid) from v17fr").fetchall())
print("v17fr vs partial", c.execute(f"""select (select count(*) from v17fr anti join '{F}/out/france_assign_partial.parquet' p using (rid, s1)),
  (select count(*) from '{F}/out/france_assign_partial.parquet' p anti join v17fr using (rid, s1))""").fetchall())
print("v17fr s1 not french", c.execute("select count(*) from v17fr where s1 not in (select eid from nm)").fetchall())
t = c.execute(f"""select v.rid, v.s1, q.name_full qn, s.name_full sn, {review_key('q.address')} qa, {review_key('s.address')} sa,
  q.name_full || ' | ' || q.address q_text, s.name_full || ' | ' || s.address s_text
  from v17fr v join nm q on q.eid=v.rid join nm s on s.eid=v.s1""").to_arrow_table()
v = verdicts(*[t.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")])
c.register("tv", pa.table({"rid": t.column("rid"), "s1": t.column("s1"), "v": v, "q_text": t.column("q_text"), "s_text": t.column("s_text")}))
c.execute("create or replace table v17v as select * from tv")
print("v17fr verdicts", c.execute("select v, count(*) from v17v group by 1 order by 2 desc").fetchall(), f"{time.time()-t0:.0f}s", flush=True)
c.execute(f"""create or replace table band as select b.rid, b.s1, b.ps, p.p, p.p_guard, q.name_full qn, s.name_full sn,
  {review_key('q.address')} qa, {review_key('s.address')} sa, q.name_full || ' | ' || q.address q_text, s.name_full || ' | ' || s.address s_text,
  b.rid in (select rid from v17fr) rid_asg, (b.rid, b.s1) in (select (rid, s1) from v17fr) pair_asg
  from '{F}/work/new/stacked.parquet' b join read_parquet('{F}/work/new/pred2.parquet') p using (rid, s1)
  join nm q on q.eid=b.rid join nm s on s.eid=b.s1""")
print("band", c.execute("select count(*), min(p), max(p), count(*) filter (where p >= 0.85), count(*) filter (where p >= 0.85 and not rid_asg), count(distinct rid) filter (where p >= 0.85 and not rid_asg), count(*) filter (where pair_asg) from band").fetchall(), flush=True)
t = c.execute("select rid, s1, qn, sn, qa, sa from band").to_arrow_table()
v = verdicts(*[t.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")])
c.register("tb", pa.table({"rid": t.column("rid"), "s1": t.column("s1"), "v": v}))
c.execute("create or replace table bandv as select b.rid, b.s1, b.ps, b.p, b.p_guard, b.q_text, b.s_text, b.rid_asg, b.pair_asg, t.v from band b join tb t using (rid, s1)")
print("band verdicts p>=0.85 unassigned", c.execute("select v, count(*) from bandv where p >= 0.85 and not rid_asg group by 1 order by 2 desc").fetchall(), flush=True)
c.execute(f"""create or replace table resc as select r.rid, r.s1, r.family, q.name_full || ' | ' || q.address q_text, s.name_full || ' | ' || s.address s_text,
  r.rid in (select rid from v17fr) rid_asg from '{SP}/rescue/test_rescue.parquet' r join nm q on q.eid=r.rid join nm s on s.eid=r.s1 where r.country='france'""")
print("rescue fr", c.execute("select count(*), count(*) filter (where not rid_asg), count(distinct rid) from resc").fetchall(), flush=True)
c.execute(f"""copy (select rid, s1, q_text, s_text from (
    select rid, s1, q_text, s_text, 0 pr from '{W}/labels60.parquet'
    union all select rid, s1, q_text, s_text, 1 from v17v where v like 'diff%'
    union all select rid, s1, q_text, s_text, 2 from bandv where p >= 0.85 and not rid_asg
    union all select rid, s1, q_text, s_text, 3 from resc where not rid_asg) group by all) to '{W}/score_a.parquet'""")
c.execute(f"copy (select rid, s1, q_text, s_text from '{W}/frce_hold.parquet') to '{W}/score_h.parquet'")
c.execute(f"""copy (select rid, s1, q_text, s_text from (select rid, s1, q_text, s_text from bandv union all select rid, s1, q_text, s_text from resc) group by all
  having (rid, s1) not in (select (rid, s1) from '{W}/score_a.parquet')) to '{W}/score_b.parquet'""")
print("score files", [c.execute(f"select count(*) from '{W}/score_{x}.parquet'").fetchone() for x in "ahb"], f"{time.time()-t0:.0f}s")
