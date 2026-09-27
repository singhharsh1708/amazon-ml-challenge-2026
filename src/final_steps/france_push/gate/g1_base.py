"""Step 5 gate. Experiment/validation script g1_base.py for this step.

Step context: Step 5 gate. Validation checks and trims on the France push removal/addition sets; writes removals_trim.parquet and additions_trim.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/gate
    $REPO_DIR/output/v17/matching_results_v17.tsv
    $REPO_DIR/output/v17/candidate_pairs_v17.tsv
    $REPO_DIR/student_resource/dataset/test/test_source{}.tsv
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
import duckdb
import pyarrow as pa
G = f"{WORK_DIR}/push/gate"
SP = f"{WORK_DIR}"
sys.path.insert(0, f"{FINAL_STEPS}/france_push/frce")
from vd import review_name, review_address, review_key, review_verdict
V17 = f"{REPO_DIR}/output/v17/matching_results_v17.tsv"
CAND = f"{REPO_DIR}/output/v17/candidate_pairs_v17.tsv"
RAW = f"{REPO_DIR}/student_resource/dataset/test/test_source{{}}.tsv"
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
RD = "read_csv('{}', delim='\t', header=true, all_varchar=true, quote='', escape='')"
c = duckdb.connect(f"{G}/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
c.execute(f"""create or replace table v17 as select {ID.format(c='source1_entity_id')} s1, {ID.format(c='m')} rid from (
  select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) m from {RD.format(V17)}
  where coalesce(matched_entity_ids, '') <> '')""")
c.execute(f"""create or replace table cand as select {ID.format(c='source1_entity_id')} s1, {ID.format(c='m')} rid from (
  select source1_entity_id, unnest(string_split(candidate_entity_ids, ',')) m from {RD.format(CAND)}
  where coalesce(candidate_entity_ids, '') <> '')""")
c.execute("create or replace table raw as " + " union all ".join(
    f"select {ID.format(c='entity_id')} eid, {s} src, business_name as name, business_address as address, country from {RD.format(RAW.format(s))}"
    for s in (1, 2, 3)))
c.execute(f"create or replace table nm as select {ID.format(c='entity_id')} eid, name_full, address from read_parquet('{SP}/fr_rerun/data/fr_new/test_s*.parquet') where country = 'france'")
print("v17 pairs", c.execute("select count(*), count(distinct rid), count(distinct s1) from v17").fetchone())
print("raw", c.execute("select src, country, count(*) from raw group by all order by 1, 2").fetchall())
c.execute("create or replace table v17fr as select v.* from v17 v join raw s on s.eid = v.s1 where s.country = 'France'")
print("v17 French pairs", c.execute("select count(*), count(distinct rid) from v17fr").fetchone())
t = c.execute(f"""select v.rid, v.s1, q.name_full qn, s.name_full sn, {review_key('q.address')} qa, {review_key('s.address')} sa
  from v17fr v join nm q on q.eid = v.rid join nm s on s.eid = v.s1""").fetchnumpy()
print("v17 French pairs with norm text", len(t["rid"]))
rows = [review_name(a, b) + review_address(x, y) for a, b, x, y in zip(t["qn"], t["sn"], t["qa"], t["sa"])]
out = pa.table({"rid": t["rid"], "s1": t["s1"], "nc": [r[0] for r in rows], "nadd": [r[1] for r in rows], "nrm": [r[2] for r in rows],
    "ac": [r[3] for r in rows], "delta": pa.array([r[4] for r in rows], pa.int64()), "v": [review_verdict(*r) for r in rows]})
c.register("o", out)
c.execute("create or replace table v17v as select * from o")
print("v17 French verdicts", c.execute("select v, count(*) from v17v group by 1 order by 2 desc").fetchall())
