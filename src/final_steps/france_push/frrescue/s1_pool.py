"""Step 5 (France push). Experiment/validation script s1_pool.py for this step.

Step context: Step 5 (France push). France rescue additions for records left without a match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frrescue
    $REPO_DIR/output/v17/matching_results_v17.tsv
    $REPO_DIR/student_resource/dataset/test/test_source{}.tsv
    {SP}/fr_rerun/out/france_assign_partial.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
W = f"{WORK_DIR}/push/frrescue"
SP = f"{WORK_DIR}"
V17 = f"{REPO_DIR}/output/v17/matching_results_v17.tsv"
RAW = f"{REPO_DIR}/student_resource/dataset/test/test_source{{}}.tsv"
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(f"{W}/work.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp'")
c.execute(f"""create or replace table v17 as
  select {ID.format(c='source1_entity_id')} as s1, {ID.format(c='m')} as rid from (
    select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as m
    from read_csv('{V17}', delim='\t', header=true, all_varchar=true, quote='', escape='')
    where matched_entity_ids is not null and matched_entity_ids <> '')""")
c.execute(f"""create or replace table v17s1 as select {ID.format(c='source1_entity_id')} as s1, coalesce(matched_entity_ids,'') = '' as empty
    from read_csv('{V17}', delim='\t', header=true, all_varchar=true, quote='', escape='')""")
c.execute(f"""create or replace table nm as select {ID.format(c='entity_id')} as eid, country, name_full, name_core, address, address_missing
    from read_parquet('{SP}/fr_rerun/data/fr_new/test_s*.parquet')""")
for s in (1, 2, 3):
    c.execute(f"""create or replace table raw{s} as select * from read_csv('{RAW.format(s)}', delim='\t', header=true, all_varchar=true, quote='', escape='')""")
print(c.execute("describe raw2").fetchall())
print("v17 pairs", c.execute("select count(*), count(distinct rid), count(distinct s1) from v17").fetchone())
print("v17 s1 rows", c.execute("select count(*), sum(empty::int) from v17s1").fetchone())
print("fr_new by src", c.execute("select eid // 10000000000, country, count(*) from nm group by all order by 1").fetchall())
fr = "(select eid from nm where eid // 10000000000 = 1)"
print("France S1 in v17s1", c.execute(f"select count(*), sum(empty::int), avg(empty::int) from v17s1 where s1 in {fr}").fetchone())
print("v17 French pairs", c.execute(f"select count(*), count(distinct rid) from v17 where s1 in {fr}").fetchone())
c.execute(f"create or replace table fap as select * from '{SP}/fr_rerun/out/france_assign_partial.parquet'")
print("partial vs v17 French", c.execute(f"""select (select count(*) from fap), (select count(*) from (select rid,s1 from fap except select rid,s1 from v17 where s1 in {fr})),
   (select count(*) from (select rid,s1 from v17 where s1 in {fr} except select rid,s1 from fap))""").fetchone())
