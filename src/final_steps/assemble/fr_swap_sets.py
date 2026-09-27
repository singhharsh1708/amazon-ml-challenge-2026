"""Compares the France rows of a matching file with the France re-run assignment and writes the removal and addition sets that swap them in. Usage: fr_swap_sets.py <matching.tsv> <france_assign.parquet> <rem_out.parquet> <add_out.parquet>.

Step context: Step 4 (decision and assembly). Turns a predictions parquet into matching_results/candidate_pairs TSVs, then applies removal/addition sets.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm/test_s1.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, duckdb
base_m, new_fr, out_rem, out_add = sys.argv[1:5]
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute("SET memory_limit='2GB'; SET threads=3")
c.execute(f"""CREATE TABLE m AS SELECT {ID.format(c='source1_entity_id')} s1, {ID.format(c='x')} rid FROM (SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) x
  FROM read_csv('{base_m}', delim='\t', header=true, all_varchar=true, quote='', escape='') WHERE coalesce(matched_entity_ids, '') <> '')""")
c.execute(f"CREATE TABLE fr AS SELECT {ID.format(c='entity_id')} s1 FROM read_parquet('{REPO_DIR}/data/norm/test_s1.parquet') WHERE country = 'france'")
c.execute("CREATE TABLE mf AS SELECT m.* FROM m JOIN fr USING (s1)")
c.execute(f"CREATE TABLE nf AS SELECT DISTINCT rid, s1 FROM read_parquet('{new_fr}')")
c.execute(f"COPY (SELECT rid, s1 FROM mf EXCEPT SELECT rid, s1 FROM nf) TO '{out_rem}' (FORMAT parquet)")
c.execute(f"COPY (SELECT rid, s1 FROM nf EXCEPT SELECT rid, s1 FROM mf) TO '{out_add}' (FORMAT parquet)")
print("base french pairs", c.execute("SELECT count(*) FROM mf").fetchone()[0], "new french pairs", c.execute("SELECT count(*) FROM nf").fetchone()[0],
      "removals", c.execute(f"SELECT count(*) FROM '{out_rem}'").fetchone()[0], "additions", c.execute(f"SELECT count(*) FROM '{out_add}'").fetchone()[0],
      "non-french s1 in new:", c.execute("SELECT count(*) FROM nf WHERE s1 NOT IN (SELECT s1 FROM fr)").fetchone()[0])
