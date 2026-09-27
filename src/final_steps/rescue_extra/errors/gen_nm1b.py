"""Step 6 (name-key rescue). Experiment/validation script gen_nm1b.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    $REPO_DIR/data/candidates
    {N}/{split}_s1.parquet
    {E}/valid_v15_assigned_full.parquet
    {E}/valid_rbest.parquet
    {R}/valid_v11_assigned.parquet
    {R}/valid_v11_scores.parquet
    {CAND}/train.parquet
    {R}/valid_rescue.parquet
    $REPO_DIR/output/v18/matching_results_v18.tsv
    {CAND}/test.parquet
    {R}/test_rescue.parquet

Outputs:
    {E}/{split}_nm1b.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time, duckdb
S = f'{WORK_DIR}'
E = f'{S}/top50/errors'
R = f'{S}/rescue'
N = f'{REPO_DIR}/data/norm'
CAND = f'{REPO_DIR}/data/candidates'
ID = "cast(substr({c},2,1) as bigint)*10000000000+cast(substr({c},4) as bigint)"
split = sys.argv[1]
DFMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
t0 = time.time()
c = duckdb.connect(f'{E}/nm1b_{split}.duckdb')
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
STOP = ['limited','ltd','llc','llp','lp','inc','pvt','corp','corporation','co','company','the','and','of','et','pllc','plc','pc','private','public','m','s']
st = ", ".join(f"'{w}'" for w in STOP)
c.execute(f"CREATE OR REPLACE MACRO toks(n) AS list_sort(list_distinct(list_filter(string_split(coalesce(n,''), ' '), x -> x <> '' AND x NOT IN ({st}))))")
c.execute("""CREATE OR REPLACE MACRO dkeys(t) AS CASE WHEN len(t) BETWEEN 2 AND 8 THEN list_concat([array_to_string(t,' ')],
   list_transform(range(1, len(t)+1), i -> array_to_string(list_concat(t[1:i-1], t[i+1:]), ' '))) WHEN len(t) = 1 THEN [t[1]] ELSE [] END""")
c.execute(f"""CREATE OR REPLACE TABLE s1 AS SELECT {ID.format(c='entity_id')} AS s1, country, name_full, coalesce(name_full,'') || ' | ' || coalesce(address,'') AS text
  FROM '{N}/{split}_s1.parquet' WHERE country IN ('us','india')""")
c.execute("CREATE OR REPLACE TABLE s1k AS SELECT s1, country, unnest(dkeys(toks(name_full))) AS k FROM s1")
c.execute("CREATE OR REPLACE TABLE kdf AS SELECT country, k, count(DISTINCT s1) df FROM s1k GROUP BY ALL HAVING count(DISTINCT s1) <= " + str(DFMAX))
if split == 'train':
    c.execute(f"""CREATE OR REPLACE TABLE blocked AS
      SELECT rid FROM '{E}/valid_v15_assigned_full.parquet' UNION SELECT rid FROM '{E}/valid_rbest.parquet' WHERE ps >= 0.7
      UNION SELECT rid FROM '{R}/valid_v11_assigned.parquet'
      UNION SELECT rid FROM '{R}/valid_v11_scores.parquet' GROUP BY rid HAVING max(p2) >= 0.85
      UNION SELECT rid FROM '{S}/wf/hunt-odd-one-out-model/runs/new_slice_oof.parquet' GROUP BY rid HAVING max(p2) >= 0.85""")
    old = f"SELECT rid, s1 FROM '{CAND}/train.parquet' UNION ALL SELECT rid, s1 FROM '{R}/valid_rescue.parquet'"
else:
    c.execute(f"""CREATE OR REPLACE TABLE blocked AS SELECT DISTINCT {ID.format(c='m')} AS rid FROM (SELECT trim(unnest(string_split(matched_entity_ids, ','))) AS m
      FROM read_csv('{REPO_DIR}/output/v18/matching_results_v18.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')
      WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '') WHERE m <> ''""")
    old = f"SELECT rid, s1 FROM '{CAND}/test.parquet' UNION ALL SELECT rid, s1 FROM '{R}/test_rescue.parquet'"
c.execute(f"""CREATE OR REPLACE TABLE rec AS SELECT * FROM (SELECT {ID.format(c='entity_id')} AS rid, country, name_full, address_missing,
    coalesce(name_full,'') || ' | ' || coalesce(address,'') AS text FROM read_parquet(['{N}/{split}_s2.parquet','{N}/{split}_s3.parquet']) WHERE country IN ('us','india') AND address_missing) x
    ANTI JOIN blocked USING (rid) ANTI JOIN (SELECT DISTINCT rid FROM read_parquet('{CAND}/{split}.parquet')) cr USING (rid)""")
print('blocked', c.execute("select count(*) from blocked").fetchone(), 'eligible records', c.execute("select count(*) from rec").fetchone(), f'{time.time()-t0:.0f}s', flush=True)
c.execute("CREATE OR REPLACE TABLE rk AS SELECT rid, country, unnest(dkeys(toks(name_full))) AS k FROM rec")
c.execute("""CREATE OR REPLACE TABLE pairs AS SELECT DISTINCT r.rid, s.s1 FROM rk r JOIN kdf d USING (country, k) JOIN s1k s ON s.k = r.k AND s.country = r.country""")
print('raw pairs', c.execute("select count(*), count(distinct rid) from pairs").fetchone(), f'{time.time()-t0:.0f}s', flush=True)
c.execute(f"CREATE OR REPLACE TABLE pairs AS SELECT p.* FROM pairs p ANTI JOIN ({old}) o USING (rid, s1)")
c.execute("""CREATE OR REPLACE TABLE pairs AS SELECT p.rid, p.s1, q.text AS q_text, s.text AS s_text, q.country, q.address_missing::int AS am,
    jaro_winkler_similarity(q.name_full, s.name_full) AS jw FROM pairs p JOIN rec q USING (rid) JOIN s1 s USING (s1)""")
c.execute("CREATE OR REPLACE TABLE pairs AS SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY rid ORDER BY jw DESC, s1) rn FROM pairs) WHERE rn <= 3")
if split == 'train':
    c.execute(f"CREATE OR REPLACE TABLE pairs AS SELECT p.*, (t.rid IS NOT NULL)::int AS label FROM pairs p LEFT JOIN '{S}/wf/hunt-odd-one-out-model/truth.parquet' t USING (rid, s1)")
    print(c.execute("select am, count(*), sum(label), count(distinct rid) from pairs group by am").fetchall())
c.execute(f"COPY pairs TO '{E}/{split}_nm1b.parquet' (FORMAT parquet)")
print('final', c.execute("select count(*), count(distinct rid) from pairs").fetchone(), f'{time.time()-t0:.0f}s', flush=True)
