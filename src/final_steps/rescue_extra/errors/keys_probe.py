"""Step 6 (name-key rescue). Experiment/validation script keys_probe.py for this step.

Step context: Step 6 (name-key rescue). Error analysis and rescue families (nm1*, hc, adr); gen_* build candidates, eval_* validate on the held-out split, apply_test writes test additions.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/errors
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
E = f'{WORK_DIR}/top50/errors'
c = duckdb.connect(f'{E}/err.duckdb')
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{E}/tmp'")
STOP = ['limited','ltd','llc','llp','lp','inc','pvt','corp','corporation','co','company','the','and','of','et','pllc','plc','pc','private','public','m','s']
st = ", ".join(f"'{w}'" for w in STOP)
c.execute(f"""CREATE OR REPLACE MACRO toks(n) AS list_sort(list_distinct(list_filter(string_split(coalesce(n,''), ' '), x -> x <> '' AND x NOT IN ({st}))))""")
c.execute("""CREATE OR REPLACE MACRO dkeys(t) AS CASE WHEN len(t) BETWEEN 2 AND 8 THEN list_concat([array_to_string(t,' ')],
   list_transform(range(1, len(t)+1), i -> array_to_string(list_concat(t[1:i-1], t[i+1:]), ' '))) WHEN len(t) = 1 THEN [t[1]] ELSE [] END""")
c.execute("""CREATE OR REPLACE TABLE s1k AS SELECT s1, country, unnest(dkeys(toks(name_full))) AS k FROM s1txt WHERE country IN ('us','india')""")
c.execute("CREATE OR REPLACE TABLE kdf AS SELECT country, k, count(DISTINCT s1) df FROM s1k GROUP BY ALL")
print(c.execute("SELECT count(*), count(*) FILTER (WHERE df<=3) FROM kdf").fetchall())
c.execute("""CREATE OR REPLACE TABLE fnk AS SELECT f.rid, f.s1, f.r_am, f.s_top IS NULL nocand, f.p_true IS NOT NULL in_cand, unnest(dkeys(toks(f.r_name))) AS k, f.country FROM fn2 f WHERE f.keep""")
print(c.execute("""SELECT r_am, nocand, in_cand, count(DISTINCT f.rid) n,
    count(DISTINCT f.rid) FILTER (WHERE s.s1 IS NOT NULL AND d.df<=3) hit3,
    count(DISTINCT f.rid) FILTER (WHERE s.s1 IS NOT NULL AND d.df<=10) hit10,
    count(DISTINCT f.rid) FILTER (WHERE s.s1 IS NOT NULL) hitany
    FROM fnk f LEFT JOIN s1k s ON s.s1=f.s1 AND s.k=f.k LEFT JOIN kdf d ON d.country=f.country AND d.k=f.k GROUP BY ALL ORDER BY ALL""").df().to_string())
