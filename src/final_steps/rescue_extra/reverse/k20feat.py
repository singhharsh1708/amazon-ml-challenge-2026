"""Step 6 (reverse additions). Experiment/validation script k20feat.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    {W}/{src}.parquet
    {W}/k20_cand_rids.parquet
    {H}/valid_s1.parquet
    {N}/{split}_s1.parquet
    {N}/{split}_s2.parquet
    {N}/{split}_s3.parquet
    {H}/truth.parquet

Outputs:
    {W}/k20_comp_all.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time, shutil
import duckdb, numpy as np, pyarrow as pa
from rapidfuzz import fuzz
S = f'{WORK_DIR}'
H = f'{S}/wf/hunt-odd-one-out-model'
W = f'{S}/top50/reverse'
N = f'{REPO_DIR}/data/norm'
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
split = sys.argv[1]
t0 = time.time()
c = duckdb.connect()
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp_k20c'")
src = sys.argv[2]
c.execute(f"CREATE TABLE revu AS SELECT * FROM '{W}/{src}.parquet'")
if split == 'train':
    c.execute(f"CREATE TABLE rev AS SELECT * FROM revu WHERE rid IN (SELECT rid FROM '{W}/k20_cand_rids.parquet') AND s1 NOT IN (SELECT s1 FROM '{H}/valid_s1.parquet')")
else:
    c.execute("CREATE TABLE rev AS SELECT * FROM revu")
c.execute(f"""CREATE TABLE s1n AS SELECT country, coalesce(name_core, '') AS nc, count(*) AS ns1 FROM '{N}/{split}_s1.parquet' GROUP BY ALL""")
c.execute(f"""CREATE TABLE q AS SELECT {ID} AS rid, country, coalesce(name_core, '') AS nc, coalesce(name_full, '') AS nf, coalesce(address, '') AS ad
    FROM read_parquet(['{N}/{split}_s2.parquet', '{N}/{split}_s3.parquet']) WHERE {ID} IN (SELECT rid FROM rev)""")
c.execute(f"""CREATE TABLE s AS SELECT {ID} AS s1, country, coalesce(name_core, '') AS nc, coalesce(name_full, '') AS nf, coalesce(address, '') AS ad
    FROM '{N}/{split}_s1.parquet' WHERE {ID} IN (SELECT s1 FROM rev)""")
c.execute("CREATE TABLE rcnt AS SELECT rid, count(*) AS nrev, max(score) AS rmax FROM revu GROUP BY rid")
c.execute("CREATE TABLE scnt AS SELECT s1, max(score) AS smax, count(*) FILTER (WHERE rk = 2) AS has2, max(score) FILTER (WHERE rk = 2) AS s2nd FROM rev GROUP BY s1")
tr = f"LEFT JOIN (SELECT rid, s1, 1 AS y FROM '{H}/truth.parquet') t USING (rid, s1)" if split == 'train' else ""
ysel = "coalesce(t.y, 0) AS y," if split == 'train' else ""
d = c.execute(f"""SELECT r.rid, r.s1, r.score, r.nk, r.rk, {ysel} q.country, q.nc AS qn, s.nc AS sn, q.nf || ' | ' || q.ad AS q_text, s.nf || ' | ' || s.ad AS s_text,
    q.ad AS qa, s.ad AS sa, coalesce(n1.ns1, 0) AS ns1_q, coalesce(n2.ns1, 0) AS ns1_s, rc.nrev, rc.rmax, sc.smax, coalesce(sc.s2nd, 0) AS s2nd
    FROM rev r JOIN q USING (rid) JOIN s USING (s1) {tr}
    LEFT JOIN s1n n1 ON n1.country = q.country AND n1.nc = q.nc LEFT JOIN s1n n2 ON n2.country = s.country AND n2.nc = s.nc
    JOIN rcnt rc USING (rid) JOIN scnt sc USING (s1)""").fetch_arrow_table()
print(d.num_rows, f'{time.time()-t0:.0f}s', flush=True)
qn, sn = d['qn'].to_pylist(), d['sn'].to_pylist()
qa, sa = d['qa'].to_pylist(), d['sa'].to_pylist()
def atok(x):
    """Return the set of address tokens with at least three characters or digits."""
    return set(t for t in x.split() if len(t) >= 3 or t.isdigit())
feat = {
    'tsr': np.array([fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)], np.float32),
    'tsort': np.array([fuzz.token_sort_ratio(a, b) for a, b in zip(qn, sn)], np.float32),
    'rcomp': np.array([fuzz.ratio(a.replace(' ', ''), b.replace(' ', '')) for a, b in zip(qn, sn)], np.float32),
    'exact': np.array([float(a == b) for a, b in zip(qn, sn)], np.float32),
    'qaddr': np.array([float(bool(a.strip())) for a in qa], np.float32),
    'atsr': np.array([fuzz.token_set_ratio(a, b) if a.strip() else -1 for a, b in zip(qa, sa)], np.float32),
    'aovl': np.array([len(atok(a) & atok(b)) for a, b in zip(qa, sa)], np.float32),
    'anum': np.array([len(set(t for t in a.split() if t.isdigit()) & set(t for t in b.split() if t.isdigit())) for a, b in zip(qa, sa)], np.float32),
    'qlen': np.array([len(a.split()) for a in qn], np.float32),
    'slen': np.array([len(a.split()) for a in sn], np.float32),
}
out = d.drop_columns(['qa', 'sa'])
for k, v in feat.items():
    out = out.append_column(k, pa.array(v))
import pyarrow.parquet as pq
pq.write_table(out, f'{W}/k20_comp_all.parquet')
print('done', f'{time.time()-t0:.0f}s')
c.close(); shutil.rmtree(f'{W}/tmp_k20c', ignore_errors=True)
