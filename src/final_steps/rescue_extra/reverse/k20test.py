"""Step 6 (reverse additions). Experiment/validation script k20test.py for this step.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data/norm
    {W}/k20_stage1.txt
    {W}/k20_tf.txt
    {W}/test_rev20.parquet
    {N}/test_s1.parquet
    {N}/test_s2.parquet
    {N}/test_s3.parquet

Outputs:
    {W}/k20_test_sel.parquet
    {W}/k20_test_ce_in.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import time, shutil
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from rapidfuzz import fuzz
S = f'{WORK_DIR}'
W = f'{S}/top50/reverse'
N = f'{REPO_DIR}/data/norm'
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
F = ['score', 'nk', 'rk', 'ns1_q', 'ns1_s', 'nrev', 'rmax', 'smax', 's2nd', 'tsr', 'tsort', 'rcomp', 'exact', 'qaddr', 'atsr', 'aovl', 'anum', 'qlen', 'slen', 'india', 'rel']
st1 = lgb.Booster(model_file=f'{W}/k20_stage1.txt')
tf = float(open(f'{W}/k20_tf.txt').read())
t0 = time.time()
c = duckdb.connect()
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp_k20t'")
c.execute(f"CREATE TABLE rev AS SELECT * FROM '{W}/test_rev20.parquet'")
c.execute(f"CREATE TABLE s1n AS SELECT country, coalesce(name_core, '') AS nc, count(*) AS ns1 FROM '{N}/test_s1.parquet' GROUP BY ALL")
c.execute(f"""CREATE TABLE q AS SELECT {ID} AS rid, country, coalesce(name_core, '') AS nc, coalesce(name_full, '') AS nf, coalesce(address, '') AS ad
    FROM read_parquet(['{N}/test_s2.parquet', '{N}/test_s3.parquet']) WHERE {ID} IN (SELECT rid FROM rev)""")
c.execute(f"""CREATE TABLE s AS SELECT {ID} AS s1, country, coalesce(name_core, '') AS nc, coalesce(name_full, '') AS nf, coalesce(address, '') AS ad
    FROM '{N}/test_s1.parquet' WHERE {ID} IN (SELECT s1 FROM rev)""")
c.execute("CREATE TABLE rcnt AS SELECT rid, count(*) AS nrev, max(score) AS rmax FROM rev GROUP BY rid")
c.execute("CREATE TABLE scnt AS SELECT s1, max(score) AS smax, max(score) FILTER (WHERE rk = 2) AS s2nd FROM rev GROUP BY s1")
print('prepared', f'{time.time()-t0:.0f}s', flush=True)
def atok(x):
    """Return the set of address tokens with at least three characters or digits."""
    return set(t for t in x.split() if len(t) >= 3 or t.isdigit())
out = []
CH = 8
for ch in range(CH):
    d = c.execute(f"""SELECT r.rid, r.s1, r.score, r.nk, r.rk, q.country, q.nc AS qn, s.nc AS sn, q.nf || ' | ' || q.ad AS q_text, s.nf || ' | ' || s.ad AS s_text,
        q.ad AS qa, s.ad AS sa, coalesce(n1.ns1, 0) AS ns1_q, coalesce(n2.ns1, 0) AS ns1_s, rc.nrev, rc.rmax, sc.smax, coalesce(sc.s2nd, 0) AS s2nd
        FROM rev r JOIN q USING (rid) JOIN s USING (s1)
        LEFT JOIN s1n n1 ON n1.country = q.country AND n1.nc = q.nc LEFT JOIN s1n n2 ON n2.country = s.country AND n2.nc = s.nc
        JOIN rcnt rc USING (rid) JOIN scnt sc USING (s1) WHERE hash(r.rid) % {CH} = {ch}""").fetch_arrow_table()
    qn, sn = d['qn'].to_pylist(), d['sn'].to_pylist()
    qa, sa = d['qa'].to_pylist(), d['sa'].to_pylist()
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
    t = d.drop_columns(['qa', 'sa'])
    for k, v in feat.items():
        t = t.append_column(k, pa.array(v))
    df = t.to_pandas()
    df['india'] = (df.country == 'india').astype(float); df['rel'] = df.score / df.rmax
    df['p1'] = st1.predict(df[F].values.astype(np.float32))
    out.append(df[df.p1 >= tf])
    print(f'chunk {ch+1}/{CH}: {len(df):,} pairs, kept {len(out[-1]):,} ({time.time()-t0:.0f}s)', flush=True)
import pandas as pd
res = pd.concat(out)
res.to_parquet(f'{W}/k20_test_sel.parquet', index=False)
res[['rid', 's1', 'q_text', 's_text']].to_parquet(f'{W}/k20_test_ce_in.parquet', index=False)
print('selected', len(res), 'records', res.rid.nunique(), res.groupby('country').size().to_dict(), f'{time.time()-t0:.0f}s')
c.close(); shutil.rmtree(f'{W}/tmp_k20t', ignore_errors=True)
