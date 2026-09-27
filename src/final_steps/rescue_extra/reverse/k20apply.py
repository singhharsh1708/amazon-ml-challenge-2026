"""Applies the reverse (k20) models to test and writes extras_adds.parquet.

Step context: Step 6 (reverse additions). Reverse-direction candidate search (k20) with its own stage-1/stage-2 models; k20apply writes extras_adds.parquet.

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/reverse
    $WORK_DIR/rescue
    $REPO_DIR/data/norm
    {W}/k20_test_pn.parquet
    {W}/k20_test_rf.parquet
    {W}/k20_test_sel.parquet
    {W}/k20_test_ce2.parquet
    {W}/k20_stage2.txt
    $REPO_DIR/output/v18/matching_results_v18.tsv
    {R}/test_rescue_adds.parquet
    $REPO_DIR/data/candidates/test.parquet
    {N}/test_s1.parquet

Outputs:
    {W}/test_additions.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
W = f'{WORK_DIR}/top50/reverse'
R = f'{WORK_DIR}/rescue'
N = f'{REPO_DIR}/data/norm'
T, M = float(sys.argv[1]), float(sys.argv[2])
F = ['c2', 'npool_q', 'npool_s', 'ncand', 'fmax', 'tsr', 'tsort', 'rcomp', 'exact', 'qaddr', 'atsr', 'aovl', 'anum', 'ns1_q', 'ns1_s', 'score', 'rel', 'ssr', 'gap', 'nrev', 'rk', 'qlen', 'slen', 'india']
PN = pd.read_parquet(f'{W}/k20_test_pn.parquet'); RF = pd.read_parquet(f'{W}/k20_test_rf.parquet')
v = pd.read_parquet(f'{W}/k20_test_sel.parquet').merge(pd.read_parquet(f'{W}/k20_test_ce2.parquet').rename(columns={'ce': 'c2'}), on=['rid', 's1'])
v = v.merge(RF, on='rid', how='left')
v = v.merge(PN.rename(columns={'nc': 'qn', 'npool': 'npool_q'}), on=['country', 'qn'], how='left')
v = v.merge(PN.rename(columns={'nc': 'sn', 'npool': 'npool_s'}), on=['country', 'sn'], how='left')
v[['npool_q', 'npool_s', 'ncand', 'fmax']] = v[['npool_q', 'npool_s', 'ncand', 'fmax']].fillna(0)
v['ssr'] = v.score / v.smax; v['gap'] = v.smax - v.s2nd
v['p'] = lgb.Booster(model_file=f'{W}/k20_stage2.txt').predict(v[F].values.astype(np.float32))
v = v.sort_values(['rid', 'p'], ascending=[True, False])
v['sec'] = v.groupby('rid').p.shift(-1)
b = v.drop_duplicates('rid').copy(); b['nsmax'] = b.sec.fillna(-1)
add = b[(b.p >= T) & (b.nsmax < M)]
print('test pairs', len(v), 'records', len(b), f'additions (p >= {T}, competitor < {M})', len(add), add.groupby('country').size().to_dict())
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
c = duckdb.connect(); c.execute("SET memory_limit='2GB'; SET threads=3")
c.register('a', add[['rid', 's1']])
c.execute(f"""CREATE TABLE asg AS SELECT DISTINCT {ID.format(c='m')} AS rid FROM (SELECT trim(unnest(string_split(matched_entity_ids, ','))) AS m
    FROM read_csv('{REPO_DIR}/output/v18/matching_results_v18.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '') WHERE m <> ''""")
print('checks: rows/distinct rid', c.execute("SELECT count(*), count(DISTINCT rid) FROM a").fetchone(),
      '| in v18 assigned', c.execute("SELECT count(*) FROM a WHERE rid IN (SELECT rid FROM asg)").fetchone()[0],
      '| in rescue adds', c.execute(f"SELECT count(*) FROM a WHERE rid IN (SELECT rid FROM '{R}/test_rescue_adds.parquet')").fetchone()[0],
      '| pair in test candidates', c.execute(f"SELECT count(*) FROM a SEMI JOIN read_parquet('{REPO_DIR}/data/candidates/test.parquet') k USING (rid, s1)").fetchone()[0],
      '| s1 country', c.execute(f"SELECT country, count(*) FROM a JOIN (SELECT {ID.format(c='entity_id')} AS s1, country FROM '{N}/test_s1.parquet') s USING (s1) GROUP BY 1").fetchall(),
      '| distinct s1', c.execute("SELECT count(DISTINCT s1) FROM a").fetchone()[0])
add[['rid', 's1']].to_parquet(f'{W}/test_additions.parquet', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 75)
print(add.sample(20, random_state=1)[['q_text', 's_text', 'p', 'nsmax']].to_string())
