"""Builds the cross-encoder training pairs (positives and hard negatives) from the training split.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {R}/models/matcher.txt
    {R}/data/features/train.parquet
    {R}/data/norm/train_s1.parquet
    {CE}/train_pairs.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, json, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb, duckdb
R = f'{REPO_DIR}'
CE = f'{WORK_DIR}/ce'
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
t = time.time()
feats = json.load(open(f'{R}/models/matcher.json'))['features']
m = lgb.Booster(model_file=f'{R}/models/matcher.txt')
pf = pq.ParquetFile(f'{R}/data/features/train.parquet')
out = []
rng = np.random.default_rng(7)
for b in pf.iter_batches(batch_size=500_000, columns=['rid', 's1', 'label'] + feats):
    X = np.column_stack([b.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in feats])
    p = m.predict(X, num_threads=6).astype(np.float32)
    keep = ((p >= 0.003) & (p <= 0.997)) | (rng.random(len(p)) < 0.015)
    out.append(pa.table({'rid': b.column('rid').filter(pa.array(keep)), 's1': b.column('s1').filter(pa.array(keep)),
                         'label': b.column('label').filter(pa.array(keep)), 'p1': p[keep]}))
    print(len(out), sum(len(o) for o in out), f'{time.time()-t:.0f}s', flush=True)
tab = pa.concat_tables(out)
c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=4")
c.register('pairs', tab)
c.execute(f"""COPY (SELECT p.rid, p.s1, p.label, p.p1,
      coalesce(q.name_full, '') || ' | ' || coalesce(q.address, '') AS q_text,
      coalesce(s.name_full, '') || ' | ' || coalesce(s.address, '') AS s_text, q.country
    FROM pairs p
    JOIN (SELECT {ID.format(c='entity_id')} AS eid, name_full, address, country FROM read_parquet('{R}/data/norm/train_s[23].parquet')) q ON q.eid = p.rid
    JOIN (SELECT {ID.format(c='entity_id')} AS eid, name_full, address FROM read_parquet('{R}/data/norm/train_s1.parquet')) s ON s.eid = p.s1
    ORDER BY hash(p.rid, p.s1)) TO '{CE}/train_pairs.parquet' (FORMAT parquet)""")
print(c.execute(f"SELECT count(*), avg(label), count(*) FILTER (WHERE p1 BETWEEN 0.003 AND 0.997) FROM '{CE}/train_pairs.parquet'").fetchall(), f'{time.time()-t:.0f}s')
print(c.execute(f"SELECT q_text, s_text, label, p1 FROM '{CE}/train_pairs.parquet' LIMIT 5").fetchall())
