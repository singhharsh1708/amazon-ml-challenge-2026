"""France re-run step 3: stage-1 and stage-2 scoring of the France candidates.

Step context: Step 4 input (France re-run). Re-runs the France slice of the pipeline with the fixed address normalization and writes france_assign_partial.parquet.

Command-line arguments used: argv[1], argv[2], argv[3].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {nd}/test_s1.parquet
    {nd}/test_s{s}.parquet
    {R}/models/matcher.txt
    {R}/data/v10/model/stage2_odd.txt
    {R}/models/v6/stage2.txt

Outputs:
    {w}/feat.parquet
    {w}/pred1.parquet
    {w}/pred2.parquet
"""
import json
import os
import shutil
import sys

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.process import cpdist

from frc import F, R, ID, con, normdir, work, Timer
import build_features as BF
import train_stage2 as T2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofeats import pair_features, sibling_index, NAMES

FEAT_PARTS = 16
S2_PARTS = 8


def add_fuzzy(batch):
    """Add the fuzzy string features to a record batch."""
    cols = {name: batch.column(name).to_pylist() for name in
            ("q_name", "s_name", "q_name_full", "s_name_full", "q_address", "s_address")}
    arrays = []
    for feature, left, right, scorer in BF.FUZZY:
        values = cpdist(cols[left], cols[right], scorer=scorer, workers=3, dtype=np.float32)
        arrays.append((feature, pa.array(values)))
    keep = [n for n in batch.schema.names if n not in cols]
    table = batch.select(keep)
    for feature, values in arrays:
        table = table.append_column(feature, values)
    return table


def features(variant, cand, T):
    """Build the pair features of a variant."""
    nd = normdir(variant)
    w = work(variant)
    c = con(f"feat_{variant}", f"{F}/tmp/feat_{variant}.duckdb")
    c.execute(f"CREATE OR REPLACE TABLE cand AS SELECT * FROM read_parquet('{cand}')")
    c.execute("""CREATE OR REPLACE TABLE rec_stats AS
        SELECT rid, max(score) AS top1, max(score) FILTER (WHERE rk = 2) AS top2, count(*) AS ncand FROM cand GROUP BY rid""")
    c.execute("""CREATE OR REPLACE TABLE s1_stats AS
        SELECT s1, count(*) AS n_cand, count(*) FILTER (WHERE rk = 1) AS n_top1, max(score) AS max_score FROM cand GROUP BY s1""")
    idx = BF.ID_EXPR.format(col="entity_id")
    c.execute(f"CREATE OR REPLACE TABLE s1_text AS {BF.TEXT_SQL.format(id=idx, path=f'{nd}/test_s1.parquet')}")
    c.execute("CREATE OR REPLACE TABLE q_text AS " + " UNION ALL ".join(
        BF.TEXT_SQL.format(id=idx, path=f"{nd}/test_s{s}.parquet") for s in (2, 3)))
    c.execute("CREATE OR REPLACE TABLE pairs AS SELECT *, 'test' AS part FROM cand")
    c.execute("DROP TABLE cand")
    T(f"{variant}: features prepared")
    sql = BF.PAIR_SQL.format(label="NULL::int", label_join="")
    writer = None
    rows = 0
    for chunk in range(FEAT_PARTS):
        batch = c.execute(sql.replace("{parts}", str(FEAT_PARTS)).replace("{part}", str(chunk))).to_arrow_table()
        table = BF.add_relative(add_fuzzy(batch))
        if writer is None:
            writer = pq.ParquetWriter(f"{w}/feat.parquet", table.schema, compression="zstd")
        writer.write_table(table)
        rows += table.num_rows
    writer.close()
    c.close()
    for p in (f"{F}/tmp/feat_{variant}.duckdb", f"{F}/tmp/feat_{variant}.duckdb.wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(f"{F}/tmp/feat_{variant}", ignore_errors=True)
    T(f"{variant}: features {rows:,} pairs")


def stage1(variant, T):
    """Score a variant with the stage-1 matcher."""
    w = work(variant)
    meta = json.loads(open(f"{R}/models/matcher.json").read())
    model = lgb.Booster(model_file=f"{R}/models/matcher.txt")
    pf = pq.ParquetFile(f"{w}/feat.parquet")
    writer = None
    for batch in pf.iter_batches(batch_size=1_000_000):
        X = np.column_stack([batch.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in meta["features"]])
        p = model.predict(X, num_threads=3).astype(np.float32)
        out = pa.table({"rid": batch.column("rid"), "s1": batch.column("s1"), "p": p} | {n: batch.column(n) for n in T2.BASE})
        if writer is None:
            writer = pq.ParquetWriter(f"{w}/pred1.parquet", out.schema, compression="zstd")
        writer.write_table(out)
    writer.close()
    T(f"{variant}: stage 1 done")


def stage2(variant, T):
    """Score a variant with the stage-2 model and the France guard."""
    nd = normdir(variant)
    w = work(variant)
    pred = f"{w}/pred1.parquet"
    src = f"(SELECT *, NULL::int AS label FROM read_parquet('{pred}'))"
    c = con(f"s2_{variant}", f"{F}/tmp/s2_{variant}.duckdb")
    c.execute(f"CREATE OR REPLACE TABLE best AS SELECT rid, arg_max(s1, p) AS s1, max(p) AS p FROM read_parquet('{pred}') GROUP BY rid")
    c.execute(f"CREATE OR REPLACE TABLE nm AS SELECT {ID} AS eid, country, name_full, address FROM read_parquet('{nd}/test_s*.parquet')")
    T2.stage2_query(c, src)
    model = lgb.Booster(model_file=f"{R}/data/v10/model/stage2_odd.txt")
    guard = lgb.Booster(model_file=f"{R}/models/v6/stage2.txt")
    feats, gfeats = model.feature_name(), guard.feature_name()
    sel = T2.SELECT_SQL.format(src=src, base=", ".join(T2.BASE))
    writer = None
    n = 0
    for k in range(S2_PARTS):
        sib = c.execute(f"""SELECT b.s1, b.rid AS sid, b.p AS sp1, b.rid // 10000000000 AS src, q.name_full AS name, q.address AS addr
            FROM best b JOIN nm q ON q.eid = b.rid WHERE b.p >= 0.5 AND b.s1 % {S2_PARTS} = {k}""").to_arrow_table()
        idx = sibling_index(sib)
        del sib
        txt = c.execute(f"""SELECT p.rid, p.s1, q.country, q.name_full AS q_name, q.address AS q_addr, s.name_full AS s_name, s.address AS s_addr
            FROM read_parquet('{pred}') p JOIN nm q ON q.eid = p.rid JOIN nm s ON s.eid = p.s1 WHERE p.p >= 0.001 AND p.s1 % {S2_PARTS} = {k}""").to_arrow_table()
        cols = {x: txt.column(x).to_pylist() for x in ["rid", "s1", "country", "q_name", "q_addr", "s_name", "s_addr"]}
        rows = [pair_features(r, co, qn, qa, sn, sa, idx.get(s, ())) for r, s, co, qn, qa, sn, sa in
                zip(cols["rid"], cols["s1"], cols["country"], cols["q_name"], cols["q_addr"], cols["s_name"], cols["s_addr"])]
        of = pa.table({"rid": txt.column("rid"), "s1": txt.column("s1")} | {x: pa.array(np.array([r[x] for r in rows], dtype=np.float32)) for x in NAMES})
        del rows, cols, txt, idx
        c.register("of_arrow", of)
        reader = c.execute(f"""SELECT q.*, o.* EXCLUDE (rid, s1) FROM (SELECT * FROM ({sel}) WHERE s1 % {S2_PARTS} = {k}) q
            LEFT JOIN of_arrow o USING (rid, s1)""").fetch_record_batch(rows_per_batch=1_000_000)
        for batch in reader:
            X = np.column_stack([batch.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in feats])
            G = np.column_stack([batch.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in gfeats])
            out = pa.table({"rid": batch.column("rid"), "s1": batch.column("s1"),
                            "p": model.predict(X, num_threads=3).astype(np.float32),
                            "p_guard": guard.predict(G, num_threads=3).astype(np.float32),
                            "p1": batch.column("p"), "fne": batch.column("first_num_equal"), "veto": batch.column("o_veto")})
            if writer is None:
                writer = pq.ParquetWriter(f"{w}/pred2.parquet", out.schema, compression="zstd")
            writer.write_table(out)
            n += out.num_rows
        c.unregister("of_arrow")
        T(f"{variant}: stage 2 part {k}: {n:,} pairs")
    writer.close()
    c.close()
    for p in (f"{F}/tmp/s2_{variant}.duckdb", f"{F}/tmp/s2_{variant}.duckdb.wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(f"{F}/tmp/s2_{variant}", ignore_errors=True)


def main(variant, cand, steps):
    """Parse the command line and run the step."""
    T = Timer()
    if "feat" in steps:
        features(variant, cand, T)
    if "s1" in steps:
        stage1(variant, T)
    if "s2" in steps:
        stage2(variant, T)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
