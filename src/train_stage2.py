"""Train the stage 2 LightGBM re-ranker.

Stage 2 re-scores the stage 1 pairs of the validation slice with context features: the pair's
stage 1 probability, its gap to the record's best and second-best candidate, how many other
records chose the same source 1 entity and with what confidence, a subset of stage 1 features
and the odd features. Performance is reported as 2-fold out-of-fold macro F0.5 over a
threshold grid. The final model is trained on all validation-slice pairs.

Input: data/features/valid.parquet, data/features/valid_predictions.parquet,
data/features/valid_odd.parquet (optional), ground truth.
Output: models/stage2.txt, models/stage2.json, data/features/valid_stage2_oof.parquet.
Run from src/: python train_stage2.py
"""

import json
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, MODEL_DIR, TRAIN_DIR

FEAT_DIR = DATA_DIR / "features"
STAGE1_PRED = FEAT_DIR / "valid_predictions.parquet"
OOF_FILE = FEAT_DIR / "valid_stage2_oof.parquet"
ODD_FILE = FEAT_DIR / "valid_odd.parquet"
MODEL_FILE = MODEL_DIR / "stage2.txt"
META_FILE = MODEL_DIR / "stage2.json"

VALID_MOD = 20
FOLDS = 2
ROUNDS = 400
THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
COPY_OFFSET = 5_000_000_000
COPY_TENTHS = 0
TRAIN_IN_SLICE = True
EXCLUDED = set()

BASE = [
    "combined", "combined_gap", "name_token_set", "address_token_set", "name_full_ratio",
    "first_num_equal", "nums_subset", "num_common", "name_translit", "address_missing",
    "source", "rk", "score_ratio", "legal_clash", "first_num_diff",
]

RSTATS_SQL = """
    CREATE OR REPLACE TABLE rstats AS
    SELECT rid, max(p) AS r_top1, max(p, 2) AS r_top2, count(*) AS r_n FROM {src} GROUP BY rid
"""

SSTATS_SQL = """
    CREATE OR REPLACE TABLE sstats AS
    SELECT t.s1,
        count(*) FILTER (WHERE t.p = r.r_top1) AS s_n_best,
        count(*) FILTER (WHERE t.p = r.r_top1 AND t.p >= 0.5) AS s_n_conf,
        coalesce(sum(t.p) FILTER (WHERE t.p = r.r_top1), 0) AS s_sum_best,
        max(t.p, 2) FILTER (WHERE t.p = r.r_top1) AS s_top2,
        count(*) AS s_n_pairs,
        max(t.combined) AS s_best_combined
    FROM {src} t JOIN rstats r USING (rid)
    GROUP BY t.s1
"""

SELECT_SQL = """
    SELECT rid, s1, label, p,
        p - r_top1 AS p_gap_rid,
        CASE WHEN is_best = 1 THEN p - r_second ELSE p - r_top1 END AS p_margin,
        r_second, r_n, is_best,
        s_n_best - is_best AS s_n_best_other,
        s_n_conf - (is_best = 1 AND p >= 0.5)::int AS s_n_conf_other,
        s_sum_best - is_best * p AS s_sum_other,
        CASE
            WHEN s_top2 IS NULL THEN 0
            WHEN is_best = 1 AND p = s_top2[1] THEN coalesce(s_top2[2], 0)
            ELSE s_top2[1]
        END AS s_max_other,
        s_n_pairs,
        combined - s_best_combined AS combined_gap_s1,
        {base}
    FROM (
        SELECT t.*, r.r_top1, r.r_n,
            CASE WHEN len(r.r_top2) > 1 THEN r.r_top2[2] ELSE 0 END AS r_second,
            (t.p = r.r_top1)::int AS is_best,
            s.s_n_best, s.s_n_conf, s.s_sum_best, s.s_top2, s.s_n_pairs, s.s_best_combined
        FROM {src} t JOIN rstats r USING (rid) JOIN sstats s USING (s1)
    )
"""

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"


def stage2_query(con, src):
    """Build the record and source 1 statistics tables for a pair source and return the feature SELECT."""
    con.execute(RSTATS_SQL.format(src=src))
    con.execute(SSTATS_SQL.format(src=src))
    return SELECT_SQL.format(src=src, base=", ".join(BASE))


def stage2_features(con, pairs_sql):
    """Materialise a pair query and return its stage 2 feature table."""
    con.execute(f"CREATE OR REPLACE TABLE pairs AS {pairs_sql}")
    table = con.execute(stage2_query(con, "pairs")).to_arrow_table()
    con.execute("DROP TABLE pairs")
    return table


def feature_names(table):
    """Return the model feature columns of a stage 2 table."""
    return [n for n in table.schema.names if n not in ("rid", "s1", "label")]


def matrix(table, names):
    """Stack the given columns of an Arrow table into a float32 matrix."""
    return np.column_stack([
        table.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in names
    ])


def macro_f05(con, threshold, table="oof"):
    """Return validation macro F0.5 for a threshold using the best p2 per record from a table."""
    return con.execute(f"""
        WITH best AS (SELECT rid, arg_max(s1, p2) AS s1, max(p2) AS p FROM {table} GROUP BY rid),
        pred AS (SELECT s1, count(*) AS n_pred FROM best WHERE p >= {threshold} GROUP BY s1),
        hit AS (
            SELECT b.s1, count(*) AS tp FROM best b JOIN truth t USING (rid, s1)
            WHERE b.p >= {threshold} GROUP BY b.s1
        ),
        per AS (
            SELECT v.s1, coalesce(n.n_true, 0) AS n_true, coalesce(p.n_pred, 0) AS n_pred, coalesce(h.tp, 0) AS tp
            FROM valid_s1 v
            LEFT JOIN (SELECT s1, count(*) AS n_true FROM truth GROUP BY s1) n USING (s1)
            LEFT JOIN pred p USING (s1)
            LEFT JOIN hit h USING (s1)
        )
        SELECT avg(CASE
            WHEN n_true = 0 AND n_pred = 0 THEN 1.0
            WHEN tp = 0 THEN 0.0
            ELSE 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp) + (n_pred - tp))
        END) FROM per
    """).fetchone()[0]


def main():
    """Build stage 2 features, run out-of-fold evaluation, then fit and save the final model."""
    start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    gt = (TRAIN_DIR / "train_ground_truth.tsv").as_posix()
    con.execute(f"""
        CREATE TABLE truth AS
        SELECT {ID_EXPR.format(col='m')} AS rid, {ID_EXPR.format(col='source1_entity_id')} AS s1
        FROM (
            SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m
            FROM read_csv('{gt}', delim = '\t', header = true, all_varchar = true, quote = '', escape = '')
            WHERE matched_entity_ids <> ''
        )
    """)
    con.execute(f"""
        CREATE TABLE valid_s1 AS
        SELECT s1 FROM (
            SELECT {ID_EXPR.format(col='entity_id')} AS s1
            FROM read_parquet('{(DATA_DIR / 'norm' / 'train_s1.parquet').as_posix()}')
        ) WHERE hash(s1) % {VALID_MOD} = 0
    """)
    con.execute(f"""
        CREATE TABLE base_pairs AS
        SELECT f.rid, f.s1, f.label, pr.p, {", ".join(f"f.{c}" for c in BASE)}
        FROM read_parquet('{(FEAT_DIR / "valid.parquet").as_posix()}') f
        JOIN read_parquet('{STAGE1_PRED.as_posix()}') pr USING (rid, s1)
    """)
    con.execute(f"""
        CREATE TABLE aug_pairs AS
        SELECT * FROM base_pairs
        UNION ALL
        SELECT rid + {COPY_OFFSET} AS rid, s1, 0 AS label, p, {", ".join(BASE)}
        FROM base_pairs
        WHERE rid NOT IN (SELECT rid FROM truth) AND hash(rid * 7) % 10 < {COPY_TENTHS}
    """)
    odd_cols, odd_join = "", ""
    if ODD_FILE.exists():
        odd_cols = ", o.* EXCLUDE (rid, s1)"
        odd_join = f"LEFT JOIN read_parquet('{ODD_FILE.as_posix()}') o USING (rid, s1)"
    table = con.execute(
        f"SELECT q.*, (hash(q.s1) % {VALID_MOD} = 0) AS in_slice{odd_cols} "
        f"FROM ({stage2_query(con, 'aug_pairs')}) q {odd_join}"
    ).to_arrow_table()
    names = [n for n in table.schema.names if n not in ("rid", "s1", "label", "in_slice") and n not in EXCLUDED]
    X = matrix(table, names)
    y = table.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
    rid = table.column("rid").to_numpy()
    copy = (rid % 10_000_000_000) >= COPY_OFFSET
    fold = np.where(copy, rid - COPY_OFFSET, rid) % FOLDS
    trainable = table.column("in_slice").to_numpy(zero_copy_only=False) if TRAIN_IN_SLICE else np.ones(len(y), bool)
    print(f"stage 2 rows {len(y):,} ({int(copy.sum()):,} decoy copies), trainable {int(trainable.sum()):,}, "
          f"features {len(names)} ({time.time() - start:.0f}s)", flush=True)

    params = {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
        "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 6, "verbose": -1, "seed": 42,
    }
    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(FOLDS):
        train = (fold != k) & trainable
        model = lgb.train(params, lgb.Dataset(X[train], y[train], feature_name=names), ROUNDS)
        oof[fold == k] = model.predict(X[fold == k])
        print(f"fold {k} done ({time.time() - start:.0f}s)", flush=True)

    ids = {"rid": table.column("rid"), "s1": table.column("s1")}
    pq.write_table(pa.table(ids | {"p2": oof}), OOF_FILE)
    con.execute(f"CREATE TABLE oof AS SELECT * FROM read_parquet('{OOF_FILE.as_posix()}')")
    con.execute(f"CREATE TABLE oof_plain AS SELECT * FROM oof WHERE rid % 10000000000 < {COPY_OFFSET}")
    previous = MODEL_DIR / "v5" / "stage2.txt"
    if previous.exists():
        old = lgb.Booster(model_file=str(previous))
        old_p = old.predict(matrix(table, old.feature_name())).astype(np.float32)
        con.register("old_arrow", pa.table(ids | {"p2": old_p}))
        con.execute("CREATE TABLE old AS SELECT * FROM old_arrow")
        con.execute(f"CREATE TABLE old_plain AS SELECT * FROM old WHERE rid % 10000000000 < {COPY_OFFSET}")

    results = []
    for threshold in THRESHOLDS:
        dense, plain = macro_f05(con, threshold), macro_f05(con, threshold, "oof_plain")
        line = f"threshold {threshold:.2f}: test-density F0.5 {dense:.5f}  plain F0.5 {plain:.5f}"
        if previous.exists():
            line += f"  | previous model: test-density {macro_f05(con, threshold, 'old'):.5f}  plain {macro_f05(con, threshold, 'old_plain'):.5f}"
        print(line, flush=True)
        results.append((threshold, dense))
    best_threshold, best_f05 = max(results, key=lambda r: r[1])
    print(f"best threshold {best_threshold:.2f}: test-density out-of-fold macro F0.5 {best_f05:.5f}")

    model = lgb.train(params, lgb.Dataset(X[trainable], y[trainable], feature_name=names), ROUNDS)
    model.save_model(str(MODEL_FILE))
    META_FILE.write_text(json.dumps({
        "features": names, "base": BASE, "threshold": best_threshold, "oof_macro_f05": best_f05,
    }, indent=2))
    for name, gain in sorted(zip(names, model.feature_importance("gain")), key=lambda r: -r[1])[:10]:
        print(f"  {name:20s} {gain:,.0f}")
    print(f"saved {MODEL_FILE} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
