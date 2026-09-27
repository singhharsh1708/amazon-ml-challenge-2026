"""Train the stage 1 LightGBM matcher.

Trains a binary LightGBM model on the train part of the pair features (5% held out for early
stopping), scores the validation part, and reports the challenge metric (per-source-1 macro
F0.5, with an empty prediction for an entity with no true match counted as 1) over a grid of
thresholds using the best candidate per record.

Input: data/features/train.parquet, data/features/valid.parquet, ground truth.
Output: models/matcher.txt, models/matcher.json (feature list, threshold, score) and
data/features/valid_predictions.parquet.
Run from src/: python train_matcher.py
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
PRED_FILE = DATA_DIR / "features" / "valid_predictions.parquet"
MODEL_FILE = MODEL_DIR / "matcher.txt"
META_FILE = MODEL_DIR / "matcher.json"

VALID_MOD = 20
NON_FEATURES = {"rid", "s1", "part", "label"}
THRESHOLDS = [round(x, 2) for x in np.arange(0.2, 0.96, 0.05)]

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"


def load(path, features=None):
    """Load a feature parquet into a float32 matrix plus labels, ids and the feature list."""
    parquet = pq.ParquetFile(path)
    if features is None:
        features = [n for n in parquet.schema_arrow.names if n not in NON_FEATURES]
    X = np.empty((parquet.metadata.num_rows, len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = pq.read_table(path, columns=[name]).column(0).to_numpy(zero_copy_only=False)
    y = pq.read_table(path, columns=["label"]).column(0).to_numpy(zero_copy_only=False).astype(np.int8)
    ids = pq.read_table(path, columns=["rid", "s1"])
    return X, y, ids, features


def macro_f05(con, threshold):
    """Return (macro F0.5, precision, recall, entities) on the validation slice for a threshold."""
    return con.execute(f"""
        WITH best AS (
            SELECT rid, arg_max(s1, p) AS s1, max(p) AS p FROM preds GROUP BY rid
        ),
        pred AS (
            SELECT s1, count(*) AS n_pred FROM best
            WHERE p >= {threshold} AND hash(s1) % {VALID_MOD} = 0 GROUP BY s1
        ),
        hit AS (
            SELECT b.s1, count(*) AS tp FROM best b JOIN truth t USING (rid, s1)
            WHERE b.p >= {threshold} AND hash(b.s1) % {VALID_MOD} = 0 GROUP BY b.s1
        ),
        per AS (
            SELECT v.s1, coalesce(n.n_true, 0) AS n_true,
                coalesce(p.n_pred, 0) AS n_pred, coalesce(h.tp, 0) AS tp
            FROM valid_s1 v
            LEFT JOIN (SELECT s1, count(*) AS n_true FROM truth GROUP BY s1) n USING (s1)
            LEFT JOIN pred p USING (s1)
            LEFT JOIN hit h USING (s1)
        )
        SELECT
            avg(CASE
                WHEN n_true = 0 AND n_pred = 0 THEN 1.0
                WHEN tp = 0 THEN 0.0
                ELSE 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp) + (n_pred - tp))
            END),
            sum(tp) / sum(n_pred), sum(tp) / sum(n_true), count(*)
        FROM per
    """).fetchone()


def main():
    """Train, save and evaluate the stage 1 matcher."""
    start = time.time()
    X, y, _, features = load(FEAT_DIR / "train.parquet")
    print(f"train rows {len(y):,}, positives {y.mean():.4f}, features {len(features)}", flush=True)

    rng = np.random.default_rng(42)
    stop_mask = rng.random(len(y)) < 0.05
    train_set = lgb.Dataset(X[~stop_mask], y[~stop_mask], feature_name=features, free_raw_data=True)
    stop_set = lgb.Dataset(X[stop_mask], y[stop_mask], reference=train_set)
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 6,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[stop_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    del X, y, train_set, stop_set
    print(f"trained {model.best_iteration} rounds in {time.time() - start:.0f}s", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE), num_iteration=model.best_iteration)

    writer = None
    for batch in pq.ParquetFile(FEAT_DIR / "valid.parquet").iter_batches(batch_size=2_000_000):
        Xv = np.column_stack([
            batch.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in features
        ])
        p = model.predict(Xv, num_iteration=model.best_iteration).astype(np.float32)
        out = pa.table({"rid": batch.column("rid"), "s1": batch.column("s1"), "p": p})
        if writer is None:
            writer = pq.ParquetWriter(PRED_FILE, out.schema)
        writer.write_table(out)
    writer.close()

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    con.execute(f"CREATE TABLE preds AS SELECT * FROM read_parquet('{PRED_FILE.as_posix()}')")
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

    results = []
    for threshold in THRESHOLDS:
        f05, precision, recall, n = macro_f05(con, threshold)
        results.append((threshold, f05))
        print(f"threshold {threshold:.2f}: macro F0.5 {f05:.5f}  precision {precision:.4f}  recall {recall:.4f}  (entities {n:,})")
    best_threshold, best_f05 = max(results, key=lambda r: r[1])
    print(f"best threshold {best_threshold:.2f}: macro F0.5 {best_f05:.5f}")

    importance = sorted(
        zip(features, model.feature_importance("gain")), key=lambda r: -r[1]
    )
    for name, gain in importance[:15]:
        print(f"  {name:24s} {gain:,.0f}")

    META_FILE.write_text(json.dumps({
        "features": features,
        "threshold": best_threshold,
        "valid_macro_f05": best_f05,
        "best_iteration": model.best_iteration,
    }, indent=2))
    print(f"saved {MODEL_FILE} and {META_FILE} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
