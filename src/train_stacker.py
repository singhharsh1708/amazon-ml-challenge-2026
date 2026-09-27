"""Stack stage 2 and cross-encoder scores with a small LightGBM model.

Features: logit of the stage 2 probability, the cross-encoder logits, logit of the stage 1
probability, address_missing, source and blocking rank. The stacker is evaluated with 2-fold
out-of-fold AUC on the validation band, refit on all of it, and applied to the test band.

Input: data/cross_encoder/{valid,test}_band.parquet, {valid,test}_{tag}.parquet for every tag,
data/features/valid*.parquet and test_predictions.parquet.
Output: models/stacker.txt and data/cross_encoder/test_stacked.parquet (rid, s1, ps).
Run from src/: python train_stacker.py [tag ...]   (default ce1 ce2 kbase)
"""

import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, MODEL_DIR, TEMP_DIR

CE_DIR = DATA_DIR / "cross_encoder"
FEAT_DIR = DATA_DIR / "features"
VALID_BAND = CE_DIR / "valid_band.parquet"
TEST_BAND = CE_DIR / "test_band.parquet"
STACKED_FILE = CE_DIR / "test_stacked.parquet"
MODEL_FILE = MODEL_DIR / "stacker.txt"
DEFAULT_TAGS = ("ce1", "ce2", "kbase")

FOLDS = 2
ROUNDS = 300
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100,
    "verbose": -1, "num_threads": 4, "seed": 1,
}


def logit(p):
    """Return the clipped log-odds of a probability array."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feature_names(n_ce):
    """Return stacker feature names for a number of cross-encoders."""
    return ["logit_p2"] + [f"ce_{k}" for k in range(n_ce)] + ["logit_p1", "address_missing", "source", "rk"]


def matrix(d, n_ce):
    """Build the stacker feature matrix from a query result dict."""
    cols = [logit(d["p2"])] + [d[f"ce_{k}"] for k in range(n_ce)] + [logit(d["p1"]), d["am"], d["source"], d["rk"]]
    return np.column_stack(cols).astype(np.float32)


def ce_sql(ce_files):
    """Return the column list and joins that attach each cross-encoder score file."""
    cols = "".join(f", e{k}.ce AS ce_{k}" for k in range(len(ce_files)))
    joins = " ".join(f"JOIN read_parquet('{f.as_posix()}') e{k} USING (rid, s1)" for k, f in enumerate(ce_files))
    return cols, joins


def valid_features(con, band_file, ce_files):
    """Load validation band rows with labels and return (data, feature matrix)."""
    cols, joins = ce_sql(ce_files)
    d = con.execute(f"""
        SELECT b.rid, b.s1, b.p2{cols}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk, f.label AS y
        FROM read_parquet('{band_file.as_posix()}') b {joins}
        JOIN (
            SELECT rid, s1, v.label, v.address_missing, v.source, v.rk, pr.p
            FROM read_parquet('{(FEAT_DIR / 'valid.parquet').as_posix()}') v
            JOIN read_parquet('{(FEAT_DIR / 'valid_predictions.parquet').as_posix()}') pr USING (rid, s1)
        ) f USING (rid, s1)
        ORDER BY b.rid, b.s1
    """).fetchnumpy()
    return d, matrix(d, len(ce_files))


def test_features(con, band_file, ce_files):
    """Load test band rows and return (data, feature matrix)."""
    cols, joins = ce_sql(ce_files)
    d = con.execute(f"""
        SELECT b.rid, b.s1, b.p AS p2{cols}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk
        FROM read_parquet('{band_file.as_posix()}') b {joins}
        JOIN read_parquet('{(FEAT_DIR / 'test_predictions.parquet').as_posix()}') f USING (rid, s1)
        ORDER BY b.rid, b.s1
    """).fetchnumpy()
    return d, matrix(d, len(ce_files))


def fit(X, y, names):
    """Train one LightGBM stacker."""
    return lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=names), ROUNDS)


def run(con, valid_ce, test_ce, valid_band=VALID_BAND, test_band=TEST_BAND, model_file=MODEL_FILE, out_file=STACKED_FILE):
    """Evaluate out of fold, fit the final stacker, score the test band and return the OOF scores."""
    if len(valid_ce) != len(test_ce) or not valid_ce:
        raise ValueError("need the same number of validation and test cross-encoder score files")
    names = feature_names(len(valid_ce))
    d, X = valid_features(con, valid_band, valid_ce)
    y = d["y"]
    fold = d["s1"] % FOLDS
    oof = np.zeros(len(y))
    for k in range(FOLDS):
        held = fold == k
        oof[held] = fit(X[~held], y[~held], names).predict(X[held])
    aucs = [f"p2 {roc_auc_score(y, d['p2']):.4f}"] + [f"ce_{k} {roc_auc_score(y, d[f'ce_{k}']):.4f}" for k in range(len(valid_ce))]
    print(f"validation band {len(y):,} pairs, positive rate {y.mean():.3f}; AUC {', '.join(aucs)}, "
          f"stacked out-of-fold {roc_auc_score(y, oof):.4f}", flush=True)
    model = fit(X, y, names)
    model.save_model(str(model_file))
    t, Xt = test_features(con, test_band, test_ce)
    ps = model.predict(Xt).astype(np.float32)
    pq.write_table(pa.table({"rid": t["rid"], "s1": t["s1"], "ps": ps}), out_file)
    print(f"stacked {len(ps):,} test band pairs, mean p2 {t['p2'].mean():.3f} -> {ps.mean():.3f}; wrote {out_file}")
    return oof


def main():
    """Command line entry: stack the given cross-encoder tags."""
    start = time.time()
    tags = sys.argv[1:] or list(DEFAULT_TAGS)
    temp = TEMP_DIR / "stacker"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'; SET threads = 4; SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        run(con, [CE_DIR / f"valid_{tag}.parquet" for tag in tags], [CE_DIR / f"test_{tag}.parquet" for tag in tags])
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    print(f"saved {MODEL_FILE} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
