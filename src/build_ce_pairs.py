"""Build the text pair files used to train and apply the cross-encoders.

train: stage 1 pairs of the train features whose stage 1 probability is in the uncertain band
TRAIN_BAND, plus a small random share of the rest. valid: validation-slice pairs with
out-of-fold stage 2 probability inside VALID_P2. test: the top TEST_RANK candidates per test
record with stage 2 probability inside TEST_P. Each row carries rid, s1, the scores and the
query and source 1 texts.

Input: data/features/train.parquet, models/matcher.*, data/features/valid_stage2_oof.parquet,
stage 2 test predictions, data/norm.
Output: data/cross_encoder/{train_pairs,valid_band,test_band}.parquet.
Run from src/: python build_ce_pairs.py [train] [valid] [test]
"""

import json
import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, MODEL_DIR, TEMP_DIR
from predict_submission import stage_predictions
from train_stage2 import OOF_FILE, VALID_MOD

CE_DIR = DATA_DIR / "cross_encoder"
NORM_DIR = DATA_DIR / "norm"
TRAIN_FEAT_FILE = DATA_DIR / "features" / "train.parquet"
MATCHER_FILE = MODEL_DIR / "matcher.txt"
MATCHER_META = MODEL_DIR / "matcher.json"
TRAIN_PAIRS = CE_DIR / "train_pairs.parquet"
VALID_BAND = CE_DIR / "valid_band.parquet"
TEST_BAND = CE_DIR / "test_band.parquet"

TRAIN_BAND = (0.003, 0.997)
TRAIN_EXTRA = 0.015
TRAIN_SEED = 7
VALID_P2 = (0.002, 0.998)
TEST_P = (0.01, 0.99)
TEST_RANK = 2
BATCH_ROWS = 500_000
ID_EXPR = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
TEXT = "coalesce({a}.name_full, '') || ' | ' || coalesce({a}.address, '')"


def with_text(split, pairs, order):
    """Return SQL that attaches query text, source 1 text and country to a pair query."""
    return f"""
        SELECT p.*, {TEXT.format(a='q')} AS q_text, {TEXT.format(a='s')} AS s_text, q.country
        FROM ({pairs}) p
        JOIN (
            SELECT {ID_EXPR} AS eid, name_full, address, country
            FROM read_parquet('{(NORM_DIR / f'{split}_s[23].parquet').as_posix()}')
        ) q ON q.eid = p.rid
        JOIN (
            SELECT {ID_EXPR} AS eid, name_full, address
            FROM read_parquet('{(NORM_DIR / f'{split}_s1.parquet').as_posix()}')
        ) s ON s.eid = p.s1
        ORDER BY {order}
    """


def write(con, split, pairs, order, out_file):
    """Write a text-joined pair query to parquet and print its size."""
    con.execute(f"COPY ({with_text(split, pairs, order)}) TO '{out_file.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    rows, rids = con.execute(f"SELECT count(*), count(DISTINCT rid) FROM read_parquet('{out_file.as_posix()}')").fetchone()
    print(f"wrote {out_file} ({rows:,} pairs, {rids:,} records)", flush=True)


def sample_train(batches, model, features, rng):
    """Yield stage 1 scored training batches filtered to the uncertain band plus a random extra share."""
    for batch in batches:
        X = np.column_stack([batch.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in features])
        p = model.predict(X).astype(np.float32)
        keep = ((p >= TRAIN_BAND[0]) & (p <= TRAIN_BAND[1])) | (rng.random(len(p)) < TRAIN_EXTRA)
        mask = pa.array(keep)
        yield pa.table({
            "rid": batch.column("rid").filter(mask), "s1": batch.column("s1").filter(mask),
            "label": batch.column("label").filter(mask), "p1": p[keep],
        })


def train_pairs(con, out_file=TRAIN_PAIRS):
    """Build the cross-encoder training pair file from the stage 1 train features."""
    features = json.loads(MATCHER_META.read_text())["features"]
    model = lgb.Booster(model_file=str(MATCHER_FILE))
    batches = pq.ParquetFile(TRAIN_FEAT_FILE).iter_batches(batch_size=BATCH_ROWS, columns=["rid", "s1", "label"] + features)
    con.register("sampled", pa.concat_tables(sample_train(batches, model, features, np.random.default_rng(TRAIN_SEED))))
    write(con, "train", "SELECT * FROM sampled", "hash(p.rid, p.s1)", out_file)
    con.unregister("sampled")


def valid_band(con, oof_file=OOF_FILE, out_file=VALID_BAND):
    """Build the validation band file from out-of-fold stage 2 scores of the validation slice."""
    pairs = f"""
        SELECT rid, s1, p2 FROM read_parquet('{oof_file.as_posix()}')
        WHERE hash(s1) % {VALID_MOD} = 0 AND p2 BETWEEN {VALID_P2[0]} AND {VALID_P2[1]}
    """
    write(con, "train", pairs, "p.rid, p.s1", out_file)


def test_band(con, pred_file, out_file=TEST_BAND):
    """Build the test band file from stage 2 test predictions."""
    pairs = f"""
        SELECT rid, s1, p FROM (
            SELECT rid, s1, p, row_number() OVER (PARTITION BY rid ORDER BY p DESC, s1) AS rk
            FROM read_parquet('{pred_file.as_posix()}')
        ) WHERE rk <= {TEST_RANK} AND p BETWEEN {TEST_P[0]} AND {TEST_P[1]}
    """
    write(con, "test", pairs, "p.rid, p.s1", out_file)


def main():
    """Build the requested pair files (train, valid, test)."""
    start = time.time()
    parts = sys.argv[1:] or ["train", "valid", "test"]
    unknown = set(parts) - {"train", "valid", "test"}
    if unknown:
        raise SystemExit(f"unknown parts {sorted(unknown)}; choose from train, valid, test")
    pred_file = stage_predictions()[0] if "test" in parts else None
    CE_DIR.mkdir(parents=True, exist_ok=True)
    temp = TEMP_DIR / "ce_pairs"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'; SET threads = 4; SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        if "train" in parts:
            train_pairs(con)
        if "valid" in parts:
            valid_band(con)
        if "test" in parts:
            test_band(con, pred_file)
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
