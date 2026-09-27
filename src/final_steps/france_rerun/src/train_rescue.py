"""Copy of a src/ module as used by the France re-run (fixed France normalization). Experiment/validation script train_rescue.py for this step.

Step context: Copy of a src/ module as used by the France re-run (fixed France normalization). It is put ahead of src/ on sys.path by the France re-run scripts.
"""
import json
import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score

from config import MODEL_DIR, TEMP_DIR
from rescue_candidates import (
    CAND_DIR, ID_EXPR, NORM_DIR, OUT_FILES, RESCUE_DIR, VALID_ASSIGNED, VALID_SCORES, truth_sql,
)
from train_stage2 import VALID_MOD

MODEL_FILE = MODEL_DIR / "rescue.txt"
META_FILE = MODEL_DIR / "rescue.json"
DEFAULT_TAGS = ("ce1", "ce2")
FAMILIES = ["aexact", "corenum", "cat", "typo", "anum", "aalpha"]
EXCLUDED_COUNTRIES = ("france",)
THRESHOLD = 0.7
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
DECOY_WEIGHT = 1.887

FOLDS = 2
ROUNDS = 300
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
    "verbose": -1, "num_threads": 2, "seed": 3,
}


def ce_files(split, tags, rescue_dir=RESCUE_DIR):
    """Return the cross-encoder score files of a split."""
    band = "valid" if split == "train" else split
    return [rescue_dir / f"{band}_rescue_{tag}.parquet" for tag in tags]


def feature_names(tags):
    """Return the rescue gate feature names."""
    return ([f"ce_{tag}" for tag in tags] + [f"fam_{fam}" for fam in FAMILIES]
            + ["n_families", "name_token_set", "name_ratio", "address_token_set", "address_missing"])


def split_text(texts):
    """Split pair texts into name and address parts."""
    parts = [t.split(" | ", 1) for t in texts]
    return [p[0] for p in parts], [p[1] if len(p) > 1 else "" for p in parts]


def matrix(d, n_ce):
    """Build the rescue gate feature matrix."""
    q_name, q_addr = split_text(d["q_text"])
    s_name, s_addr = split_text(d["s_text"])
    fams = [set(f.split(",")) for f in d["family"]]
    cols = [d[f"ce_{k}"] for k in range(n_ce)]
    cols += [np.array([fam in f for f in fams], np.float32) for fam in FAMILIES]
    cols += [
        np.array([len(f) for f in fams], np.float32),
        np.array([fuzz.token_set_ratio(a, b) for a, b in zip(q_name, s_name)], np.float32),
        np.array([fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")) for a, b in zip(q_name, s_name)], np.float32),
        np.array([fuzz.token_set_ratio(a, b) if a else -1 for a, b in zip(q_addr, s_addr)], np.float32),
        np.array([float(not a) for a in q_addr], np.float32),
    ]
    return np.column_stack(cols).astype(np.float32)


def ce_sql(files):
    """Return the SQL columns and joins for the cross-encoder scores."""
    cols = "".join(f", e{k}.ce AS ce_{k}" for k in range(len(files)))
    joins = " ".join(f"JOIN read_parquet('{f.as_posix()}') e{k} USING (rid, s1)" for k, f in enumerate(files))
    return cols, joins


def accepted_pairs(con, model, rescue_file, files, threshold, excluded=EXCLUDED_COUNTRIES):
    """Return the rescue pairs the gate accepts above threshold."""
    cols, joins = ce_sql(files)
    countries = ", ".join(repr(c) for c in excluded)
    d = con.execute(f"""
        SELECT r.rid, r.s1, r.file_row_number AS pos, r.family, r.q_text, r.s_text, r.country{cols}
        FROM read_parquet('{rescue_file.as_posix()}', file_row_number = true) r {joins}
        WHERE r.country NOT IN ({countries})
        ORDER BY r.file_row_number
    """).fetchnumpy()
    ps = model.predict(matrix(d, len(files)))
    con.register("rescue_scored", pa.table({"rid": d["rid"], "s1": d["s1"], "pos": d["pos"], "ps": ps}))
    out = con.execute(f"""
        SELECT rid, s1 FROM (
            SELECT rid, s1, ps, row_number() OVER (PARTITION BY rid ORDER BY ps DESC, s1, pos) AS rk FROM rescue_scored
        ) WHERE rk = 1 AND ps >= {threshold}
        ORDER BY rid
    """).fetchnumpy()
    con.unregister("rescue_scored")
    print(f"rescue scored {len(ps):,} pairs outside {', '.join(excluded)}, accepted {len(out['rid']):,} at {threshold}",
          flush=True)
    return out


def rescue_additions(con, model_file=MODEL_FILE, meta_file=META_FILE, rescue_dir=RESCUE_DIR):
    """Return the rescue additions for the test split."""
    empty = {"rid": np.zeros(0, dtype=np.int64), "s1": np.zeros(0, dtype=np.int64)}
    if not model_file.exists() or not meta_file.exists():
        return empty
    meta = json.loads(meta_file.read_text())
    rescue_file = rescue_dir / OUT_FILES["test"].name
    files = ce_files("test", meta["ce_tags"], rescue_dir)
    if not rescue_file.exists() or not all(f.exists() for f in files):
        return empty
    model = lgb.Booster(model_file=str(model_file))
    return accepted_pairs(con, model, rescue_file, files, meta["threshold"], tuple(meta["excluded_countries"]))


def training_pairs(con, rescue_file, files, scores_file, cand_file):
    """Return the labelled rescue pairs for training the gate."""
    cols, joins = ce_sql(files)
    con.execute(f"CREATE OR REPLACE TABLE touch AS SELECT DISTINCT rid FROM read_parquet('{scores_file.as_posix()}')")
    con.execute(f"""
        CREATE OR REPLACE TABLE cand AS
        SELECT rid, s1 FROM read_parquet('{cand_file.as_posix()}')
        WHERE rid IN (SELECT rid FROM read_parquet('{rescue_file.as_posix()}'))
    """)
    return con.execute(f"""
        SELECT * FROM (
            SELECT r.rid, r.s1, r.family, r.label, r.q_text, r.s_text, r.country{cols}, t.s1 AS ts,
                (NOT r.assigned AND NOT (r.rid NOT IN (SELECT rid FROM touch)
                    AND EXISTS (SELECT 1 FROM cand k WHERE k.rid = r.rid AND k.s1 = t.s1))) AS eligible
            FROM read_parquet('{rescue_file.as_posix()}') r {joins}
            LEFT JOIN truth t ON t.rid = r.rid
        ) WHERE eligible
        ORDER BY rid, s1
    """).fetchnumpy()


def validation_scope(con, assigned_file):
    """Create the validation scope table from an assignment."""
    con.execute(f"""
        CREATE OR REPLACE TABLE valid_s1 AS
        SELECT s.s1, count(t.rid) AS n_true
        FROM (
            SELECT {ID_EXPR.format(col='entity_id')} AS s1
            FROM read_parquet('{(NORM_DIR / 'train_s1.parquet').as_posix()}')
        ) s LEFT JOIN truth t USING (s1)
        WHERE hash(s.s1) % {VALID_MOD} = 0
        GROUP BY s.s1
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE valid_assigned AS
        SELECT a.rid, a.s1, t.s1 AS ts FROM read_parquet('{assigned_file.as_posix()}') a LEFT JOIN truth t USING (rid)
    """)


def macro_f05(con, extra):
    """Return held-out macro F0.5 with the extra pairs added."""
    return con.execute(f"""
        WITH picked AS (
            SELECT rid, s1, ts FROM valid_assigned
            UNION ALL
            SELECT rid, s1, ts FROM rescue_best WHERE {extra}
        ),
        per AS (
            SELECT v.s1, v.n_true,
                count(a.rid) FILTER (WHERE a.ts = a.s1) AS tp,
                count(a.rid) FILTER (WHERE a.ts IS NULL) AS fp_decoy,
                count(a.rid) FILTER (WHERE a.ts <> a.s1) AS fp_wrong
            FROM valid_s1 v LEFT JOIN picked a USING (s1)
            GROUP BY v.s1, v.n_true
        )
        SELECT
            avg(CASE
                WHEN n_true = 0 THEN greatest(0, 1 - fp_wrong - {DECOY_WEIGHT} * fp_decoy)
                WHEN tp = 0 THEN 0.0
                ELSE 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp) + fp_wrong + {DECOY_WEIGHT} * fp_decoy)
            END),
            avg(CASE
                WHEN n_true = 0 AND tp + fp_wrong + fp_decoy = 0 THEN 1.0
                WHEN tp = 0 THEN 0.0
                ELSE 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp) + fp_wrong + fp_decoy)
            END)
        FROM per
    """).fetchone()


def evaluate(con, d, oof, assigned_file):
    """Report held-out F0.5 for out-of-fold rescue predictions."""
    validation_scope(con, assigned_file)
    con.register("rescue_oof", pa.table({"rid": d["rid"], "s1": d["s1"], "ps": oof}))
    con.execute("""
        CREATE OR REPLACE TABLE rescue_best AS
        SELECT b.rid, b.s1, b.ps, t.s1 AS ts
        FROM (SELECT rid, arg_max(s1, ps) AS s1, max(ps) AS ps FROM rescue_oof GROUP BY rid) b
        LEFT JOIN truth t USING (rid)
    """)
    con.unregister("rescue_oof")
    base = macro_f05(con, "false")
    print(f"assigned only: decoy-weighted F0.5 {base[0]:.5f}  plain F0.5 {base[1]:.5f}", flush=True)
    for threshold in THRESHOLDS:
        n, tp = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE ts = s1) FROM rescue_best WHERE ps >= {threshold}"
        ).fetchone()
        f = macro_f05(con, f"ps >= {threshold}")
        print(f"threshold {threshold:.2f}: add {n:,} ({tp:,} true)  decoy-weighted F0.5 {f[0]:.5f} ({f[0] - base[0]:+.5f})"
              f"  plain F0.5 {f[1]:.5f} ({f[1] - base[1]:+.5f})", flush=True)


def fit(X, y, names):
    """Train the LightGBM rescue gate."""
    return lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=names), ROUNDS)


def run(con, tags, rescue_file=OUT_FILES["train"], rescue_dir=RESCUE_DIR, assigned_file=VALID_ASSIGNED,
        scores_file=VALID_SCORES, cand_file=CAND_DIR / "train.parquet", model_file=MODEL_FILE, meta_file=META_FILE):
    """Run one configuration and report or write its result."""
    names = feature_names(tags)
    files = ce_files("train", tags, rescue_dir)
    con.execute(f"CREATE OR REPLACE TABLE truth AS {truth_sql()}")
    d = training_pairs(con, rescue_file, files, scores_file, cand_file)
    X = matrix(d, len(tags))
    y = d["label"].astype(np.int8)
    print(f"eligible rescue pairs {len(y):,}, true {int(y.sum()):,}", flush=True)
    fold = d["s1"] % FOLDS
    oof = np.zeros(len(y))
    for k in range(FOLDS):
        held = fold == k
        oof[held] = fit(X[~held], y[~held], names).predict(X[held])
    aucs = [f"{tag} {roc_auc_score(y, d[f'ce_{k}']):.4f}" for k, tag in enumerate(tags)]
    print(f"AUC {', '.join(aucs)}, rescue model out-of-fold {roc_auc_score(y, oof):.4f}", flush=True)
    evaluate(con, d, oof, assigned_file)
    model = fit(X, y, names)
    model.save_model(str(model_file))
    meta_file.write_text(json.dumps({
        "ce_tags": list(tags), "features": names, "threshold": THRESHOLD,
        "excluded_countries": list(EXCLUDED_COUNTRIES),
    }, indent=2))
    return model


def main():
    """Parse the command line and run the step."""
    start = time.time()
    tags = [t for arg in sys.argv[1:] for t in arg.split(",") if t] or list(DEFAULT_TAGS)
    temp = TEMP_DIR / "rescue_train"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '1200MB'; SET threads = 2; SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        run(con, tags)
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    print(f"saved {MODEL_FILE} with cross-encoders {', '.join(tags)} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
