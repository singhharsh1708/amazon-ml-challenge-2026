"""Score the test candidates and write the submission files.

Steps: stage 1 scores for all test pairs (models/matcher.txt), stage 2 scores plus the France
guard model scores (models/stage2.txt, models/v6/stage2.txt), the stacked cross-encoder scores
when data/cross_encoder/test_stacked.parquet exists (France capped by the smaller value), then the
decision stage: best candidate per record, decoy and swap vetoes, expected-F0.5 selection per
source 1 entity, the house-number sibling rule, exact-address additions, France rules and rescue
additions. Finally the challenge validator is run on the outputs.

Input: data/features/test.parquet, data/features/test_odd.parquet, data/norm/test_s*.parquet,
data/candidates/test.parquet, trained models.
Output: output/matching_results.tsv and output/candidate_pairs.tsv.
Run from src/: python predict_submission.py [threshold] [rule,rule,...|norules]
  or python predict_submission.py stage1   (stage 1 scores only)
"""

import json
import shutil
import subprocess
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, DATA_ROOT, MODEL_DIR, OUTPUT_DIR, PROJECT_DIR, TEMP_DIR
from address_rules import exact_address_matches
from decoy_families import family_flags
from decoy_veto import build_vocab, load_decoy_words, type_swap_flags, type_vocab, veto_flags
from expected_f import select as select_expected
from france_rules import france_changes
from rescue_candidates import RESCUE_DIR
from train_rescue import META_FILE as RESCUE_META
from train_rescue import MODEL_FILE as RESCUE_MODEL
from train_rescue import rescue_additions
from train_stage2 import BASE, stage2_query

FEAT_FILE = DATA_DIR / "features" / "test.parquet"
PRED_FILE = DATA_DIR / "features" / "test_predictions.parquet"
PRED2_FILE = DATA_DIR / "features" / "test_predictions_stage2.parquet"
CAND_FILE = DATA_DIR / "candidates" / "test.parquet"
ODD_FILE = DATA_DIR / "features" / "test_odd.parquet"
S1_FILE = DATA_DIR / "norm" / "test_s1.parquet"
NORM_GLOB = DATA_DIR / "norm" / "test_s*.parquet"
SWAP_COUNTRIES = set()
SIBLING_MAX_P = 0.95
TYPE_SWAP_COUNTRIES = {"france"}
RULES = ("sibling", "additions", "expected", "typeswap", "replacement", "shift", "promoted", "france", "rescue")
DEFAULT_RULES = ("sibling", "additions", "expected", "france", "rescue")
EXPECTED_FLOOR = 0.01
EXPECTED_DECOY_WEIGHT = 1.0
MODEL_FILE = MODEL_DIR / "matcher.txt"
META_FILE = MODEL_DIR / "matcher.json"
STAGE2_MODEL = MODEL_DIR / "stage2.txt"
STAGE2_META = MODEL_DIR / "stage2.json"
GUARD_MODEL = MODEL_DIR / "v6" / "stage2.txt"
GUARD_COUNTRIES = ("france",)
STAGE2_PARTS = 8
STACKED_FILE = DATA_DIR / "cross_encoder" / "test_stacked.parquet"
FINAL_FILE = DATA_DIR / "cross_encoder" / "test_predictions_final.parquet"

MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

DB_MEMORY = "4GB"
DB_THREADS = 4

BATCH_ROWS = 2_000_000
CANDIDATE_PARTS = 8
ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"
ID_STR = "'S' || ({col} // 10000000000)::varchar || '-' || ({col} % 10000000000)::varchar"


def predict(model, features):
    """Score all test feature rows with the stage 1 model and write test_predictions.parquet."""
    parquet = pq.ParquetFile(FEAT_FILE)
    writer = None
    rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
            X = np.column_stack([
                batch.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in features
            ])
            p = model.predict(X).astype(np.float32)
            out = pa.table(
                {"rid": batch.column("rid"), "s1": batch.column("s1"), "p": p}
                | {n: batch.column(n) for n in BASE}
            )
            if writer is None:
                writer = pq.ParquetWriter(PRED_FILE, out.schema, compression="zstd")
            writer.write_table(out)
            rows += len(p)
            print(f"scored {rows:,} pairs", flush=True)
    finally:
        if writer is not None:
            writer.close()


def predict_stage2(model, features, guard=None):
    """Score all test pairs with the stage 2 model (and optional guard) and write the stage 2 file."""
    TEMP = TEMP_DIR / "predict"
    TEMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    con.execute(f"SET temp_directory = '{TEMP.as_posix()}'")
    con.execute("SET max_temp_directory_size = '15GB'")
    writer = None
    rows = 0
    try:
        src = f"(SELECT *, NULL::int AS label FROM read_parquet('{PRED_FILE.as_posix()}'))"
        query = stage2_query(con, src)
        for k in range(STAGE2_PARTS):
            odd = "(SELECT NULL::bigint AS rid, NULL::bigint AS s1 WHERE false)"
            if ODD_FILE.exists():
                odd = f"(SELECT * FROM read_parquet('{ODD_FILE.as_posix()}') WHERE s1 % {STAGE2_PARTS} = {k})"
            reader = con.execute(f"""
                SELECT q.*, o.* EXCLUDE (rid, s1)
                FROM (SELECT * FROM ({query}) WHERE s1 % {STAGE2_PARTS} = {k}) q LEFT JOIN {odd} o USING (rid, s1)
            """).fetch_record_batch(rows_per_batch=BATCH_ROWS)
            for batch in reader:
                cols = {"rid": batch.column("rid"), "s1": batch.column("s1")}
                cols["p"] = model.predict(stage2_matrix(batch, features)).astype(np.float32)
                if guard is not None:
                    cols["p_guard"] = guard.predict(stage2_matrix(batch, guard.feature_name())).astype(np.float32)
                out = pa.table(cols)
                if writer is None:
                    writer = pq.ParquetWriter(PRED2_FILE, out.schema, compression="zstd")
                writer.write_table(out)
                rows += out.num_rows
                print(f"stage 2 scored {rows:,} pairs", flush=True)
    finally:
        if writer is not None:
            writer.close()
        con.close()
        shutil.rmtree(TEMP, ignore_errors=True)


def stage2_matrix(batch, features):
    """Stack the given columns of a record batch into a float32 matrix."""
    return np.column_stack([batch.column(n).to_numpy(zero_copy_only=False).astype(np.float32) for n in features])


def stage1_predictions():
    """Ensure stage 1 test predictions exist and return the stage 1 metadata."""
    meta = json.loads(META_FILE.read_text())
    if not PRED_FILE.exists():
        predict(lgb.Booster(model_file=str(MODEL_FILE)), meta["features"])
    return meta


def stage_predictions():
    """Ensure the latest stage predictions exist; returns (prediction file, metadata)."""
    meta = stage1_predictions()
    if not STAGE2_MODEL.exists():
        return PRED_FILE, meta
    meta = json.loads(STAGE2_META.read_text())
    if not PRED2_FILE.exists():
        guard = lgb.Booster(model_file=str(GUARD_MODEL)) if GUARD_MODEL.exists() else None
        predict_stage2(lgb.Booster(model_file=str(STAGE2_MODEL)), meta["features"], guard)
    return PRED2_FILE, meta


def stacked_predictions(con, pred_file, stacked_file, out_file):
    """Replace band scores by stacked scores (France keeps the smaller value) and write the final file."""
    guard = ", n.p_guard" if "p_guard" in pq.ParquetFile(pred_file).schema_arrow.names else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE capped AS
        SELECT {ID_EXPR.format(col='entity_id')} AS rid FROM read_parquet('{NORM_GLOB.as_posix()}')
        WHERE country IN ({', '.join(repr(c) for c in GUARD_COUNTRIES)})
    """)
    con.execute(f"""
        COPY (
            SELECT n.rid, n.s1, (CASE
                WHEN c.rid IS NOT NULL THEN least(n.p, coalesce(st.ps, n.p))
                ELSE coalesce(st.ps, n.p)
            END)::float AS p{guard}
            FROM read_parquet('{pred_file.as_posix()}') n
            LEFT JOIN read_parquet('{stacked_file.as_posix()}') st USING (rid, s1)
            LEFT JOIN capped c ON c.rid = n.rid
        ) TO '{out_file.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    rows, stacked = con.execute(f"""
        SELECT (SELECT count(*) FROM read_parquet('{out_file.as_posix()}')),
            (SELECT count(*) FROM read_parquet('{stacked_file.as_posix()}'))
    """).fetchone()
    print(f"wrote {out_file} ({rows:,} pairs, {stacked:,} with stacked scores)")


def apply_stacker(pred_file):
    """Apply stacked cross-encoder scores to a prediction file and return the final file path."""
    temp = TEMP_DIR / "stacked"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        stacked_predictions(con, pred_file, STACKED_FILE, FINAL_FILE)
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    return FINAL_FILE


def apply_france(con, pred_file, temp, enabled):
    """Apply France removals and additions to the assignment tables; returns (removed, proposed) counts."""
    con.execute("CREATE TABLE france (rid BIGINT, s1 BIGINT)")
    if not enabled:
        return 0, 0
    assigned_file = temp / "france_assigned.parquet"
    con.execute(f"""
        COPY (SELECT rid, s1 FROM after_sibling UNION ALL SELECT rid, s1 FROM additions)
        TO '{assigned_file.as_posix()}' (FORMAT parquet)
    """)
    removals, proposed = france_changes(con, pred_file, assigned_file)
    assigned_file.unlink(missing_ok=True)
    con.register("france_removals_arrow", pa.table({
        "rid": pa.array(removals["rid"], pa.int64()), "s1": pa.array(removals["s1"], pa.int64()),
    }))
    con.register("france_additions_arrow", pa.table({
        "rid": pa.array(proposed["rid"], pa.int64()), "s1": pa.array(proposed["s1"], pa.int64()),
    }))
    before = con.execute("SELECT (SELECT count(*) FROM after_sibling) + (SELECT count(*) FROM additions)").fetchone()[0]
    for table in ("after_sibling", "additions"):
        con.execute(f"DELETE FROM {table} WHERE (rid, s1) IN (SELECT rid, s1 FROM france_removals_arrow)")
    after = con.execute("SELECT (SELECT count(*) FROM after_sibling) + (SELECT count(*) FROM additions)").fetchone()[0]
    con.execute("""
        INSERT INTO france
        SELECT rid, min(s1) FROM france_additions_arrow
        WHERE rid NOT IN (SELECT rid FROM after_sibling) AND rid NOT IN (SELECT rid FROM additions)
            AND s1 IN (SELECT eid FROM s1)
        GROUP BY rid
    """)
    con.unregister("france_removals_arrow")
    con.unregister("france_additions_arrow")
    return before - after, len(proposed["rid"])


def write_outputs(threshold, pred_file, rules=RULES):
    """Run the decision stage at a threshold with the given rules and write both submission files."""
    temp = TEMP_DIR / "predict"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DB_MEMORY}'")
    con.execute(f"SET threads = {DB_THREADS}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    con.execute(f"""
        CREATE TABLE s1 AS
        SELECT entity_id, {ID_EXPR.format(col='entity_id')} AS eid FROM read_parquet('{S1_FILE.as_posix()}')
    """)
    con.execute(f"""
        CREATE TABLE names AS
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, country, name_full, name_core, address
        FROM read_parquet('{NORM_GLOB.as_posix()}')
    """)
    guarded = "p_guard" in pq.ParquetFile(pred_file).schema_arrow.names
    guard_cols = (", arg_max(s1, (p_guard, -s1)) FILTER (WHERE p_guard IS NOT NULL) AS g_s1, max(p_guard) AS g_p"
                  if guarded else "")
    guard_p = "CASE WHEN b.s1 = b.g_s1 THEN least(b.p, b.g_p) ELSE 0 END" if guarded else "b.p"
    floor = min(threshold, EXPECTED_FLOOR) if "expected" in rules else threshold
    con.execute(f"""
        CREATE TABLE best AS
        SELECT * FROM (
            SELECT b.rid, b.s1, CASE WHEN q.country IN ({', '.join(repr(c) for c in GUARD_COUNTRIES)}) THEN {guard_p} ELSE b.p END AS p
            FROM (
                SELECT rid, arg_max(s1, (p, -s1)) FILTER (WHERE p IS NOT NULL) AS s1, max(p) AS p{guard_cols}
                FROM read_parquet('{pred_file.as_posix()}') GROUP BY rid
            ) b JOIN names q ON q.eid = b.rid
        ) WHERE p >= {floor}
    """)
    pairs = con.execute("""
        SELECT b.rid, b.s1, b.p, q.country, q.name_full AS q_full, s.name_full AS s_full,
            q.name_core AS q_core, s.name_core AS s_core
        FROM best b JOIN names q ON q.eid = b.rid JOIN names s ON s.eid = b.s1
    """).fetchnumpy()
    s1_names = con.execute(
        "SELECT country, name_full, name_core, address FROM names WHERE eid // 10000000000 = 1"
    ).fetchnumpy()
    decoy_hit, swap_hit = veto_flags(
        pairs["country"], pairs["q_full"], pairs["s_full"], pairs["p"],
        build_vocab(s1_names["country"], s1_names["name_full"]), load_decoy_words(), SWAP_COUNTRIES,
    )
    type_hit = np.zeros(len(decoy_hit), dtype=bool)
    if "typeswap" in rules:
        type_hit = type_swap_flags(
            pairs["country"], pairs["q_core"], pairs["s_core"],
            type_vocab(s1_names["country"], s1_names["name_core"], s1_names["address"]), TYPE_SWAP_COUNTRIES,
        ) & ~decoy_hit
    keep = ~(decoy_hit | swap_hit | type_hit)
    guard_country = np.isin(pairs["country"], GUARD_COUNTRIES)
    if "expected" in rules:
        free = keep & ~guard_country
        chosen = np.zeros(len(keep), dtype=bool)
        chosen[free] = select_expected(pairs["s1"][free], pairs["p"][free], EXPECTED_DECOY_WEIGHT, pairs["rid"][free])
        keep &= np.where(guard_country, pairs["p"] >= threshold, chosen)
        print(f"expected-F0.5 selection kept {int(chosen.sum()):,} of {int(free.sum()):,} non-guarded pairs")
    con.register("kept_arrow", pa.table({"rid": pairs["rid"][keep], "s1": pairs["s1"][keep], "p": pairs["p"][keep]}))
    con.execute(f"""
        CREATE TABLE kept AS
        SELECT k.rid, k.s1, k.p, f.first_num_equal AS fne
        FROM kept_arrow k LEFT JOIN read_parquet('{PRED_FILE.as_posix()}') f USING (rid, s1)
    """)
    con.execute(f"""
        CREATE TABLE sibling AS
        SELECT k.rid FROM kept k
        WHERE {'true' if 'sibling' in rules else 'false'} AND k.p < {SIBLING_MAX_P} AND k.fne = 0 AND EXISTS (
            SELECT 1 FROM kept x
            WHERE x.s1 = k.s1 AND x.rid <> k.rid AND x.rid // 10000000000 = k.rid // 10000000000 AND x.fne = 1
        )
    """)
    con.execute("CREATE TABLE after_sibling AS SELECT * FROM kept WHERE rid NOT IN (SELECT rid FROM sibling)")
    family_drop = 0
    if {"replacement", "shift", "promoted"} & set(rules):
        fam = con.execute(f"""
            WITH k AS (
                SELECT a.rid, a.s1, f.p AS p1, q.country, q.name_full AS qf, s.name_full AS sf,
                    q.address AS qa, s.address AS sa,
                    nullif(regexp_extract(q.address, '\\b([0-9]+)\\b', 1), '') AS qn1
                FROM after_sibling a
                JOIN read_parquet('{PRED_FILE.as_posix()}') f USING (rid, s1)
                JOIN names q ON q.eid = a.rid JOIN names s ON s.eid = a.s1
            ),
            g AS (SELECT s1, qn1, count(*) AS n FROM k WHERE qn1 IS NOT NULL GROUP BY s1, qn1)
            SELECT k.rid, k.p1, k.country, k.qf, k.sf, k.qa, k.sa,
                CASE WHEN k.qn1 IS NULL THEN 0 ELSE g.n - 1 END AS sib_q
            FROM k LEFT JOIN g ON g.s1 = k.s1 AND g.qn1 = k.qn1
        """).fetchnumpy()
        shift, replacement, promoted = family_flags(
            fam["country"], fam["qf"], fam["sf"], fam["qa"], fam["sa"], fam["p1"], fam["sib_q"],
        )
        dropped = np.zeros(len(shift), dtype=bool)
        for name, flags in (("shift", shift), ("replacement", replacement), ("promoted", promoted)):
            if name in rules:
                dropped |= flags
        family_drop = int(dropped.sum())
        con.register("family_drop_arrow", pa.table({"rid": fam["rid"][dropped]}))
        con.execute("DELETE FROM after_sibling WHERE rid IN (SELECT rid FROM family_drop_arrow)")
        print(f"decoy families: number shift {int(shift.sum()):,}, replacement word {int(replacement.sum()):,}, "
              f"stage-1 promoted {int(promoted.sum()):,}, dropped {family_drop:,}")
    additions = exact_address_matches(con, "test", CAND_FILE)
    con.register("additions_arrow", pa.table({"rid": additions["rid"], "s1": additions["s1"]}))
    con.execute(f"""
        CREATE TABLE additions AS
        SELECT rid, s1 FROM additions_arrow WHERE {'true' if 'additions' in rules else 'false'} AND rid NOT IN (SELECT rid FROM kept)
    """)
    france_removed, france_proposed = apply_france(con, pred_file, temp, "france" in rules)
    rescued = rescue_additions(con, RESCUE_MODEL, RESCUE_META, RESCUE_DIR) if "rescue" in rules else {"rid": [], "s1": []}
    con.register("rescue_arrow", pa.table({
        "rid": pa.array(rescued["rid"], pa.int64()), "s1": pa.array(rescued["s1"], pa.int64()),
    }))
    con.execute("""
        CREATE TABLE rescue AS
        SELECT rid, s1 FROM rescue_arrow
        WHERE rid NOT IN (SELECT rid FROM after_sibling) AND rid NOT IN (SELECT rid FROM additions)
            AND rid NOT IN (SELECT rid FROM france)
    """)
    con.execute("""
        CREATE TABLE assigned AS
        SELECT s1, rid FROM after_sibling
        UNION ALL
        SELECT s1, rid FROM additions
        UNION ALL
        SELECT s1, rid FROM france
        UNION ALL
        SELECT s1, rid FROM rescue
    """)
    if "france" in rules:
        n_france = con.execute("SELECT count(*) FROM france").fetchone()[0]
        print(f"france rules removed {france_removed:,} assignments and added {n_france:,} of {france_proposed:,} "
              "proposed pairs for unassigned records")
    if "rescue" in rules:
        print(f"rescue added {con.execute('SELECT count(*) FROM rescue').fetchone()[0]:,} of {len(rescued['rid']):,} "
              "accepted pairs for unassigned records")
    n_sibling, n_added = con.execute("SELECT (SELECT count(*) FROM sibling), (SELECT count(*) FROM additions)").fetchone()
    print(f"vetoed {int(decoy_hit.sum()):,} decoy-word, {int(type_hit.sum()):,} type-word-swap and {int(swap_hit.sum()):,} word-swap assignments of {len(keep):,}; "
          f"dropped {n_sibling:,} by the house-number sibling rule; added {n_added:,} exact-address matches")
    con.execute(f"""
        COPY (
            SELECT s.entity_id AS source1_entity_id,
                coalesce(string_agg({ID_STR.format(col='a.rid')}, ',' ORDER BY a.rid), '') AS matched_entity_ids
            FROM s1 s LEFT JOIN assigned a ON a.s1 = s.eid
            GROUP BY s.entity_id ORDER BY s.entity_id
        ) TO '{MATCHING_FILE.as_posix()}' (HEADER, DELIMITER '\t', QUOTE '')
    """)
    parts = []
    for part in range(CANDIDATE_PARTS):
        part_file = CANDIDATE_FILE.parent / f"candidate_pairs.part{part}.tsv"
        con.execute(f"""
            COPY (
                SELECT s.entity_id AS source1_entity_id,
                    coalesce(string_agg({ID_STR.format(col='c.rid')}, ',' ORDER BY c.rid), '') AS candidate_entity_ids
                FROM (SELECT * FROM s1 WHERE eid % {CANDIDATE_PARTS} = {part}) s
                LEFT JOIN (
                    SELECT s1, rid FROM read_parquet('{CAND_FILE.as_posix()}') WHERE s1 % {CANDIDATE_PARTS} = {part}
                    UNION
                    SELECT s1, rid FROM additions WHERE s1 % {CANDIDATE_PARTS} = {part}
                    UNION
                    SELECT s1, rid FROM france WHERE s1 % {CANDIDATE_PARTS} = {part}
                    UNION
                    SELECT s1, rid FROM rescue WHERE s1 % {CANDIDATE_PARTS} = {part}
                ) c ON c.s1 = s.eid
                GROUP BY s.entity_id ORDER BY s.entity_id
            ) TO '{part_file.as_posix()}' (HEADER {'true' if part == 0 else 'false'}, DELIMITER '\t', QUOTE '')
        """)
        parts.append(part_file)
    with open(CANDIDATE_FILE, "wb") as out:
        for part_file in parts:
            with open(part_file, "rb") as src:
                shutil.copyfileobj(src, out)
            part_file.unlink()
    stats = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE coalesce(matched_entity_ids, '') = ''),
            sum(CASE WHEN coalesce(matched_entity_ids, '') = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END)
        FROM read_csv('{MATCHING_FILE.as_posix()}', delim = '\t', header = true, all_varchar = true, quote = '')
    """).fetchone()
    print(f"matching_results: {stats[0]:,} entities, {stats[1]:,} empty, {stats[2]:,} matches")
    con.close()
    shutil.rmtree(temp, ignore_errors=True)


def main():
    """Command line entry: build predictions, write outputs and run the challenge validator."""
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["stage1"]:
        stage1_predictions()
        print(f"stage 1 test predictions in {PRED_FILE}, done in {time.time() - start:.0f}s")
        return
    pred_file, meta = stage_predictions()
    if STACKED_FILE.exists():
        pred_file = apply_stacker(pred_file)
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else meta["threshold"]
    arg = sys.argv[2] if len(sys.argv) > 2 else ",".join(DEFAULT_RULES)
    rules = () if arg == "norules" else tuple(r for r in arg.split(",") if r)
    unknown = set(rules) - set(RULES)
    if unknown:
        raise SystemExit(f"unknown rules {sorted(unknown)}; choose from {RULES} or norules")
    write_outputs(threshold, pred_file, rules)
    print(f"rules: {', '.join(rules) or 'none'}")
    print(f"{pred_file.name} at threshold {threshold}, done in {time.time() - start:.0f}s")
    result = subprocess.run(
        [
            sys.executable, str(DATA_ROOT / "utils" / "validate_submission.py"),
            "--matching", str(MATCHING_FILE),
            "--candidate", str(CANDIDATE_FILE),
            "--test-dir", str(DATA_ROOT / "dataset" / "test"),
        ],
        cwd=PROJECT_DIR,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
