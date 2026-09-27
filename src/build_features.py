"""Pairwise features for every blocked candidate pair (stage 1 input).

For each (source 2/3 record, source 1 candidate) pair this computes blocking statistics, record
and source 1 context counts, token Jaccard, house-number agreement, legal-form clash flags,
rapidfuzz string similarities on names and addresses, and per-record gap/rank versions of the
main similarities. On the training split, pairs are labelled from the ground truth and split into two
parts: records with any candidate in the validation slice (hash(s1) % 20 == 0) form the
valid part and half of the remaining records form the train part; the test split is one part.

Input: data/candidates/{split}.parquet, data/norm/{split}_s*.parquet, ground truth for train.
Output: data/features/{train,valid}.parquet or data/features/test.parquet.
Run from src/: python build_features.py [train|test]
"""

import shutil
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from config import DATA_DIR, TEMP_DIR, TRAIN_DIR

NORM_DIR = DATA_DIR / "norm"
CAND_DIR = DATA_DIR / "candidates"
FEAT_DIR = DATA_DIR / "features"

VALID_MOD = 20
TRAIN_RECORD_MOD = 2
PARTS = 64

RELATIVE = [
    "name_token_set", "name_ratio", "name_jw", "name_full_ratio",
    "address_token_set", "address_ratio", "combined",
]

MEMORY_LIMIT = "4GB"
THREADS = 4
MAX_TEMP = "15GB"

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"

FUZZY = [
    ("name_ratio", "q_name", "s_name", fuzz.ratio),
    ("name_token_sort", "q_name", "s_name", fuzz.token_sort_ratio),
    ("name_token_set", "q_name", "s_name", fuzz.token_set_ratio),
    ("name_partial", "q_name", "s_name", fuzz.partial_ratio),
    ("name_jw", "q_name", "s_name", JaroWinkler.normalized_similarity),
    ("name_full_ratio", "q_name_full", "s_name_full", fuzz.ratio),
    ("name_full_token_set", "q_name_full", "s_name_full", fuzz.token_set_ratio),
    ("address_ratio", "q_address", "s_address", fuzz.ratio),
    ("address_token_sort", "q_address", "s_address", fuzz.token_sort_ratio),
    ("address_token_set", "q_address", "s_address", fuzz.token_set_ratio),
    ("address_partial", "q_address", "s_address", fuzz.partial_ratio),
]

PAIR_SQL = """
    SELECT
        p.rid, p.s1, p.part, {label} AS label,
        p.score, p.nkeys, p.rk,
        r.top1, r.top2, r.ncand,
        p.score / r.top1 AS score_ratio,
        r.top1 - p.score AS gap_top1,
        r.top1 - coalesce(r.top2, 0) AS record_margin,
        st.n_cand AS s1_n_cand, st.n_top1 AS s1_n_top1,
        p.score / st.max_score AS score_ratio_s1,
        (p.rid // 10000000000)::int AS source,
        q.name_translit::int AS name_translit,
        q.address_missing::int AS address_missing,
        (s.address = '')::int AS s1_address_missing,
        length(q.name_core) AS q_name_len, length(s.name_core) AS s_name_len,
        length(q.address) AS q_address_len, length(s.address) AS s_address_len,
        len(list_intersect(q.nt, s.nt)) / greatest(len(list_distinct(list_concat(q.nt, s.nt))), 1) AS name_jaccard,
        (q.nt[1] = s.nt[1])::int AS name_first_equal,
        len(list_intersect(q.adt, s.adt)) / greatest(len(list_distinct(list_concat(q.adt, s.adt))), 1) AS address_jaccard,
        len(list_intersect(q.nums, s.nums)) AS num_common,
        len(q.nums) AS q_num_count, len(s.nums) AS s_num_count,
        (len(q.nums) > 0 AND list_has_all(s.nums, q.nums))::int AS nums_subset,
        (q.nums[1] = s.nums[1])::int AS first_num_equal,
        abs(try_cast(q.nums[1] AS DOUBLE) - try_cast(s.nums[1] AS DOUBLE)) AS first_num_diff,
        len(list_filter(q.nums, x -> NOT list_contains(s.nums, x))) AS q_nums_unmatched,
        (len(q.legal) > 0 AND len(s.legal) > 0 AND len(list_intersect(q.legal, s.legal)) = 0)::int AS legal_clash,
        (len(list_intersect(q.legal, s.legal)) > 0)::int AS legal_same,
        len(q.legal) AS q_legal_n, len(s.legal) AS s_legal_n,
        q.name_core AS q_name, s.name_core AS s_name,
        q.name_full AS q_name_full, s.name_full AS s_name_full,
        q.address AS q_address, s.address AS s_address
    FROM pairs p
    JOIN rec_stats r USING (rid)
    JOIN s1_stats st USING (s1)
    JOIN q_text q ON q.eid = p.rid
    JOIN s1_text s ON s.eid = p.s1
    {label_join}
    WHERE p.rid % {{parts}} = {{part}} AND q.eid % {{parts}} = {{part}}
"""

TEXT_SQL = """
    SELECT {id} AS eid, name_core, name_full, address, name_translit, address_missing,
        list_filter(string_split(name_core, ' '), x -> length(x) >= 2) AS nt,
        list_filter(string_split(address, ' '), x -> length(x) >= 2) AS adt,
        regexp_extract_all(address, '\\b[0-9]+\\b') AS nums,
        list_distinct(list_transform(
            list_filter(string_split(name_full, ' '), x -> x IN (
                'pvt', 'private', 'ltd', 'limited', 'llp', 'llc', 'inc', 'incorporated', 'corp',
                'corporation', 'co', 'company', 'lp', 'plc', 'pllc', 'pc',
                'sarl', 'sas', 'sasu', 'eurl', 'sa', 'snc', 'sci')),
            x -> CASE
                WHEN x IN ('pvt', 'private', 'ltd', 'limited') THEN 'pvtltd'
                WHEN x IN ('inc', 'incorporated') THEN 'inc'
                WHEN x IN ('corp', 'corporation') THEN 'corp'
                WHEN x IN ('co', 'company') THEN 'co'
                ELSE x END)) AS legal
    FROM read_parquet('{path}')
"""


def connect(db_path, temp_dir):
    """Open an on-disk DuckDB database with the memory, thread and spill limits used here."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    con.execute(f"SET max_temp_directory_size = '{MAX_TEMP}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def prepare(con, split):
    """Load candidates, per-record and per-source-1 statistics, text columns and (train) labels into DuckDB.
    """
    cand = (CAND_DIR / f"{split}.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE TABLE cand AS SELECT * FROM read_parquet('{cand}')")
    con.execute("""
        CREATE OR REPLACE TABLE rec_stats AS
        SELECT rid, max(score) AS top1, max(score) FILTER (WHERE rk = 2) AS top2, count(*) AS ncand
        FROM cand GROUP BY rid
    """)
    con.execute("""
        CREATE OR REPLACE TABLE s1_stats AS
        SELECT s1, count(*) AS n_cand, count(*) FILTER (WHERE rk = 1) AS n_top1, max(score) AS max_score
        FROM cand GROUP BY s1
    """)
    s1_path = (NORM_DIR / f"{split}_s1.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE TABLE s1_text AS {TEXT_SQL.format(id=ID_EXPR.format(col='entity_id'), path=s1_path)}")
    con.execute("CREATE OR REPLACE TABLE q_text AS " + " UNION ALL ".join(
        TEXT_SQL.format(id=ID_EXPR.format(col="entity_id"), path=(NORM_DIR / f"{split}_s{s}.parquet").as_posix())
        for s in (2, 3)
    ))
    if split == "train":
        gt = (TRAIN_DIR / "train_ground_truth.tsv").as_posix()
        con.execute(f"""
            CREATE OR REPLACE TABLE truth AS
            SELECT {ID_EXPR.format(col='m')} AS rid, {ID_EXPR.format(col='source1_entity_id')} AS s1
            FROM (
                SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m
                FROM read_csv('{gt}', delim = '\t', header = true, all_varchar = true, quote = '', escape = '')
                WHERE matched_entity_ids <> ''
            )
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE rec_part AS
            SELECT rid, CASE
                WHEN bool_or(hash(s1) % {VALID_MOD} = 0) THEN 'valid'
                WHEN hash(rid) % {TRAIN_RECORD_MOD} = 0 THEN 'train'
            END AS part
            FROM cand GROUP BY rid
        """)
        con.execute("""
            CREATE OR REPLACE TABLE pairs AS
            SELECT c.*, r.part FROM cand c JOIN rec_part r USING (rid) WHERE r.part IS NOT NULL
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE pairs AS SELECT *, 'test' AS part FROM cand")
    con.execute("DROP TABLE cand")


def add_fuzzy(batch):
    """Add rapidfuzz name and address similarity columns to a batch and drop the raw text columns."""
    cols = {name: batch.column(name).to_pylist() for name in
            ("q_name", "s_name", "q_name_full", "s_name_full", "q_address", "s_address")}
    arrays = []
    for feature, left, right, scorer in FUZZY:
        values = cpdist(cols[left], cols[right], scorer=scorer, workers=-1, dtype=np.float32)
        arrays.append((feature, pa.array(values)))
    keep = [n for n in batch.schema.names if n not in cols]
    table = batch.select(keep)
    for feature, values in arrays:
        table = table.append_column(feature, values)
    return table


def add_relative(table):
    """Add the combined score plus per-record gap-to-best and rank columns for the main similarities."""
    df = table.select(["rid", "name_token_set", "name_ratio", "name_jw", "name_full_ratio",
                       "address_token_set", "address_ratio"]).to_pandas()
    df["combined"] = df["name_token_set"] + df["address_token_set"]
    table = table.append_column("combined", pa.array(df["combined"].to_numpy(np.float32)))
    grouped = df.groupby("rid", sort=False)
    for feature in RELATIVE:
        best = grouped[feature].transform("max")
        rank = grouped[feature].rank(ascending=False, method="min")
        table = table.append_column(f"{feature}_gap", pa.array((df[feature] - best).to_numpy(np.float32)))
        table = table.append_column(f"{feature}_rank", pa.array(rank.to_numpy(np.float32)))
    return table


def main():
    """Build features part by part and stream them into one parquet file per split part."""
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / f"features_{split}.duckdb"
    start = time.time()
    temp_dir = TEMP_DIR / f"features_{split}"
    con = connect(db_path, temp_dir)
    writers = {}
    try:
        prepare(con, split)
        print(f"prepared in {time.time() - start:.1f}s", flush=True)
        if split == "train":
            sql = PAIR_SQL.format(
                label="(t.rid IS NOT NULL)::int",
                label_join="LEFT JOIN truth t ON t.rid = p.rid AND t.s1 = p.s1",
            )
        else:
            sql = PAIR_SQL.format(label="NULL::int", label_join="")
        rows = 0
        for chunk in range(PARTS):
            batch = con.execute(
                sql.replace("{parts}", str(PARTS)).replace("{part}", str(chunk))
            ).to_arrow_table()
            table = add_relative(add_fuzzy(batch))
            for part in set(table.column("part").to_pylist()):
                sub = table.filter(pc.equal(table.column("part"), part))
                if part not in writers:
                    writers[part] = pq.ParquetWriter(FEAT_DIR / f"{part}.parquet", sub.schema, compression="zstd")
                writers[part].write_table(sub)
            rows += table.num_rows
            print(f"part {chunk + 1}/{PARTS}: {rows:,} pairs ({time.time() - start:.0f}s)", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        con.close()
        db_path.unlink(missing_ok=True)
        db_path.with_suffix(".duckdb.wal").unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"done in {time.time() - start:.1f}s: {sorted(writers)}")


if __name__ == "__main__":
    main()
