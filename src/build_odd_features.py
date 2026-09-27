"""Compute odd (sibling) features for all scored pairs of a split.

Input: data/features/{valid|test}_predictions.parquet (stage 1 scores), data/norm/{split}_s*.parquet,
data/decoy_words.json and data/replacement_words.json.
Output: data/features/{valid|test}_odd.parquet.
Run from src/: python build_odd_features.py train|test   (train writes the validation file)
"""

import shutil
import sys
import time

import duckdb
import pyarrow.parquet as pq

from config import DATA_DIR, TEMP_DIR
from odd_features import feature_table, load_word_lists, sibling_index, sibling_sql, text_sql

FEAT_DIR = DATA_DIR / "features"
NORM_DIR = DATA_DIR / "norm"
PARTS = 8
ID_EXPR = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"


def build(split):
    """Compute odd features in PARTS source-1 hash parts and write them to parquet."""
    start = time.time()
    pred_file = FEAT_DIR / f"{'valid' if split == 'train' else split}_predictions.parquet"
    out_file = FEAT_DIR / f"{'valid' if split == 'train' else split}_odd.parquet"
    temp = TEMP_DIR / f"odd_{split}"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'; SET threads = 4; SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'; SET max_temp_directory_size = '6GB'")
    writer = None
    try:
        con.execute(f"CREATE TABLE pairs AS SELECT rid, s1, p FROM read_parquet('{pred_file.as_posix()}')")
        con.execute("CREATE TABLE best AS SELECT rid, arg_max(s1, p) AS s1, max(p) AS p FROM pairs GROUP BY rid")
        con.execute(f"""
            CREATE TABLE names AS SELECT {ID_EXPR} AS eid, country, name_full, address
            FROM read_parquet('{(NORM_DIR / f'{split}_s*.parquet').as_posix()}')
        """)
        decoys, replacements = load_word_lists()
        rows = 0
        for k in range(PARTS):
            where = f"s1 % {PARTS} = {k}"
            idx = sibling_index(con.execute(sibling_sql("best", "names", f"b.{where}")).to_arrow_table())
            text = con.execute(text_sql("pairs", "names", f"p.{where}")).to_arrow_table()
            table = feature_table(text, idx, decoys, replacements)
            del text, idx
            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
            print(f"{split} part {k}: {rows:,} pairs ({time.time() - start:.0f}s)", flush=True)
    finally:
        if writer is not None:
            writer.close()
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    print(f"wrote {out_file} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    build(sys.argv[1])
