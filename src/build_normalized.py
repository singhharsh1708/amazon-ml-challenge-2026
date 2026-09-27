"""Normalise every source file into parquet.

Reads the raw train and/or test source TSVs in chunks, applies normalize_name and
normalize_address in a process pool and writes one parquet per split and source with columns
entity_id, country, name_full, name_core, address, name_translit, address_missing.

Input: {train,test}_source{1,2,3}.tsv under the data root.
Output: data/norm/{split}_s{1,2,3}.parquet.
Run from src/: python build_normalized.py [train] [test]   (both splits when no argument)
"""

import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from config import DATA_DIR, TEST_DIR, TRAIN_DIR
from normalize import normalize_address, normalize_name

NORM_DIR = DATA_DIR / "norm"
CHUNK_ROWS = 200_000

SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("name_full", pa.string()),
    ("name_core", pa.string()),
    ("address", pa.string()),
    ("name_translit", pa.bool_()),
    ("address_missing", pa.bool_()),
])


def normalize_chunk(args):
    """Normalise one chunk of (ids, names, addresses, countries) into an Arrow record batch."""
    ids, names, addresses, countries = args
    name_full, name_core, address, translit, missing = [], [], [], [], []
    for name, addr, country in zip(names, addresses, countries):
        full, core = normalize_name(name)
        norm_addr = normalize_address(addr, country)
        name_full.append(full)
        name_core.append(core)
        address.append(norm_addr)
        translit.append(bool(name) and not name.isascii())
        missing.append(norm_addr == "")
    return pa.record_batch(
        [
            pa.array(ids),
            pa.array([c.lower() for c in countries]),
            pa.array(name_full),
            pa.array(name_core),
            pa.array(address),
            pa.array(translit),
            pa.array(missing),
        ],
        schema=SCHEMA,
    )


def read_chunks(path):
    """Stream a raw source TSV and yield column slices of CHUNK_ROWS rows."""
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={
                "entity_id": pa.string(),
                "business_name": pa.string(),
                "business_address": pa.string(),
                "country": pa.string(),
            }
        ),
    )
    for batch in reader:
        cols = batch.to_pydict()
        n = batch.num_rows
        for start in range(0, n, CHUNK_ROWS):
            end = start + CHUNK_ROWS
            yield (
                cols["entity_id"][start:end],
                cols["business_name"][start:end],
                cols["business_address"][start:end],
                cols["country"][start:end],
            )


def build(split, source, pool):
    """Normalise one split/source file and write it to data/norm as zstd parquet."""
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    src = folder / f"{split}_source{source}.tsv"
    dst = NORM_DIR / f"{split}_s{source}.parquet"
    start = time.time()
    rows = 0
    with pq.ParquetWriter(dst, SCHEMA, compression="zstd") as writer:
        for batch in pool.imap(normalize_chunk, read_chunks(src)):
            writer.write_batch(batch)
            rows += batch.num_rows
    print(f"{dst.name}: {rows:,} rows in {time.time() - start:.1f}s", flush=True)


def main():
    """Normalise sources 1 to 3 for the requested splits using all CPU cores."""
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    splits = sys.argv[1:] or ["train", "test"]
    with Pool(os.cpu_count()) as pool:
        for split in splits:
            for source in (1, 2, 3):
                build(split, source, pool)


if __name__ == "__main__":
    main()
