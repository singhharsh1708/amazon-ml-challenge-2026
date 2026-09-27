"""Copy of a src/ module as used by the France re-run (fixed France normalization). Experiment/validation script block_candidates.py for this step.

Step context: Copy of a src/ module as used by the France re-run (fixed France normalization). It is put ahead of src/ on sys.path by the France re-run scripts.

Command-line arguments used: argv[1], argv[2].
"""
import os
import shutil
import sys
import time

import duckdb

from config import DATA_DIR, TEMP_DIR

NORM_DIR = DATA_DIR / "norm"
CAND_DIR = DATA_DIR / "candidates"

TOP_K = int(os.environ.get("TOP_K", 10))
SCORE_RATIO = 0.5
UNIGRAM_CAP = 50
BIGRAM_CAP = 500
CHUNKS = 40

MEMORY_LIMIT = "4GB"
THREADS = 4
MAX_TEMP = "10GB"

ID_EXPR = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"


def keys_sql(split, source, where=""):
    """Return the SQL that computes family keys for a pair file."""
    path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
    return f"""
    select distinct eid, hash(country || '|' || tk) as h, tk[1] in ('n', 'a') as unigram
    from (
        with r as (
            select {ID_EXPR} as eid, country, name_core,
                list_filter(string_split(name_core, ' '), x -> length(x) >= 2) as nt,
                list_filter(string_split(address, ' '), x -> length(x) >= 2) as adt
            from read_parquet('{path}') {where}
        )
        select eid, country, 'n' || unnest(nt) as tk from r
        union all
        select eid, country, 'a' || unnest(adt) from r
        union all
        select eid, country, 'N' || least(nt[i], nt[i + 1]) || '_' || greatest(nt[i], nt[i + 1])
        from (select *, unnest(range(1, len(nt))) as i from r where len(nt) >= 2)
        union all
        select eid, country, 'A' || least(adt[i], adt[i + 1]) || '_' || greatest(adt[i], adt[i + 1])
        from (select *, unnest(range(1, len(adt))) as i from r where len(adt) >= 2)
        union all
        select eid, country, 'C' || replace(name_core, ' ', '') from r where length(name_core) >= 4
        union all
        select eid, country, 'X' || n || '|' || unnest(adt)
        from (select eid, country, unnest(nt) as n, adt from r)
    )
    """


def connect(db_path, temp_dir):
    """Open a DuckDB database with a private temp directory."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    con.execute(f"SET max_temp_directory_size = '{MAX_TEMP}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def build_index(con, split):
    """Build the blocking index of a split."""
    start = time.time()
    con.execute(f"CREATE OR REPLACE TABLE s1_keys AS {keys_sql(split, 1)}")
    n1 = con.execute(
        f"SELECT count(*) FROM read_parquet('{(NORM_DIR / f'{split}_s1.parquet').as_posix()}')"
    ).fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE s1_index AS
        WITH df AS (SELECT h, unigram, count(*) AS df FROM s1_keys GROUP BY h, unigram)
        SELECT k.h, k.eid, ln({n1} / df.df) AS idf
        FROM s1_keys k JOIN df USING (h, unigram)
        WHERE df.df <= CASE WHEN df.unigram THEN {UNIGRAM_CAP} ELSE {BIGRAM_CAP} END
    """)
    con.execute("DROP TABLE s1_keys")
    rows = con.execute("SELECT count(*) FROM s1_index").fetchone()[0]
    print(f"s1 index: {rows:,} postings in {time.time() - start:.1f}s", flush=True)


def block_chunk(con, split, chunk):
    """Generate the candidate pairs of one hash chunk of a split."""
    where = f"WHERE hash(entity_id) % {CHUNKS} = {chunk}"
    con.execute(f"""
        INSERT INTO cand
        SELECT rid, s1, score, nkeys,
            row_number() OVER (PARTITION BY rid ORDER BY score DESC, s1) AS rk
        FROM (
            SELECT q.eid AS rid, s.eid AS s1, sum(s.idf) AS score, count(*) AS nkeys
            FROM (({keys_sql(split, 2, where)}) UNION ALL ({keys_sql(split, 3, where)})) q
            JOIN s1_index s ON q.h = s.h
            GROUP BY q.eid, s.eid
        )
        QUALIFY rk <= {TOP_K} AND score >= {SCORE_RATIO} * max(score) OVER (PARTITION BY rid)
    """)


def main():
    """Parse the command line and run the step."""
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    max_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNKS
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / f"blocking_{split}.duckdb"
    suffix = "" if max_chunks == CHUNKS else "_sample"
    out_path = CAND_DIR / f"{split}{suffix}.parquet"
    start = time.time()
    temp_dir = TEMP_DIR / f"blocking_{split}"
    con = connect(db_path, temp_dir)
    try:
        build_index(con, split)
        con.execute("""
            CREATE OR REPLACE TABLE cand (
                rid BIGINT, s1 BIGINT, score DOUBLE, nkeys BIGINT, rk BIGINT
            )
        """)
        for chunk in range(max_chunks):
            t = time.time()
            block_chunk(con, split, chunk)
            print(f"chunk {chunk + 1}/{max_chunks} in {time.time() - t:.1f}s", flush=True)
        con.execute(
            f"COPY cand TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows, records = con.execute(
            "SELECT count(*), count(DISTINCT rid) FROM cand"
        ).fetchone()
        print(f"{split}: {rows:,} candidate pairs for {records:,} records "
              f"in {time.time() - start:.1f}s -> {out_path}")
    finally:
        con.close()
        db_path.unlink(missing_ok=True)
        db_path.with_suffix(".duckdb.wal").unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
