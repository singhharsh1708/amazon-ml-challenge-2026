"""Measure blocking recall on the training split.

Joins the training ground truth with the candidate file and prints recall at ranks 1, 3, 5, 10,
15, 20 and overall, plus recall by source.

Input: data/candidates/{name}.parquet and train_ground_truth.tsv.
Output: printed report only.
Run from src/: python evaluate_blocking.py [train] [max_chunks]
"""

import sys

import duckdb

from config import DATA_DIR, TRAIN_DIR

CHUNKS = 40
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"


def main():
    """Compute and print candidate recall at several ranks, overall and per source."""
    name = sys.argv[1] if len(sys.argv) > 1 else "train"
    max_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNKS
    CAND_FILE = DATA_DIR / "candidates" / f"{name}.parquet"
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 4")
    con.execute(f"""
        CREATE TABLE truth AS
        SELECT {ID_EXPR.format(col='m')} AS rid, {ID_EXPR.format(col='source1_entity_id')} AS s1
        FROM (
            SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m
            FROM read_csv('{GT_FILE.as_posix()}', delim = '\t', header = true,
                          all_varchar = true, quote = '', escape = '')
            WHERE matched_entity_ids <> ''
        )
        WHERE hash(m) % {CHUNKS} < {max_chunks}
    """)
    row = con.execute(f"""
        SELECT count(*),
            avg(coalesce(rk <= 1, false)::int), avg(coalesce(rk <= 3, false)::int), avg(coalesce(rk <= 5, false)::int),
            avg(coalesce(rk <= 10, false)::int), avg(coalesce(rk <= 15, false)::int),
            avg(coalesce(rk <= 20, false)::int), avg((rk IS NOT NULL)::int)
        FROM truth t
        LEFT JOIN read_parquet('{CAND_FILE.as_posix()}') c USING (rid, s1)
    """).fetchone()
    print(f"true pairs: {row[0]:,}")
    print("recall@1 {:.4f}  @3 {:.4f}  @5 {:.4f}  @10 {:.4f}  @15 {:.4f}  @20 {:.4f}  any {:.4f}".format(*row[1:]))
    by_source = con.execute(f"""
        SELECT rid // 10000000000 AS source, avg(coalesce(rk <= 1, false)::int), avg((rk IS NOT NULL)::int)
        FROM truth t
        LEFT JOIN read_parquet('{CAND_FILE.as_posix()}') c USING (rid, s1)
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    for source, r1, rany in by_source:
        print(f"source {source}: recall@1 {r1:.4f}  any {rany:.4f}")


if __name__ == "__main__":
    main()
