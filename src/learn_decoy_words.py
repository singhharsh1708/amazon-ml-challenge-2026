"""Learn per-country decoy words from the training candidates.

A decoy word is a name token that the source 2/3 record adds to an otherwise contained source 1
name and that almost never appears in true matches (at least MIN_DECOY decoy occurrences and a
true share of at most MAX_TRUE_SHARE). Only the training part (validation slice excluded) is used.

Input: data/candidates/train.parquet, data/norm/train_s*.parquet, ground truth.
Output: data/decoy_words.json.
Run from src/: python learn_decoy_words.py
"""

import json
import time

import duckdb

from config import DATA_DIR, TRAIN_DIR

NORM_DIR = DATA_DIR / "norm"
CAND_FILE = DATA_DIR / "candidates" / "train.parquet"
OUT_FILE = DATA_DIR / "decoy_words.json"

VALID_MOD = 20
MIN_DECOY = 200
MAX_TRUE_SHARE = 0.002

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"
TOKENS = "list_filter(string_split({col}, ' '), x -> regexp_full_match(x, '[a-z]{{3,}}'))"


def main():
    """Count added name tokens on true and decoy top-1 pairs and write the decoy word lists."""
    start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    con.execute("SET threads = 4")
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
        CREATE TABLE names AS
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, country,
            {TOKENS.format(col='name_full')} AS full_tokens,
            list_filter(string_split(name_core, ' '), x -> length(x) >= 2) AS core_tokens
        FROM read_parquet('{(NORM_DIR / 'train_s*.parquet').as_posix()}')
    """)
    con.execute(f"""
        CREATE TABLE added AS
        SELECT q.country, t.rid IS NOT NULL AS is_true,
            list_filter(q.full_tokens, x -> NOT list_contains(s.full_tokens, x)) AS extra
        FROM read_parquet('{CAND_FILE.as_posix()}') c
        JOIN names q ON q.eid = c.rid
        JOIN names s ON s.eid = c.s1
        LEFT JOIN truth t ON t.rid = c.rid AND t.s1 = c.s1
        LEFT JOIN truth any_t ON any_t.rid = c.rid
        WHERE c.rk = 1
          AND hash(c.s1) % {VALID_MOD} <> 0
          AND len(s.core_tokens) > 0
          AND list_has_all(q.full_tokens, s.core_tokens)
          AND (t.rid IS NOT NULL OR any_t.rid IS NULL)
    """)
    rows = con.execute(f"""
        SELECT country, token,
            count(*) FILTER (WHERE NOT is_true) AS decoy_n,
            count(*) FILTER (WHERE is_true) AS true_n
        FROM (SELECT country, is_true, unnest(extra) AS token FROM added)
        GROUP BY country, token
        HAVING count(*) FILTER (WHERE NOT is_true) >= {MIN_DECOY}
           AND count(*) FILTER (WHERE is_true) <= {MAX_TRUE_SHARE} * count(*) FILTER (WHERE NOT is_true)
        ORDER BY country, decoy_n DESC
    """).fetchall()
    words = {}
    for country, token, decoy_n, true_n in rows:
        words.setdefault(country, []).append(token)
    OUT_FILE.write_text(json.dumps(words, indent=1))
    for country, tokens in words.items():
        print(f"{country}: {len(tokens)} decoy words, e.g. {tokens[:25]}")
    print(f"saved {OUT_FILE} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
