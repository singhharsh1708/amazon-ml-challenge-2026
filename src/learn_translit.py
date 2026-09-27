"""Learn a token-level transliteration map from the training ground truth.

For every true (source 1, source 2/3) pair whose source 2/3 name contains non-ASCII characters,
the two normalised names are aligned token by token when they have the same length. A token is
mapped to its most frequent counterpart when that counterpart is seen at least MIN_COUNT times
and in at least MIN_SHARE of the aligned occurrences.

Input: train_source{1,2,3}.tsv and train_ground_truth.tsv under the data root.
Output: data/translit_map.json (read by normalize.py).
Run from src/: python learn_translit.py
"""

import json
import time
from collections import Counter, defaultdict

import duckdb

from config import DATA_DIR, TRAIN_DIR
from normalize import normalize_name

MAP_FILE = DATA_DIR / "translit_map.json"
MIN_COUNT = 20
MIN_SHARE = 0.6


def main():
    """Align transliterated true pairs, count token correspondences and write the map."""
    start = time.time()
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    con.execute("SET threads = 4")

    def rd(path):
        """Return a DuckDB read_csv expression for a raw tab-separated challenge file."""
        return (
            f"read_csv('{path.as_posix()}', delim = '\t', header = true, all_varchar = true, "
            "quote = '', escape = '')"
        )

    rows = con.execute(f"""
        WITH pairs AS (
            SELECT source1_entity_id AS s1, trim(unnest(string_split(matched_entity_ids, ','))) AS m
            FROM {rd(TRAIN_DIR / 'train_ground_truth.tsv')}
            WHERE matched_entity_ids <> ''
        ),
        noisy AS (
            SELECT entity_id, business_name FROM {rd(TRAIN_DIR / 'train_source2.tsv')}
            WHERE regexp_matches(business_name, '[^\\x00-\\x7F]')
            UNION ALL
            SELECT entity_id, business_name FROM {rd(TRAIN_DIR / 'train_source3.tsv')}
            WHERE regexp_matches(business_name, '[^\\x00-\\x7F]')
        )
        SELECT n.business_name, s.business_name
        FROM pairs p
        JOIN noisy n ON n.entity_id = p.m
        JOIN {rd(TRAIN_DIR / 'train_source1.tsv')} s ON s.entity_id = p.s1
    """).fetchall()
    print(f"transliterated matched pairs: {len(rows):,} ({time.time() - start:.0f}s)")

    counts = defaultdict(Counter)
    aligned = 0
    for noisy, clean in rows:
        left = normalize_name(noisy, use_map=False)[0].split()
        right = normalize_name(clean, use_map=False)[0].split()
        if len(left) != len(right) or not left:
            continue
        aligned += 1
        for a, b in zip(left, right):
            counts[a][b] += 1

    mapping = {}
    for token, targets in counts.items():
        target, n = targets.most_common(1)[0]
        total = sum(targets.values())
        if target != token and n >= MIN_COUNT and n / total >= MIN_SHARE:
            mapping[token] = target
    MAP_FILE.write_text(json.dumps(mapping, indent=0, sort_keys=True))
    print(f"aligned {aligned:,} pairs, learned {len(mapping):,} mappings -> {MAP_FILE}")
    for token, target in sorted(mapping.items(), key=lambda kv: -sum(counts[kv[0]].values()))[:40]:
        print(f"  {token} -> {target} ({counts[token][target]:,}/{sum(counts[token].values()):,})")


if __name__ == "__main__":
    main()
