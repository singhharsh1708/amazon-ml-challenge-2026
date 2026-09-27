"""Decoy family flags: house-number shifts and replacement words.

Provides flags used by the decision stage for three decoy families: a house number shifted by a
typical decoy offset, an added "replacement" name word learned from training decoys, and pairs
whose stage 1 probability is below STAGE1_FLOOR. Running the file learns the replacement words.

Input (learning): ground truth, data/norm/train_s*.parquet, data/candidates/train.parquet,
data/decoy_words.json.
Output (learning): data/replacement_words.json.
Run from src/: python decoy_families.py
"""

import json
import time

import duckdb
import numpy as np
from rapidfuzz import fuzz

from config import DATA_DIR, TRAIN_DIR

REPLACEMENT_FILE = DATA_DIR / "replacement_words.json"
NORM_DIR = DATA_DIR / "norm"
CAND_FILE = DATA_DIR / "candidates" / "train.parquet"

VALID_MOD = 20
SAMPLE_MOD = 10
MIN_DECOY = 20
MAX_TRUE_SHARE = 0.002
SHIFT_DECOY = {3, 4, 5, 7, 9, 11, 13, 21}
STAGE1_FLOOR = 0.3

LEGAL = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co",
    "company", "lp", "plc", "pllc", "pc", "sarl", "sas", "sasu", "eurl", "sa", "snc", "sci", "gmbh",
}
STOP = {
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "ms", "m", "s", "shri", "sri", "dba",
    "formerly", "known", "as", "com", "www", "d", "l", "a",
}
ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"


def added_words(q_full, s_full):
    """Return name words in the query that have no exact, substring or fuzzy counterpart in the source 1
    name.
    """
    q_words = [t for t in (q_full or "").split() if t not in LEGAL and t not in STOP]
    s_words = [t for t in (s_full or "").split() if t not in LEGAL and t not in STOP]
    s_set, q_set, s_joined = set(s_words), set(q_words), "".join(s_words)
    added = []
    for t in q_words:
        if t in s_set or (len(t) >= 4 and t in s_joined):
            continue
        if len(t) >= 6 and sum(1 for o in s_words if len(o) >= 3 and o in t) >= 2:
            continue
        if any(fuzz.ratio(t, o) >= 75 for o in s_words if o not in q_set):
            continue
        added.append(t)
    return added


def decoy_shift(q_addr, s_addr):
    """Return True when an unmatched query house number differs from a source 1 number by a decoy offset."""
    q_nums = [t for t in (q_addr or "").split() if t.isdigit()]
    s_nums = [t for t in (s_addr or "").split() if t.isdigit()]
    q_unm = [int(x) for x in q_nums if x not in set(s_nums)]
    s_unm = [int(x) for x in s_nums if x not in set(q_nums)]
    return any(a - b in SHIFT_DECOY for a in q_unm for b in s_unm)


def load_replacement_words():
    """Load the per-country replacement word sets, or an empty dict when the file is missing."""
    if not REPLACEMENT_FILE.exists():
        return {}
    return {country: set(words) for country, words in json.loads(REPLACEMENT_FILE.read_text()).items()}


def family_flags(countries, q_fulls, s_fulls, q_addrs, s_addrs, p1s, first_num_siblings):
    """Return boolean arrays (shift, replacement, promoted) for a batch of assigned pairs."""
    words = load_replacement_words()
    shift = np.zeros(len(p1s), dtype=bool)
    replacement = np.zeros(len(p1s), dtype=bool)
    for i, (country, qf, sf, qa, sa) in enumerate(zip(countries, q_fulls, s_fulls, q_addrs, s_addrs)):
        if first_num_siblings[i] == 0 and decoy_shift(qa, sa):
            shift[i] = True
        vocab = words.get(country)
        if vocab and any(t in vocab for t in added_words(qf, sf)):
            replacement[i] = True
    promoted = np.asarray(p1s, dtype=np.float64) < STAGE1_FLOOR
    return shift, replacement, promoted


def learn():
    """Learn per-country replacement words from sampled true and decoy pairs and write them to JSON."""
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
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, country, name_full
        FROM read_parquet('{(NORM_DIR / 'train_s*.parquet').as_posix()}')
    """)
    pairs = con.execute(f"""
        WITH sampled AS (
            SELECT t.rid, t.s1, true AS is_true FROM truth t WHERE hash(t.rid) % {SAMPLE_MOD} = 0
            UNION ALL
            SELECT c.rid, c.s1, false AS is_true FROM read_parquet('{CAND_FILE.as_posix()}') c
            WHERE c.rk = 1 AND hash(c.rid) % {SAMPLE_MOD} = 0 AND c.rid NOT IN (SELECT rid FROM truth)
        )
        SELECT p.is_true, q.country, q.name_full AS qf, s.name_full AS sf
        FROM sampled p JOIN names q ON q.eid = p.rid JOIN names s ON s.eid = p.s1
        WHERE hash(p.s1) % {VALID_MOD} <> 0
    """).fetchnumpy()
    decoy_vocab = {
        country: set(words)
        for country, words in json.loads((DATA_DIR / "decoy_words.json").read_text()).items()
    }
    counts = {}
    for is_true, country, qf, sf in zip(pairs["is_true"], pairs["country"], pairs["qf"], pairs["sf"]):
        q_tokens, s_tokens = set((qf or "").split()), set((sf or "").split())
        if any(t in decoy_vocab.get(country, ()) and t not in s_tokens for t in q_tokens):
            continue
        for t in set(added_words(qf, sf)):
            if len(t) >= 3:
                counts.setdefault((country, t), [0, 0])[int(bool(is_true))] += 1
    words = {}
    for (country, t), (decoy_n, true_n) in counts.items():
        if decoy_n >= MIN_DECOY and true_n <= MAX_TRUE_SHARE * decoy_n:
            words.setdefault(country, []).append(t)
    REPLACEMENT_FILE.write_text(json.dumps({k: sorted(v) for k, v in words.items()}, indent=1))
    print(f"pairs {len(pairs['qf']):,}; replacement words: " + ", ".join(f"{k} {len(v)}" for k, v in words.items())
          + f" ({time.time() - start:.0f}s) -> {REPLACEMENT_FILE}")


if __name__ == "__main__":
    learn()
