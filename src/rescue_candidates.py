"""Generate rescue candidates for records left unassigned.

Blocking misses some true pairs. For unassigned source 2/3 records this generates extra
candidate pairs outside the blocked candidates through six key families: exact address
(aexact), core name plus house number (corenum), concatenated/anagram name (cat), house number
plus sorted street word (anum), street word pairs (aalpha) and single-deletion name typos
(typo). Keys are capped by document frequency, every family except exact address is gated to
records with an out-of-vocabulary name token, the name families require a shared address token,
and the TOP pairs per record are kept by family count and priority.

train builds pairs for the validation slice (with labels and an assigned flag) to train the
rescue model; test builds pairs for test records not present in the matching file.

Input: data/norm, data/candidates, output/matching_results.tsv (test) or
data/rescue/valid_assigned.parquet and valid_scores.parquet (train).
Output: data/rescue/valid_rescue.parquet or data/rescue/test_rescue.parquet.
Run from src/: python rescue_candidates.py [train|test] [matching.tsv]
"""

import shutil
import sys
import time
from pathlib import Path

import duckdb

from config import DATA_DIR, OUTPUT_DIR, TEMP_DIR, TRAIN_DIR
from train_stage2 import VALID_MOD

NORM_DIR = DATA_DIR / "norm"
CAND_DIR = DATA_DIR / "candidates"
RESCUE_DIR = DATA_DIR / "rescue"
VALID_ASSIGNED = RESCUE_DIR / "valid_assigned.parquet"
VALID_SCORES = RESCUE_DIR / "valid_scores.parquet"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
OUT_FILES = {"train": RESCUE_DIR / "valid_rescue.parquet", "test": RESCUE_DIR / "test_rescue.parquet"}

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"
LEGAL = [
    "limited", "ltd", "llc", "llp", "lp", "inc", "pvt", "corp", "corporation", "co",
    "company", "the", "and", "of", "et", "pllc", "plc", "pc",
]
DF_MAX = 100
GEN_CAPS = {"aexact": 10, "corenum": 10, "cat": 10, "anum": 10, "aalpha": 10, "typo": 10}
CAPS = {"aexact": 3, "corenum": 5, "cat": 10, "anum": (1, 3), "aalpha": 1, "typo": 10}
UNGATED = {"aexact"}
ADDR_CHECK = {"cat", "typo", "corenum"}
PRIORITY = {"aexact": 6, "corenum": 5, "cat": 4, "anum": 3, "aalpha": 2, "typo": 1}
TOP = 5
SLICE_DF = 20
PASS_P = 0.85
RECORD_CHUNKS = 10
FINAL_CHUNKS = 20


def connect(db, temp):
    """Open a DuckDB database with spill limits and register the string and subset macros."""
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute(f"""
        SET memory_limit = '1200MB'; SET threads = 2; SET preserve_insertion_order = false;
        SET temp_directory = '{temp.as_posix()}'; SET max_temp_directory_size = '6GB'
    """)
    con.execute("CREATE OR REPLACE MACRO srt(s) AS array_to_string(list_sort(string_split(s, '')), '')")
    con.execute("""
        CREATE OR REPLACE MACRO dels(sw) AS list_distinct(list_concat([sw], list_transform(
            range(1, length(sw) + 1), dj -> substr(sw, 1, dj - 1) || substr(sw, dj + 1))))
    """)
    con.execute("""
        CREATE OR REPLACE MACRO subsets(st) AS list_filter(range(1, (1 << len(st))::bigint), sm -> bit_count(sm) >= 2)
    """)
    con.execute("""
        CREATE OR REPLACE MACRO pick(pt, pm) AS list_filter(pt, (px, pi) -> ((pm >> (pi - 1)) & 1) = 1)
    """)
    return con


def truth_sql(gt=GROUND_TRUTH):
    """Return SQL for (rid, s1) ground truth pairs as numeric ids."""
    return f"""
        SELECT {ID_EXPR.format(col='m')} AS rid, {ID_EXPR.format(col='source1_entity_id')} AS s1
        FROM (
            SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m
            FROM read_csv('{gt.as_posix()}', delim = '\t', header = true, all_varchar = true, quote = '', escape = '')
            WHERE matched_entity_ids <> ''
        )
    """


def base_sql(path, where="", keep=""):
    """Return SQL with core name, name tokens and address tokens of a normalised source file."""
    legal = ", ".join(f"'{w}'" for w in LEGAL)
    return f"""
        SELECT * FROM (
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, country,
            coalesce(name_core, '') AS core,
            list_filter(string_split(coalesce(name_core, ''), ' '), x -> x <> '') AS ct,
            list_filter(string_split(coalesce(name_full, ''), ' '), x -> x <> '' AND x NOT IN ({legal})) AS ft,
            list_distinct(list_filter(string_split(coalesce(address, ''), ' '), x -> x <> '')) AS adt
        FROM read_parquet('{path}') {where}
        ) b {keep}
    """


def enrich_sql(path, where="", keep=""):
    """Extend base_sql with alphabetic address words and house numbers."""
    return f"""
        SELECT *,
            list_filter(adt, x -> regexp_matches(x, '^[a-z]{{4,}}$')) AS al,
            list_distinct(list_filter(list_transform(adt, x -> regexp_extract(x, '([0-9]+)', 1)), d -> d <> '')) AS nums
        FROM ({base_sql(path, where, keep)})
    """


def family_keys(fam, s1_side):
    """Return the SQL list expression producing one family's keys for the source 1 or query side."""
    if fam == "cat":
        if s1_side:
            return """list_distinct(list_transform(list_filter(
                list_transform(subsets(ft[1:7]), m -> array_to_string(pick(ft[1:7], m), '')),
                s -> length(s) >= 7), s -> srt(s)))"""
        return """list_distinct(list_concat(
            CASE WHEN length(replace(core, ' ', '')) >= 7 THEN [srt(replace(core, ' ', ''))] ELSE [] END,
            list_transform(list_filter(ct, x -> length(x) >= 8), x -> srt(x))))"""
    if fam == "aexact":
        return """CASE WHEN len(adt) >= 3 AND list_bool_or(list_transform(adt, x -> regexp_matches(x, '[0-9]')))
            THEN [array_to_string(list_sort(adt), ' ')] ELSE [] END"""
    if fam == "anum":
        return "list_distinct(flatten(list_transform(nums, d -> list_transform(al, a -> d || '|' || srt(a)))))"
    if fam == "aalpha":
        return """list_distinct(flatten(list_transform(range(1, len(al[1:8]) + 1),
            i -> list_transform(range(i + 1, len(al[1:8]) + 1),
                j -> least(al[i], al[j]) || '|' || greatest(al[i], al[j])))))"""
    if fam == "corenum":
        return """CASE WHEN length(replace(core, ' ', '')) >= 3
            THEN list_transform(nums, d -> replace(core, ' ', '') || '|' || d) ELSE [] END"""
    raise ValueError(fam)


def keys_sql(fam, path, s1_side, where="", keep="", gate_typo=False):
    """Return SQL emitting hashed (h, eid) keys of one family for a normalised source file."""
    if fam == "typo":
        oov = "LEFT JOIN voc w USING (country, tok)" if gate_typo else ""
        flag = "w.tok IS NULL" if gate_typo else "true"
        return f"""SELECT h, eid FROM (
            WITH e AS ({base_sql(path, where, keep)}),
            t0 AS (SELECT eid, country, unnest(ct[1:5]) AS tok, unnest(range(1, len(ct[1:5]) + 1)) AS pos FROM e),
            t AS (SELECT t0.*, {flag} AS oov FROM t0 {oov}),
            dv AS (SELECT tok, unnest(dels(srt(tok))) AS v FROM (SELECT DISTINCT tok FROM t WHERE length(tok) >= 4))
            SELECT {"DISTINCT" if s1_side else ""} hash('typo|' || a.country || '|' || a.tok || '|' || dv.v) AS h, a.eid
            FROM t a JOIN t b ON a.eid = b.eid AND a.pos <> b.pos
            JOIN dv ON dv.tok = b.tok
            WHERE length(a.tok) >= 3 AND b.oov
        )"""
    return f"""
        SELECT hash('{fam}|' || country || '|' || k) AS h, eid
        FROM (SELECT eid, country, unnest({family_keys(fam, s1_side)}) AS k FROM ({enrich_sql(path, where, keep)}))
    """


def record_keys_sql(split, fams, where="", keep="", gate_typo=False):
    """Return SQL for the keys of all requested families over sources 2 and 3."""
    parts = []
    for source in (2, 3):
        path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
        for fam in fams:
            k = keep[fam] if isinstance(keep, dict) else keep
            parts.append(f"SELECT '{fam}' AS fam, h, eid AS rid FROM ({keys_sql(fam, path, False, where, k, gate_typo)})")
    return " UNION ALL ".join(parts)


def build_s1_keys(con, split):
    """Build the s1k table of source 1 keys with document frequency at most DF_MAX."""
    path = (NORM_DIR / f"{split}_s1.parquet").as_posix()
    con.execute("CREATE OR REPLACE TABLE s1k (fam VARCHAR, h UBIGINT, s1 BIGINT, df INTEGER)")
    for fam in GEN_CAPS:
        start = time.time()
        con.execute(f"CREATE OR REPLACE TEMP TABLE k AS {keys_sql(fam, path, True)}")
        con.execute(f"""
            INSERT INTO s1k
            SELECT '{fam}', k.h, k.eid, d.df
            FROM k JOIN (SELECT h, count(*) AS df FROM k GROUP BY h HAVING count(*) <= {DF_MAX}) d USING (h)
        """)
        total = con.execute("SELECT count(*) FROM k").fetchone()[0]
        kept = con.execute(f"SELECT count(*) FROM s1k WHERE fam = '{fam}'").fetchone()[0]
        con.execute("DROP TABLE k")
        print(f"s1 keys {fam}: {total:,} generated, {kept:,} with df <= {DF_MAX} in {time.time() - start:.1f}s",
              flush=True)


def load_scope(con, split, matching_file, valid_assigned, valid_scores):
    """Create the tables that define which records are already assigned (and the train slice)."""
    if split == "train":
        con.execute(f"""
            CREATE OR REPLACE TABLE slice AS
            SELECT {ID_EXPR.format(col='entity_id')} AS s1 FROM read_parquet('{(NORM_DIR / 'train_s1.parquet').as_posix()}')
            WHERE hash({ID_EXPR.format(col='entity_id')}) % {VALID_MOD} = 0
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE assigned AS
            SELECT rid FROM read_parquet('{valid_assigned.as_posix()}')
            UNION
            SELECT rid FROM read_parquet('{valid_scores.as_posix()}') GROUP BY rid HAVING max(p2) >= {PASS_P}
        """)
        con.execute(f"CREATE OR REPLACE TABLE truth AS {truth_sql()}")
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE assigned AS
            SELECT DISTINCT {ID_EXPR.format(col='m')} AS rid
            FROM (SELECT trim(unnest(string_split(matched_entity_ids, ','))) AS m
                  FROM read_csv('{matching_file.as_posix()}', delim = '\t', header = true, all_varchar = true,
                                quote = '', escape = '')
                  WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '')
            WHERE m <> ''
        """)


def cap_cond(caps):
    """Return the SQL condition applying per-family document frequency caps."""
    conds = []
    for fam, cap in caps.items():
        if isinstance(cap, (list, tuple)):
            conds.append(f"(fam = '{fam}' AND (df <= {cap[0]} OR (nk >= 2 AND df <= {cap[1]})))")
        else:
            conds.append(f"(fam = '{fam}' AND df <= {cap})")
    return " OR ".join(conds)


def drop_candidates(con, table, split):
    """Remove pairs that are already in the blocked candidate file."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT p.* FROM {table} p
        ANTI JOIN (SELECT rid, s1 FROM read_parquet('{(CAND_DIR / f'{split}.parquet').as_posix()}')) c USING (rid, s1)
    """)
    n = con.execute(f"SELECT count(*), count(DISTINCT rid) FROM {table}").fetchone()
    print(f"{table} outside existing candidates: {n[0]:,} rows for {n[1]:,} records", flush=True)


def slice_pairs(con, split):
    """Match validation-slice keys to find records with a rescue pair in the slice (train only)."""
    fams = list(GEN_CAPS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1k_slice AS
        SELECT fam, h, s1, df FROM s1k SEMI JOIN slice USING (s1) WHERE df <= {SLICE_DF}
    """)
    con.execute("CREATE OR REPLACE TABLE spairs (rid BIGINT, s1 BIGINT, fam VARCHAR, df INTEGER, nk INTEGER)")
    for chunk in range(RECORD_CHUNKS):
        start = time.time()
        where = f"WHERE hash(entity_id) % {RECORD_CHUNKS} = {chunk}"
        con.execute(f"""
            INSERT INTO spairs
            SELECT q.rid, s.s1, s.fam, min(s.df), count(*)
            FROM ({record_keys_sql(split, fams, where)}) q JOIN s1k_slice s ON s.h = q.h
            GROUP BY q.rid, s.s1, s.fam
        """)
        print(f"slice pairs {chunk + 1}/{RECORD_CHUNKS} in {time.time() - start:.1f}s", flush=True)
    con.execute("DROP TABLE s1k_slice")
    drop_candidates(con, "spairs", split)


def records_sql(split):
    """Return SQL listing the source 2/3 records with country and core name."""
    parts = []
    for source in (2, 3):
        path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
        parts.append(f"""SELECT {ID_EXPR.format(col='entity_id')} AS rid, country, coalesce(name_core, '') AS core
            FROM read_parquet('{path}')""")
    return " UNION ALL ".join(parts)


def build_population(con, split):
    """Build the record population to rescue and its out-of-vocabulary gate."""
    con.execute(f"""
        CREATE OR REPLACE TABLE voc AS
        SELECT DISTINCT country, unnest(string_split(coalesce(name_full, ''), ' ')) AS tok
        FROM read_parquet('{(NORM_DIR / f'{split}_s1.parquet').as_posix()}')
    """)
    if split == "train":
        con.execute(f"CREATE OR REPLACE TABLE pop AS SELECT DISTINCT rid FROM spairs WHERE {cap_cond(GEN_CAPS)}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE pop AS SELECT rid FROM ({records_sql(split)}) ANTI JOIN assigned USING (rid)")
    con.execute(f"""
        CREATE OR REPLACE TABLE gate AS
        SELECT DISTINCT r.rid
        FROM (SELECT rid, country, unnest(string_split(core, ' ')) AS tok FROM ({records_sql(split)})) r
        SEMI JOIN pop USING (rid)
        ANTI JOIN voc USING (country, tok)
        WHERE length(r.tok) >= 3
    """)
    n = con.execute("SELECT (SELECT count(*) FROM pop), (SELECT count(*) FROM gate)").fetchone()
    print(f"population {n[0]:,} records, {n[1]:,} with an out-of-vocabulary name token", flush=True)


def final_pairs(con, split):
    """Match gated record keys against capped source 1 keys in chunks into the pairs table."""
    fams = list(GEN_CAPS)
    build_population(con, split)
    con.execute(f"CREATE OR REPLACE TEMP TABLE s1kc AS SELECT h, s1, fam, df FROM s1k WHERE {cap_cond(GEN_CAPS)}")
    keep = {fam: f"SEMI JOIN {'pop' if fam in UNGATED else 'gate'} k ON k.rid = b.eid" for fam in fams}
    con.execute("CREATE OR REPLACE TABLE pairs (rid BIGINT, s1 BIGINT, fam VARCHAR, df INTEGER, nk INTEGER)")
    for chunk in range(FINAL_CHUNKS):
        start = time.time()
        where = f"WHERE hash(entity_id) % {FINAL_CHUNKS} = {chunk}"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE qk AS
            SELECT h, rid FROM ({record_keys_sql(split, fams, where, keep, gate_typo=True)})
        """)
        con.execute("""
            INSERT INTO pairs
            SELECT q.rid, s.s1, s.fam, min(s.df), count(*)
            FROM qk q JOIN s1kc s ON s.h = q.h
            GROUP BY q.rid, s.s1, s.fam
        """)
        print(f"pairs {chunk + 1}/{FINAL_CHUNKS} in {time.time() - start:.1f}s", flush=True)
    con.execute("DROP TABLE qk")
    con.execute("DROP TABLE s1kc")
    drop_candidates(con, "pairs", split)


def address_tokens_sql(split, sources, col):
    """Return SQL for distinct address tokens of the entities that appear in pairs."""
    parts = []
    for source in sources:
        path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
        parts.append(f"""SELECT {ID_EXPR.format(col='entity_id')} AS eid,
            list_distinct(list_filter(string_split(coalesce(address, ''), ' '),
                                      x -> length(x) >= 3 OR regexp_matches(x, '^[0-9]+$'))) AS adt
            FROM read_parquet('{path}')""")
    return f"SELECT * FROM ({' UNION ALL '.join(parts)}) SEMI JOIN (SELECT DISTINCT {col} AS eid FROM pairs) USING (eid)"


def address_filter(con, split):
    """Keep name-family pairs only when they share an address token (or the query has none)."""
    checked = ", ".join(f"'{fam}'" for fam in ADDR_CHECK)
    con.execute(f"CREATE OR REPLACE TEMP TABLE qa AS {address_tokens_sql(split, (2, 3), 'rid')}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE sa AS {address_tokens_sql(split, (1,), 's1')}")
    con.execute(f"""
        CREATE OR REPLACE TABLE fpairs AS
        SELECT p.* FROM pairs p
        LEFT JOIN qa ON qa.eid = p.rid
        LEFT JOIN sa ON sa.eid = p.s1
        WHERE p.fam NOT IN ({checked}) OR coalesce(len(qa.adt), 0) = 0 OR list_has_any(qa.adt, sa.adt)
    """)
    con.execute("DROP TABLE qa")
    con.execute("DROP TABLE sa")
    n = con.execute("SELECT (SELECT count(*) FROM pairs), (SELECT count(*) FROM fpairs)").fetchone()
    print(f"address check kept {n[1]:,} of {n[0]:,} family matches", flush=True)


def select_sql(caps=CAPS, top=TOP, table="fpairs"):
    """Return SQL that keeps the top pairs per record by family count, priority and frequency."""
    prio = " ".join(f"WHEN '{fam}' THEN {p}" for fam, p in PRIORITY.items())
    return f"""
        SELECT rid, s1, family FROM (
            SELECT rid, s1, family,
                row_number() OVER (PARTITION BY rid ORDER BY nfam DESC, best DESC, mindf, s1) AS rk
            FROM (
                SELECT rid, s1, string_agg(fam, ',' ORDER BY fam) AS family, count(*) AS nfam,
                    max(CASE fam {prio} ELSE 0 END) AS best, min(df) AS mindf
                FROM {table} WHERE {cap_cond(caps)}
                GROUP BY rid, s1
            )
        ) WHERE rk <= {top}
    """


def text_sql(split, sources):
    """Return SQL for 'name | address' texts of normalised sources."""
    parts = []
    for source in sources:
        path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
        parts.append(f"""SELECT {ID_EXPR.format(col='entity_id')} AS eid, country,
            coalesce(name_full, '') || ' | ' || coalesce(address, '') AS text FROM read_parquet('{path}')""")
    return " UNION ALL ".join(parts)


def write_output(con, split, out_file):
    """Select the final pairs, attach texts (and labels for train) and write the parquet."""
    con.execute(f"CREATE OR REPLACE TABLE chosen AS {select_sql()}")
    if split == "train":
        con.execute(f"""
            COPY (
                SELECT c.rid, c.s1, c.family,
                    (t.rid IS NOT NULL)::INTEGER AS label,
                    (a.rid IS NOT NULL) AS assigned,
                    q.text AS q_text, s.text AS s_text, q.country
                FROM chosen c
                SEMI JOIN slice USING (s1)
                LEFT JOIN truth t USING (rid, s1)
                LEFT JOIN assigned a USING (rid)
                JOIN ({text_sql(split, (2, 3))}) q ON q.eid = c.rid
                JOIN ({text_sql(split, (1,))}) s ON s.eid = c.s1
                ORDER BY c.rid, c.s1
            ) TO '{out_file.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)
    else:
        con.execute(f"""
            COPY (
                SELECT c.rid, c.s1, c.family, q.text AS q_text, s.text AS s_text, q.country
                FROM chosen c
                ANTI JOIN assigned USING (rid)
                JOIN ({text_sql(split, (2, 3))}) q ON q.eid = c.rid
                JOIN ({text_sql(split, (1,))}) s ON s.eid = c.s1
                ORDER BY c.rid, c.s1
            ) TO '{out_file.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)
    n = con.execute(f"SELECT count(*), count(DISTINCT rid) FROM read_parquet('{out_file.as_posix()}')").fetchone()
    print(f"wrote {out_file} ({n[0]:,} pairs for {n[1]:,} records)", flush=True)


def generate(split, out_file, temp, matching_file=MATCHING_FILE, valid_assigned=VALID_ASSIGNED,
             valid_scores=VALID_SCORES):
    """Run the full rescue candidate generation for one split."""
    temp.mkdir(parents=True, exist_ok=True)
    con = connect(temp / f"rescue_{split}.duckdb", temp / "spill")
    try:
        build_s1_keys(con, split)
        load_scope(con, split, matching_file, valid_assigned, valid_scores)
        if split == "train":
            slice_pairs(con, split)
        final_pairs(con, split)
        address_filter(con, split)
        write_output(con, split, out_file)
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)


def main():
    """Command line entry: generate rescue candidates for a split."""
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    if split not in OUT_FILES:
        raise SystemExit(f"unknown split {split}; choose from {sorted(OUT_FILES)}")
    matching_file = Path(sys.argv[2]) if len(sys.argv) > 2 else MATCHING_FILE
    start = time.time()
    RESCUE_DIR.mkdir(parents=True, exist_ok=True)
    generate(split, OUT_FILES[split], TEMP_DIR / f"rescue_{split}", matching_file)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
