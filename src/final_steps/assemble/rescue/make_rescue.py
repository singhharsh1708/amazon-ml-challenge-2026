"""Generates rescue candidate pairs for source-1 records without a confident match.

Step context: Step 4 input (rescue additions). Rescue candidates for unmatched records scored by ce1/ce2 and a LightGBM gate; produces test_rescue_adds.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/data
    $REPO_DIR/output/v11/matching_results_v11_ce_ef.tsv
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import shutil
import sys
import time
from pathlib import Path

import duckdb

DATA = Path(f"{REPO_DIR}/data")
NORM = DATA / "norm"
CAND = DATA / "candidates"
HERE = Path(WORK_DIR) / "rescue"
WORK = HERE / "work"
HUNT = HERE.parent / "wf" / "hunt-odd-one-out-model"
TRUTH = HUNT / "truth.parquet"
V11_ASSIGNED = HERE / "valid_v11_assigned.parquet"
V11_SCORES = HERE / "valid_v11_scores.parquet"
V11_TEST = Path(f"{REPO_DIR}/output/v11/matching_results_v11_ce_ef.tsv")

ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
LEGAL = ["limited", "ltd", "llc", "llp", "lp", "inc", "pvt", "corp", "corporation", "co",
         "company", "the", "and", "of", "et", "pllc", "plc", "pc"]
FAMILIES = ["cat", "ini", "typo", "aexact", "anum", "anumt", "aalpha", "corenum"]
DF_MAX = 100
GEN_CAPS = {"aexact": 10, "corenum": 10, "cat": 10, "anum": 10, "aalpha": 10, "typo": 10}
CAPS = {"aexact": 3, "corenum": 5, "cat": 10, "anum": [1, 3], "aalpha": 1, "typo": 10}
UNGATED = {"aexact"}
ADDR_CHECK = {"cat", "typo", "corenum"}
PRIORITY = {"aexact": 6, "corenum": 5, "cat": 4, "anum": 3, "aalpha": 2, "typo": 1}
TOP = 5
SLICE_MOD = 20
RECORD_CHUNKS = 10
FINAL_CHUNKS = 20
SLICE_DF = 20
PASS_P = 0.85


def connect(db):
    """Open a DuckDB database with a private temp directory."""
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = WORK / f"tmp_{db.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute(f"""
        SET memory_limit = '1200MB'; SET threads = 2; SET preserve_insertion_order = false;
        SET temp_directory = '{tmp.as_posix()}'; SET max_temp_directory_size = '6GB'
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
    return con, tmp


def base_sql(path, where="", keep=""):
    """Return the SQL that loads the base columns of a pair file."""
    legal = ", ".join(f"'{w}'" for w in LEGAL)
    return f"""
        SELECT * FROM (
        SELECT {ID.format(c='entity_id')} AS eid, country,
            coalesce(name_core, '') AS core,
            list_filter(string_split(coalesce(name_core, ''), ' '), x -> x <> '') AS ct,
            list_filter(string_split(coalesce(name_full, ''), ' '), x -> x <> '' AND x NOT IN ({legal})) AS ft,
            list_distinct(list_filter(string_split(coalesce(address, ''), ' '), x -> x <> '')) AS adt
        FROM read_parquet('{path}') {where}
        ) b {keep}
    """


def enrich_sql(path, where="", keep=""):
    """Return the SQL that joins record text onto a pair file."""
    return f"""
        SELECT *,
            list_filter(adt, x -> regexp_matches(x, '^[a-z]{{4,}}$')) AS al,
            list_distinct(list_filter(list_transform(adt, x -> regexp_extract(x, '([0-9]+)', 1)), d -> d <> '')) AS nums
        FROM ({base_sql(path, where, keep)})
    """


def family_keys(fam, s1_side):
    """Return the SQL key expression for one rescue family."""
    if fam == "cat":
        if s1_side:
            return """list_distinct(list_transform(list_filter(
                list_transform(subsets(ft[1:7]), m -> array_to_string(pick(ft[1:7], m), '')),
                s -> length(s) >= 7), s -> srt(s)))"""
        return """list_distinct(list_concat(
            CASE WHEN length(replace(core, ' ', '')) >= 7 THEN [srt(replace(core, ' ', ''))] ELSE [] END,
            list_transform(list_filter(ct, x -> length(x) >= 8), x -> srt(x))))"""
    if fam == "ini":
        if s1_side:
            return """list_distinct(flatten(list_transform(
                list_filter(range(1, len(ft[1:6]) + 1), i -> length(ft[i]) >= 3),
                i -> list_transform(
                    list_filter(range(1, (1 << len(ft[1:6]))::bigint),
                                m -> bit_count(m) BETWEEN 1 AND 2 AND ((m >> (i - 1)) & 1) = 0),
                    m -> srt(ft[i] || array_to_string(list_transform(pick(ft[1:6], m), x -> x[1]), ''))))))"""
        return """list_distinct(list_concat(
            CASE WHEN length(replace(core, ' ', '')) >= 5 THEN [srt(replace(core, ' ', ''))] ELSE [] END,
            list_transform(list_filter(ct, x -> length(x) >= 5), x -> srt(x))))"""
    if fam == "aexact":
        return """CASE WHEN len(adt) >= 3 AND list_bool_or(list_transform(adt, x -> regexp_matches(x, '[0-9]')))
            THEN [array_to_string(list_sort(adt), ' ')] ELSE [] END"""
    if fam == "anum" or (fam == "anumt" and not s1_side):
        return "list_distinct(flatten(list_transform(nums, d -> list_transform(al, a -> d || '|' || srt(a)))))"
    if fam == "anumt":
        return """list_distinct(flatten(list_transform(
            list_distinct(flatten(list_transform(list_filter(nums, d -> length(d) >= 3),
                d -> [substr(d, 2), substr(d, 1, length(d) - 1)]))),
            d -> list_transform(al, a -> d || '|' || srt(a)))))"""
    if fam == "aalpha":
        return """list_distinct(flatten(list_transform(range(1, len(al[1:8]) + 1),
            i -> list_transform(range(i + 1, len(al[1:8]) + 1),
                j -> least(al[i], al[j]) || '|' || greatest(al[i], al[j])))))"""
    if fam == "corenum":
        return """CASE WHEN length(replace(core, ' ', '')) >= 3
            THEN list_transform(nums, d -> replace(core, ' ', '') || '|' || d) ELSE [] END"""
    raise ValueError(fam)


def keys_sql(fam, path, s1_side, where="", keep="", gate_typo=False):
    """Return the SQL that computes family keys for a pair file."""
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
    """Return the SQL that computes family keys for records of a split."""
    parts = []
    for src in (2, 3):
        path = (NORM / f"{split}_s{src}.parquet").as_posix()
        for fam in fams:
            k = keep[fam] if isinstance(keep, dict) else keep
            parts.append(f"SELECT '{fam}' AS fam, h, eid AS rid FROM ({keys_sql(fam, path, False, where, k, gate_typo)})")
    return " UNION ALL ".join(parts)


def build_s1_keys(con, split, fams):
    """Build the lookup keys of source-1 records for each rescue family."""
    path = (NORM / f"{split}_s1.parquet").as_posix()
    con.execute("CREATE TABLE IF NOT EXISTS s1k (fam VARCHAR, h UBIGINT, s1 BIGINT, df INTEGER)")
    done = {r[0] for r in con.execute("SELECT DISTINCT fam FROM s1k").fetchall()}
    for fam in fams:
        if fam in done:
            continue
        t = time.time()
        con.execute(f"CREATE OR REPLACE TEMP TABLE k AS {keys_sql(fam, path, True)}")
        con.execute(f"""
            INSERT INTO s1k
            SELECT '{fam}', k.h, k.eid, d.df
            FROM k JOIN (SELECT h, count(*) AS df FROM k GROUP BY h HAVING count(*) <= {DF_MAX}) d USING (h)
        """)
        total = con.execute("SELECT count(*) FROM k").fetchone()[0]
        kept = con.execute(f"SELECT count(*) FROM s1k WHERE fam = '{fam}'").fetchone()[0]
        con.execute("DROP TABLE k")
        print(f"s1 keys {fam}: {total:,} generated, {kept:,} with df <= {DF_MAX} in {time.time() - t:.1f}s",
              flush=True)


def load_scope(con, split):
    """Load the records in scope for a split."""
    if split == "train":
        con.execute(f"""
            CREATE OR REPLACE TABLE slice AS
            SELECT {ID.format(c='entity_id')} AS s1 FROM read_parquet('{(NORM / 'train_s1.parquet').as_posix()}')
            WHERE hash({ID.format(c='entity_id')}) % {SLICE_MOD} = 0
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE assigned AS
            SELECT rid FROM read_parquet('{V11_ASSIGNED.as_posix()}')
            UNION
            SELECT rid FROM read_parquet('{V11_SCORES.as_posix()}') GROUP BY rid HAVING max(p2) >= {PASS_P}
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TABLE assigned AS
            SELECT DISTINCT {ID.format(c='m')} AS rid
            FROM (SELECT trim(unnest(string_split(matched_entity_ids, ','))) AS m
                  FROM read_csv('{V11_TEST.as_posix()}', delim='\\t', header=true, all_varchar=true,
                                quote='', escape='')
                  WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '')
            WHERE m <> ''
        """)


def cap_cond(caps):
    """Return the SQL condition that enforces the per-feature caps."""
    conds = []
    for f, c in caps.items():
        if isinstance(c, (list, tuple)):
            conds.append(f"(fam = '{f}' AND (df <= {c[0]} OR (nk >= 2 AND df <= {c[1]})))")
        else:
            conds.append(f"(fam = '{f}' AND df <= {c})")
    return " OR ".join(conds)


def drop_candidates(con, table, split):
    """Remove candidate pairs already present in the submission."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT p.* FROM {table} p
        ANTI JOIN (SELECT rid, s1 FROM read_parquet('{(CAND / f'{split}.parquet').as_posix()}')) c USING (rid, s1)
    """)
    n = con.execute(f"SELECT count(*), count(DISTINCT rid) FROM {table}").fetchone()
    print(f"{table} outside existing candidates: {n[0]:,} rows for {n[1]:,} records", flush=True)


def slice_pairs(con, split, fams):
    """Return the candidate pairs of the rescue slice."""
    con.execute("CREATE TABLE IF NOT EXISTS stage (name VARCHAR)")
    if con.execute("SELECT count(*) FROM stage WHERE name = 'spairs'").fetchone()[0]:
        return
    fam_list = ", ".join(f"'{f}'" for f in fams)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1k_slice AS
        SELECT fam, h, s1, df FROM s1k SEMI JOIN slice USING (s1) WHERE fam IN ({fam_list}) AND df <= {SLICE_DF}
    """)
    con.execute("CREATE OR REPLACE TABLE spairs (rid BIGINT, s1 BIGINT, fam VARCHAR, df INTEGER, nk INTEGER)")
    for chunk in range(RECORD_CHUNKS):
        t = time.time()
        where = f"WHERE hash(entity_id) % {RECORD_CHUNKS} = {chunk}"
        con.execute(f"""
            INSERT INTO spairs
            SELECT q.rid, s.s1, s.fam, min(s.df), count(*)
            FROM ({record_keys_sql(split, fams, where)}) q JOIN s1k_slice s ON s.h = q.h
            GROUP BY q.rid, s.s1, s.fam
        """)
        print(f"slice pairs {chunk + 1}/{RECORD_CHUNKS} in {time.time() - t:.1f}s", flush=True)
    con.execute("DROP TABLE s1k_slice")
    drop_candidates(con, "spairs", split)
    con.execute("INSERT INTO stage VALUES ('spairs')")


def records_sql(split):
    """Return the SQL that loads the normalized records of a split."""
    parts = []
    for src in (2, 3):
        path = (NORM / f"{split}_s{src}.parquet").as_posix()
        parts.append(f"""SELECT {ID.format(c='entity_id')} AS rid, country, coalesce(name_core, '') AS core
            FROM read_parquet('{path}')""")
    return " UNION ALL ".join(parts)


def build_population(con, split, caps):
    """Build the population of source-1 records eligible for rescue."""
    con.execute(f"""
        CREATE OR REPLACE TABLE voc AS
        SELECT DISTINCT country, unnest(string_split(coalesce(name_full, ''), ' ')) AS tok
        FROM read_parquet('{(NORM / f'{split}_s1.parquet').as_posix()}')
    """)
    if split == "train":
        con.execute(f"CREATE OR REPLACE TABLE pop AS SELECT DISTINCT rid FROM spairs WHERE {cap_cond(caps)}")
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


def final_pairs(con, split, caps):
    """Select the final rescue pairs under the caps."""
    fams = list(caps)
    build_population(con, split, caps)
    con.execute(f"CREATE OR REPLACE TEMP TABLE s1kc AS SELECT h, s1, fam, df FROM s1k WHERE {cap_cond(caps)}")
    keep = {f: "SEMI JOIN pop k ON k.rid = b.eid" if f in UNGATED else "SEMI JOIN gate k ON k.rid = b.eid"
            for f in fams}
    con.execute("CREATE OR REPLACE TABLE pairs (rid BIGINT, s1 BIGINT, fam VARCHAR, df INTEGER, nk INTEGER)")
    for chunk in range(FINAL_CHUNKS):
        t = time.time()
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
        print(f"pairs {chunk + 1}/{FINAL_CHUNKS} in {time.time() - t:.1f}s", flush=True)
    con.execute("DROP TABLE qk")
    con.execute("DROP TABLE s1kc")
    drop_candidates(con, "pairs", split)


def address_tokens_sql(split, sources, table, col):
    """Return SQL that tokenizes the addresses of a split."""
    parts = []
    for src in sources:
        path = (NORM / f"{split}_s{src}.parquet").as_posix()
        parts.append(f"""SELECT {ID.format(c='entity_id')} AS eid,
            list_distinct(list_filter(string_split(coalesce(address, ''), ' '),
                                      x -> length(x) >= 3 OR regexp_matches(x, '^[0-9]+$'))) AS adt
            FROM read_parquet('{path}')""")
    return f"SELECT * FROM ({' UNION ALL '.join(parts)}) SEMI JOIN (SELECT DISTINCT {col} AS eid FROM {table}) USING (eid)"


def address_filter(con, split):
    """Drop candidate pairs whose addresses conflict."""
    checked = ", ".join(f"'{f}'" for f in ADDR_CHECK)
    con.execute(f"CREATE OR REPLACE TEMP TABLE qa AS {address_tokens_sql(split, (2, 3), 'pairs', 'rid')}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE sa AS {address_tokens_sql(split, (1,), 'pairs', 's1')}")
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


def select_pairs(con, caps=CAPS, top=TOP, table="fpairs"):
    """Select the top candidate pairs per record under the caps."""
    cond = cap_cond(caps)
    prio = " ".join(f"WHEN '{f}' THEN {p}" for f, p in PRIORITY.items())
    return f"""
        SELECT rid, s1, family FROM (
            SELECT rid, s1, family,
                row_number() OVER (PARTITION BY rid ORDER BY nfam DESC, best DESC, mindf, s1) AS rk
            FROM (
                SELECT rid, s1, string_agg(fam, ',' ORDER BY fam) AS family, count(*) AS nfam,
                    max(CASE fam {prio} ELSE 0 END) AS best, min(df) AS mindf
                FROM {table} WHERE {cond}
                GROUP BY rid, s1
            )
        ) WHERE rk <= {top}
    """


def text_sql(split, sources):
    """Return the SQL that builds the text of each record."""
    parts = []
    for src in sources:
        path = (NORM / f"{split}_s{src}.parquet").as_posix()
        parts.append(f"""SELECT {ID.format(c='entity_id')} AS eid, country,
            coalesce(name_full, '') || ' | ' || coalesce(address, '') AS text FROM read_parquet('{path}')""")
    return " UNION ALL ".join(parts)


def write_output(con, split):
    """Write the rescue candidates of a split to parquet."""
    con.execute(f"CREATE OR REPLACE TABLE chosen AS {select_pairs(con)}")
    if split == "train":
        out = HERE / "valid_rescue.parquet"
        con.execute(f"""
            COPY (
                SELECT c.rid, c.s1, c.family,
                    (t.rid IS NOT NULL)::INTEGER AS label,
                    (a.rid IS NOT NULL) AS assigned,
                    q.text AS q_text, s.text AS s_text, q.country
                FROM chosen c
                SEMI JOIN slice USING (s1)
                LEFT JOIN (SELECT rid, s1 FROM read_parquet('{TRUTH.as_posix()}')) t USING (rid, s1)
                LEFT JOIN assigned a USING (rid)
                JOIN ({text_sql(split, (2, 3))}) q ON q.eid = c.rid
                JOIN ({text_sql(split, (1,))}) s ON s.eid = c.s1
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    else:
        out = HERE / "test_rescue.parquet"
        con.execute(f"""
            COPY (
                SELECT c.rid, c.s1, c.family, q.text AS q_text, s.text AS s_text, q.country
                FROM chosen c
                ANTI JOIN assigned USING (rid)
                JOIN ({text_sql(split, (2, 3))}) q ON q.eid = c.rid
                JOIN ({text_sql(split, (1,))}) s ON s.eid = c.s1
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    n = con.execute(f"SELECT count(*), count(DISTINCT rid) FROM read_parquet('{out.as_posix()}')").fetchone()
    print(f"{out}: {n[0]:,} pairs for {n[1]:,} records", flush=True)


def main():
    """Generate rescue candidates for the split given on the command line."""
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    keep = "--keep" in sys.argv
    fams = FAMILIES if split == "train" else list(GEN_CAPS)
    db = WORK / f"rescue_{split}.duckdb"
    start = time.time()
    con, tmp = connect(db)
    try:
        build_s1_keys(con, split, fams)
        load_scope(con, split)
        if split == "train":
            slice_pairs(con, split, fams)
        if "--select-only" not in sys.argv:
            final_pairs(con, split, GEN_CAPS)
        address_filter(con, split)
        write_output(con, split)
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)
        if not keep:
            db.unlink(missing_ok=True)
            db.with_suffix(".duckdb.wal").unlink(missing_ok=True)
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
