"""France-specific proposal, review and partial-filter rules.

France records have a higher decoy rate, so their assignments are checked by name and address
classes. propose() suggests assignments for unassigned France records (rules A1, A2, A3, A3b,
A5, N1, N2, R5). review() removes suspicious assigned pairs (business-type word swaps,
"compagnie" swaps, duplicated "france") and keeps the proposals that pass extra checks.
partial_filter() compares a new France assignment set against a previous one, drops new pairs
that change the address and restores previous pairs that the classes judge as the same entity.

Input: data/norm/test_s*.parquet, the final prediction parquet with p and p_guard,
the current matching_results.tsv, and data/candidates/test.parquet.
Output: data/france_changes/{removals,additions}.parquet, or a partial-filter parquet.
Run from src/:
  python france_rules.py [scores.parquet] [matching.tsv] [out_dir]
  python france_rules.py partial --new N --previous P --scores S --base B --out O [--norm-dir D]
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from address_rules import DROP, KEY, REGION
from config import DATA_DIR, OUTPUT_DIR, TEST_DIR

NORM_GLOB = DATA_DIR / "norm" / "test_s*.parquet"
CAND_FILE = DATA_DIR / "candidates" / "test.parquet"
SCORES_FILE = DATA_DIR / "cross_encoder" / "test_predictions_final.parquet"
ASSIGNMENT_FILE = OUTPUT_DIR / "matching_results.tsv"
OUT_DIR = DATA_DIR / "france_changes"

ID_EXPR = "cast(substr({col}, 2, 1) as bigint) * 10000000000 + cast(substr({col}, 4) as bigint)"
SOURCE_DIV = 10000000000
PARTS = 8

LEGAL = {
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "gmbh", "inc", "llc", "ltd", "pvt", "private",
    "limited", "llp", "lp", "corp", "co", "company", "plc", "pllc", "pc", "incorporated", "corporation",
}
ARTICLES = {"de", "du", "des", "la", "le", "les", "et", "and", "the", "of", "d", "l", "a"}
ALIASES = {"formerly", "aka", "fka", "nee", "known", "as", "com", "www", "dba"}
HONORIFICS = {"ms", "m", "s", "shri", "sri", "smt", "mr", "dr"}
PROPOSAL_NOISE = LEGAL | ARTICLES | ALIASES | HONORIFICS
REGION_RE = re.compile(REGION)
BISTER = {"bis", "ter", "quater", "b", "t"}

REVIEW_STOP = {
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "eirl", "gmbh", "inc", "llc", "ltd", "pvt",
    "private", "limited", "llp", "lp", "corp", "co", "plc", "pllc", "pc", "incorporated", "corporation",
    "de", "du", "des", "la", "le", "les", "et", "and", "the", "of", "d", "l", "a", "ms", "shri", "sri",
    "smt", "mr", "dr", "m", "s",
}
REVIEW_SKIP = {"bis", "ter", "b", "t", "ndeg"}
REVIEW_REGION = (
    r"\b(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|france)\b"
)
NUMBER_SUFFIX = r"\b([0-9]+)(bis|ter|[a-df-z])\b"

SAME_ADDRESS = ("same", "region", "typo", "subset", "bister")
DELTAS = (1, 2, 3, 4, 5, 7, 9, 11, 13, 21)
ADD_NOISE = ["fils", "cie", "associes", "services", "service", "compagnie", "frs", "freres", "st", "societe", "centre", "partners"]
DUAL_WORDS = ["groupe", "france", "developpement"]
KEEP_NOISE = ["fils", "cie", "groupe", "services", "developpement", "associes", "france", "frs", "freres", "st"]
TYPE_EXCLUDE = KEEP_NOISE + ["cb", "compagnie"]
TYPE_MIN = 150
RARE_DF = 1000
PRIORITY = ("A3", "A3b", "R5", "A5", "N1", "A1", "A2", "N2")
TABLES = (
    "fr_names", "fr_assigned", "fr_scores", "fr_rec", "fr_prop_pairs", "fr_prop", "fr_s1key", "fr_rules",
    "fr_r5_pairs", "fr_proposed", "fr_review_pairs", "fr_review_cls", "fr_review", "fr_review_s1",
    "fr_removals", "fr_additions",
)
PARTIAL_THRESHOLD = 0.85
PARTIAL_NOISE = set(KEEP_NOISE) | {"compagnie", "et", "cb"}
PARTIAL_SAME_NAME = ("eq", "typo", "spaceless", "acr")
PARTIAL_SAME_ADDRESS = {"same", "same_ns", "same_num_wtypo", "missing", "s_missing", "numdrop"}
PARTIAL_DIFF_ADDRESS = ("numchg1", "numchg1_wtypo", "numchg", "street")
PARTIAL_TABLES = (
    "pt_names", "pt_new", "pt_prev", "pt_pairs", "pt_cls", "pt_new_only", "pt_prev_only", "pt_drop", "pt_kept",
    "pt_restore_all", "pt_restore",
)


def sql_list(words):
    """Return a DuckDB list literal of quoted words."""
    return "[" + ", ".join(f"'{w}'" for w in words) + "]"


def sql_in(values):
    """Return a SQL IN tuple of Python literal values."""
    return "(" + ", ".join(repr(v) for v in values) + ")"


def proposal_name(qn, sn):
    """Classify how the query name differs from the source 1 name; returns (class, added, removed)."""
    qa, sa = (qn or "").split(), (sn or "").split()
    q = [t for t in qa if t not in PROPOSAL_NOISE and len(t) > 1]
    s = [t for t in sa if t not in PROPOSAL_NOISE and len(t) > 1]
    if not q:
        q = [t for t in qa if t not in LEGAL and t not in ALIASES]
    if (not q and not s) or qa == sa:
        return "same", "", ""
    qs, ss = set(q), set(s)
    if qs == ss:
        return ("legal" if set(qa) != set(sa) else "order"), "", ""
    qj, sj = "".join(q), "".join(s)
    if qj == sj or (len(q) == 1 and len(s) >= 2 and sj and sj in q[0]):
        return "spaceless", "", ""
    if len(q) == 1 and 2 <= len(q[0]) <= 6 and len(s) >= 2:
        ini = "".join(t[0] for t in s)
        ini_all = "".join(t[0] for t in sa if t not in LEGAL)
        if q[0] in (ini, ini_all) or ini.startswith(q[0]):
            return "acronym", "", ""
    add = [t for t in q if t not in ss]
    rm = [t for t in s if t not in qs]
    typo = False
    for a in list(add):
        match = next((
            r for r in rm
            if fuzz.ratio(a, r) >= 75 or (len(a) >= 4 and len(r) >= 4 and (a.startswith(r) or r.startswith(a)))
        ), None)
        if match is not None:
            add.remove(a)
            rm.remove(match)
            typo = True
    if add and not rm:
        add = [a for a in add if not (len(a) >= 4 and a in sj)]
    if rm and not add:
        rm = [r for r in rm if not (len(r) >= 4 and r in qj)]
    if not add and not rm:
        return ("typo" if typo else "legal"), "", ""
    if not rm:
        return "add", " ".join(add), ""
    if not add:
        return "rm", "", " ".join(rm)
    return "swap", " ".join(add), " ".join(rm)


def proposal_address(qa, sa):
    """Classify how the query address differs from the source 1 address; returns (class, number delta)."""
    if not qa:
        return "missing", None
    if not sa:
        return "s_missing", None
    if sorted(qa.split()) == sorted(sa.split()):
        return "same", None
    q2, s2 = REGION_RE.sub(" ", qa).split(), REGION_RE.sub(" ", sa).split()
    if sorted(q2) == sorted(s2):
        return "region", None
    qn = [t for t in q2 if t.isdigit()]
    sn = [t for t in s2 if t.isdigit()]
    qw = sorted(t for t in q2 if not t.isdigit() and t not in BISTER)
    sw = sorted(t for t in s2 if not t.isdigit() and t not in BISTER)
    delta = None
    if qn and sn and qn[0] != sn[0]:
        try:
            delta = int(qn[0]) - int(sn[0])
        except ValueError:
            delta = None
    wsim = fuzz.token_sort_ratio(" ".join(qw), " ".join(sw))
    if sorted(qn) == sorted(sn):
        if qw == sw:
            return ("same" if sorted(t for t in q2 if t in BISTER) == sorted(t for t in s2 if t in BISTER) else "bister"), None
        if set(qw) <= set(sw) or set(sw) <= set(qw):
            return "subset", None
        return ("typo" if wsim >= 85 else "street"), None
    if not qn and sn:
        return ("num_dropped" if wsim >= 85 else "other"), None
    if qn and not sn:
        return ("num_added" if wsim >= 85 else "other"), None
    if set(qn) & set(sn) and wsim >= 85:
        return "num_partial", delta
    return ("num_change" if wsim >= 85 else "other"), delta


def proposal_classes(qn, sn, qa, sa):
    """Return the concatenated name and address classes used by the proposal rules."""
    return proposal_name(qn, sn) + proposal_address(qa, sa)


def review_tokens(name):
    """Return name tokens without review stop words and single letters."""
    return [t for t in (name or "").split() if t not in REVIEW_STOP and len(t) > 1]


def review_name(qn, sn):
    """Classify the name difference for review; returns (class, added words, removed words)."""
    q, s = review_tokens(qn), review_tokens(sn)
    if not q or not s:
        return "empty", "", ""
    qs, ss = set(q), set(s)
    if qs == ss:
        return "eq", "", ""
    if "".join(q) == "".join(s):
        return "spaceless", "", ""
    if len(q) == 1 and 2 <= len(q[0]) <= 5 and len(s) >= 2:
        if q[0] in ("".join(t[0] for t in s), "".join(t[0] for t in (sn or "").split() if t)):
            return "acr", "", ""
    add, rm = sorted(qs - ss), sorted(ss - qs)
    typo = False
    for a in list(add):
        match = next((r for r in rm if fuzz.ratio(a, r) >= 80), None)
        if match is not None:
            add.remove(a)
            rm.remove(match)
            typo = True
    if not add and not rm:
        return "typo", "", ""
    if not (qs & ss) and not typo:
        return "disjoint", " ".join(add), " ".join(rm)
    if not rm:
        return "add", " ".join(add), ""
    if not add:
        return "rm", "", " ".join(rm)
    if len(add) == 1 and len(rm) == 1:
        return "swap1", add[0], rm[0]
    return "other", " ".join(add), " ".join(rm)


def review_address(qk, sk):
    """Classify the address-key difference for review; returns (class, number delta)."""
    if not qk:
        return "missing", None
    if not sk:
        return "s_missing", None
    if qk == sk:
        return "same", None
    qt, st = qk.split(), sk.split()
    qn = [t for t in qt if t.isdigit()]
    sn = [t for t in st if t.isdigit()]
    qw = sorted(t for t in qt if not t.isdigit() and t not in REVIEW_SKIP)
    sw = sorted(t for t in st if not t.isdigit() and t not in REVIEW_SKIP)
    if qw == sw:
        if sorted(qn) == sorted(sn):
            return "same_ns", None
        if not qn and sn:
            return "numdrop", None
        if len(qn) == 1 and len(sn) == 1:
            return "numchg1", int(qn[0]) - int(sn[0])
        return "numchg", None
    ws = fuzz.token_sort_ratio(" ".join(qw), " ".join(sw))
    if sorted(qn) == sorted(sn) and qn:
        return ("same_num_wtypo" if ws >= 85 else "street"), None
    if len(qn) == 1 and len(sn) == 1 and ws >= 85:
        return "numchg1_wtypo", int(qn[0]) - int(sn[0])
    return "other", None


def review_classes(qn, sn, qk, sk):
    """Return the concatenated name and address classes used by the review rules."""
    return review_name(qn, sn) + review_address(qk, sk)


def review_key(col):
    """Return SQL for a sorted address key without region names and with number suffixes split."""
    text = f"regexp_replace(coalesce({col}, ''), '{REVIEW_REGION}', ' ', 'g')"
    text = f"regexp_replace({text}, '{NUMBER_SUFFIX}', '\\1 \\2', 'g')"
    return f"array_to_string(list_sort(list_filter(string_split({text}, ' '), x -> x <> '' and x <> 'ndeg')), ' ')"


def classify(con, source, target, classes):
    """Apply a Python classifier to a pair table in parts and store the classes in a DuckDB table."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {target} (
            rid BIGINT, s1 BIGINT, nc VARCHAR, nadd VARCHAR, nrm VARCHAR, ac VARCHAR, delta BIGINT
        )
    """)
    for part in range(PARTS):
        tab = con.execute(f"SELECT rid, s1, qn, sn, qa, sa FROM {source} WHERE rid % {PARTS} = {part}").to_arrow_table()
        cols = [tab.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")]
        rows = [classes(*x) for x in zip(*cols)]
        out = pa.table({
            "rid": tab.column("rid"), "s1": tab.column("s1"),
            "nc": [r[0] for r in rows], "nadd": [r[1] for r in rows], "nrm": [r[2] for r in rows],
            "ac": [r[3] for r in rows], "delta": pa.array([r[4] for r in rows], pa.int64()),
        })
        con.register("fr_part", out)
        con.execute(f"INSERT INTO {target} SELECT * FROM fr_part")
        con.unregister("fr_part")


def assignment_sql(path):
    """Return SQL reading (rid, s1) assignments from a matching TSV or a parquet file."""
    if path.suffix == ".tsv":
        return f"""
            SELECT {ID_EXPR.format(col='m')} AS rid, {ID_EXPR.format(col='source1_entity_id')} AS s1 FROM (
                SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS m
                FROM read_csv('{path.as_posix()}', delim = '\t', header = true, all_varchar = true, quote = '', escape = '')
                WHERE coalesce(matched_entity_ids, '') <> ''
            ) WHERE m <> ''
        """
    return f"SELECT rid, s1 FROM read_parquet('{path.as_posix()}')"


def raw_names(sources):
    """Return SQL for lower-cased raw business names of France records in the given test sources."""
    files = "[" + ", ".join(f"'{(TEST_DIR / f'test_source{s}.tsv').as_posix()}'" for s in sources) + "]"
    return f"""
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, lower(business_name) AS name
        FROM read_csv({files}, delim = '\t', header = true, all_varchar = true, quote = '', escape = '')
        WHERE lower(country) = 'france'
    """


def load_inputs(con, scores_file, assignment_file):
    """Load France names, current assignments and pair scores into temp tables."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_names AS
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, name_full, name_core, address
        FROM read_parquet('{NORM_GLOB.as_posix()}') WHERE country = 'france'
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_assigned AS
        SELECT rid, s1 FROM ({assignment_sql(assignment_file)})
        WHERE rid IN (SELECT eid FROM fr_names WHERE eid // {SOURCE_DIV} > 1)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_scores AS
        SELECT rid, s1, p, p_guard FROM read_parquet('{scores_file.as_posix()}')
        WHERE rid IN (SELECT eid FROM fr_names WHERE eid // {SOURCE_DIV} > 1)
    """)


def propose(con):
    """Classify France records against their best candidate and store rule-based proposals in fr_proposed.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_rec AS
        SELECT n.eid AS rid, a.s1 AS asg, b.s1a, b.pa, b.g_s1, b.g_p, coalesce(a.s1, b.s1a) AS ps1
        FROM fr_names n
        LEFT JOIN fr_assigned a ON a.rid = n.eid
        LEFT JOIN (
            SELECT rid, first(s1 ORDER BY p DESC, s1) AS s1a, max(p) AS pa,
                first(s1 ORDER BY p_guard DESC, s1) FILTER (WHERE p_guard IS NOT NULL) AS g_s1, max(p_guard) AS g_p
            FROM fr_scores GROUP BY rid
        ) b ON b.rid = n.eid
        WHERE n.eid // {SOURCE_DIV} > 1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE fr_prop_pairs AS
        SELECT r.rid, r.ps1 AS s1, q.name_full AS qn, s.name_full AS sn, q.address AS qa, s.address AS sa
        FROM fr_rec r JOIN fr_names q ON q.eid = r.rid JOIN fr_names s ON s.eid = r.ps1
    """)
    classify(con, "fr_prop_pairs", "fr_prop", proposal_classes)
    akey = (
        f"array_to_string(list_sort(list_filter(string_split(regexp_replace(address, '{REGION}', ' ', 'g'), ' '), "
        "t -> t <> '' AND t <> 'ndeg')), ' ')"
    )
    skey = (
        f"array_to_string(list_sort(list_filter(string_split(regexp_replace(address, '{REGION}', ' ', 'g'), ' '), "
        "t -> t <> '' AND t <> 'ndeg' AND NOT regexp_full_match(t, '[0-9]+') AND t NOT IN ('bis', 'ter'))), ' ')"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_s1key AS
        SELECT eid, name_full, name_core, {akey} AS akey, {skey} AS skey,
            regexp_extract(address, '\\b([0-9]+)\\b', 1) AS num
        FROM fr_names WHERE eid // {SOURCE_DIV} = 1
    """)
    same = sql_in(SAME_ADDRESS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_rules AS
        WITH s1u AS (SELECT eid, count(*) OVER (PARTITION BY akey) AS n_at_addr FROM fr_s1key),
        twins AS (
            SELECT eid FROM (
                SELECT eid, count(*) OVER w AS g, count(DISTINCT num) OVER w AS gn
                FROM fr_s1key WHERE skey <> '' WINDOW w AS (PARTITION BY name_core, skey)
            ) WHERE g >= 2 AND gn >= 2
        ),
        df AS (
            SELECT t, count(*) AS df FROM (SELECT DISTINCT eid, unnest(string_split(name_full, ' ')) AS t FROM fr_s1key)
            GROUP BY t
        ),
        b AS (
            SELECT r.rid, r.ps1 AS s1, r.asg, r.s1a, r.pa, r.g_s1, r.g_p, c.nc, c.ac, c.delta,
                list_filter(string_split(c.nadd, ' '), x -> x <> '') AS aw,
                list_filter(string_split(c.nrm, ' '), x -> x <> '') AS rw,
                u.n_at_addr, r.ps1 IN (SELECT eid FROM twins) AS is_twin
            FROM fr_rec r JOIN fr_prop c ON c.rid = r.rid AND c.s1 = r.ps1 LEFT JOIN s1u u ON u.eid = r.ps1
        ),
        mr AS (
            SELECT rid, min(coalesce(d.df, 0)) AS min_rm_df
            FROM (SELECT rid, unnest(rw) AS t FROM b WHERE len(rw) > 0) x LEFT JOIN df d USING (t)
            GROUP BY rid
        ),
        bb AS (SELECT b.*, coalesce(mr.min_rm_df, 99999) AS min_rm_df FROM b LEFT JOIN mr USING (rid))
        SELECT rid, s1, CASE
            WHEN asg IS NULL AND s1a = g_s1 AND pa >= 0.85 AND g_p < 0.85
                AND nc IN ('legal', 'order', 'rm', 'typo', 'spaceless', 'same') AND ac IN {same} THEN 'A3'
            WHEN asg IS NULL AND nc = 'acronym' AND ac IN {same} AND n_at_addr = 1 THEN 'A5'
            WHEN asg IS NULL AND s1a = g_s1 AND nc IN ('same', 'order', 'typo', 'spaceless') AND ac = 'num_change'
                AND NOT is_twin AND (delta < 0 OR delta NOT IN {sql_in(DELTAS)}) THEN 'N1'
            WHEN asg IS NULL AND s1a = g_s1 AND nc = 'legal' AND ac = 'num_change'
                AND NOT is_twin AND delta < 0 AND abs(delta) IN {sql_in(DELTAS)} THEN 'N2'
            WHEN asg IS NULL AND s1a = g_s1 AND ac IN {same} AND nc IN ('add', 'swap') AND len(aw) > 0
                AND list_has_all({sql_list(ADD_NOISE)}, aw) AND min_rm_df >= {RARE_DF} THEN 'A1'
            WHEN asg IS NULL AND s1a = g_s1 AND ac IN {same} AND nc IN ('add', 'swap') AND len(aw) > 0
                AND list_has_all({sql_list(ADD_NOISE + DUAL_WORDS)}, aw) AND min_rm_df >= {RARE_DF} THEN 'A2'
            WHEN asg IS NULL AND s1a = g_s1 AND pa >= 0.5 AND pa < 0.85 AND ac IN {same}
                AND nc IN ('same', 'legal', 'order', 'typo', 'spaceless', 'rm') AND n_at_addr = 1 THEN 'A3b'
            END AS rule
        FROM bb
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_r5_pairs AS
        WITH k AS (SELECT eid, eid // {SOURCE_DIV} AS src, name_full, {KEY} AS k FROM fr_names),
        rs1 AS (
            SELECT k, any_value(eid) AS s1, any_value(name_full) AS s_name FROM k
            WHERE src = 1 AND regexp_matches(k, '[0-9]') GROUP BY k HAVING count(*) = 1
        ),
        voc AS (SELECT DISTINCT unnest(string_split(name_full, ' ')) AS t FROM k WHERE src = 1),
        rp AS (
            SELECT r.eid AS rid, u.s1,
                list_filter(string_split(r.name_full, ' '), x -> x <> '' AND NOT list_contains({sql_list(DROP)}, x)) AS qt,
                array_to_string(list_transform(
                    list_filter(string_split(u.s_name, ' '), x -> x <> '' AND NOT list_contains({sql_list(DROP)}, x)),
                    x -> x[1]), '') AS initials
            FROM k r JOIN rs1 u USING (k) WHERE r.src > 1 AND regexp_matches(r.k, '[0-9]')
        ),
        oov AS (
            SELECT rid, s1, bool_and(v.t IS NULL) AS all_oov
            FROM (SELECT rid, s1, unnest(qt) AS t FROM rp) p LEFT JOIN voc v USING (t) GROUP BY rid, s1
        )
        SELECT p.rid, p.s1 FROM rp p LEFT JOIN oov o USING (rid, s1)
        WHERE ((len(p.qt) = 1 AND regexp_full_match(p.qt[1], '[a-z]{{2,4}}') AND starts_with(p.initials, p.qt[1]))
            OR coalesce(o.all_oov, false))
          AND p.rid NOT IN (SELECT rid FROM fr_assigned)
    """)
    order = "CASE rule " + " ".join(f"WHEN '{r}' THEN {i}" for i, r in enumerate(PRIORITY)) + " END"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_proposed AS
        SELECT rid, arg_min(s1, pr) AS s1, arg_min(rule, pr) AS rule FROM (
            SELECT rid, s1, rule, {order} AS pr FROM (
                SELECT rid, s1, rule FROM fr_rules WHERE rule IS NOT NULL
                UNION ALL
                SELECT rid, s1, 'R5' FROM fr_r5_pairs
                WHERE (rid, s1) IN (
                    SELECT (c.rid, c.s1) FROM read_parquet('{CAND_FILE.as_posix()}') c
                    WHERE c.rid IN (SELECT rid FROM fr_r5_pairs)
                )
            )
        ) GROUP BY rid
    """)


def review(con):
    """Classify assigned and proposed France pairs and build the fr_removals and fr_additions tables."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_review_pairs AS
        WITH f AS (
            SELECT sc.rid, sc.s1, sc.p, a.s1 IS NOT NULL AS asg_this,
                row_number() OVER (PARTITION BY sc.rid ORDER BY (a.s1 IS NOT NULL) DESC, sc.p DESC, sc.s1) = 1 AS top
            FROM fr_scores sc LEFT JOIN fr_assigned a ON a.rid = sc.rid AND a.s1 = sc.s1
            WHERE sc.s1 IN (SELECT eid FROM fr_names WHERE eid // {SOURCE_DIV} = 1)
        )
        SELECT f.rid, f.s1, f.p, f.asg_this, f.top,
            q.name_full AS qn, s.name_full AS sn, {review_key('q.address')} AS qa, {review_key('s.address')} AS sa
        FROM f
        JOIN fr_names q ON q.eid = f.rid JOIN fr_names s ON s.eid = f.s1
        LEFT JOIN (SELECT rid, s1 FROM fr_proposed WHERE rule IN ('A1', 'A2')) x ON x.rid = f.rid AND x.s1 = f.s1
        WHERE f.top OR f.asg_this OR x.rid IS NOT NULL
    """)
    classify(con, "fr_review_pairs", "fr_review_cls", review_classes)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE fr_review AS
        SELECT r.rid, r.s1, r.p, r.asg_this, r.top, c.nc, c.nadd, c.nrm, c.delta, CASE
            WHEN c.ac IN ('same', 'same_ns', 'same_num_wtypo') THEN 'same'
            WHEN c.ac IN ('numchg1', 'numchg1_wtypo') THEN 'numchg1'
            ELSE c.ac END AS ag
        FROM fr_review_pairs r JOIN fr_review_cls c USING (rid, s1)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_review_s1 AS
        SELECT eid, count(*) OVER (PARTITION BY ak) AS n_at_addr
        FROM (SELECT eid, {review_key('address')} AS ak FROM fr_names WHERE eid // {SOURCE_DIV} = 1)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_removals AS
        WITH types AS (
            SELECT nadd FROM fr_review
            WHERE top AND ag = 'numchg1' AND nc = 'swap1' AND NOT list_contains({sql_list(TYPE_EXCLUDE)}, nadd)
                AND regexp_full_match(nadd, '[a-z]{{3,}}')
            GROUP BY nadd HAVING count(*) >= {TYPE_MIN}
        ),
        dup AS (SELECT rid, s1 FROM fr_review WHERE asg_this AND nc = 'eq' AND delta IN {sql_in(DELTAS)})
        SELECT rid, s1, 'R1_typeswap' AS rule FROM fr_review
        WHERE top AND asg_this AND ag = 'same' AND nc = 'swap1' AND nadd IN (SELECT nadd FROM types)
        UNION ALL
        SELECT rid, s1, 'R1_compagnie' FROM fr_review
        WHERE top AND asg_this AND ag = 'same' AND nc = 'swap1' AND nadd = 'compagnie' AND nrm <> 'cie'
        UNION ALL
        SELECT d.rid, d.s1, 'R2_dupfrance' FROM dup d
        JOIN ({raw_names((2, 3))}) q ON q.eid = d.rid
        JOIN ({raw_names((1,))}) s ON s.eid = d.s1
        WHERE len(regexp_extract_all(q.name, 'france')) > len(regexp_extract_all(s.name, 'france'))
            AND len(regexp_extract_all(s.name, 'france')) >= 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fr_additions AS
        WITH props AS (
            SELECT x.rid, x.s1, x.rule, r.nadd, r.nrm, sc.p, n.n_at_addr,
                list_filter(string_split(coalesce(r.nadd, ''), ' '), w -> w <> '') AS aw
            FROM fr_proposed x
            LEFT JOIN fr_scores sc ON sc.rid = x.rid AND sc.s1 = x.s1
            LEFT JOIN fr_review r ON r.rid = x.rid AND r.s1 = x.s1
            LEFT JOIN fr_review_s1 n ON n.eid = x.s1
        )
        SELECT rid, s1, CASE
            WHEN rule IN ('A1', 'A2') THEN 'A1A2_noise' WHEN rule IN ('R5', 'A5') THEN 'R5A5' ELSE rule END AS rule
        FROM props
        WHERE (rule IN ('A1', 'A2')
                AND ((len(aw) > 0 AND list_has_all({sql_list(KEEP_NOISE)}, aw)) OR (nadd = 'compagnie' AND nrm = 'cie'))
                AND NOT (n_at_addr > 1 AND p < 0.1))
            OR (rule IN ('R5', 'A5') AND p >= 0.1)
            OR rule = 'A3'
            OR (rule IN ('N1', 'N2') AND p >= 0.5)
    """)


def france_changes(con, scores_file, assignment_file):
    """Run proposal and review and return (removals, additions) as numpy dicts."""
    load_inputs(con, Path(scores_file), Path(assignment_file))
    propose(con)
    review(con)
    for kind in ("removals", "additions"):
        counts = con.execute(f"SELECT rule, count(*) FROM fr_{kind} GROUP BY rule ORDER BY rule").fetchall()
        print(f"france {kind}: " + ", ".join(f"{rule} {n:,}" for rule, n in counts), flush=True)
    removals = con.execute("SELECT rid, s1, rule FROM fr_removals ORDER BY rid").fetchnumpy()
    additions = con.execute("SELECT rid, s1, rule FROM fr_additions ORDER BY rid").fetchnumpy()
    for table in TABLES:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    return removals, additions


def review_verdict(nc, nadd, nrm, ac, delta):
    """Map review classes of a changed pair to a verdict such as same, diff_addr or unsure_*."""
    added = [w for w in (nadd or "").split() if w]
    removed = [w for w in (nrm or "").split() if w]
    if ac in PARTIAL_DIFF_ADDRESS:
        if (nc in PARTIAL_SAME_NAME and ac in ("numchg1", "numchg1_wtypo") and delta is not None
                and abs(delta) not in DELTAS):
            return "unsure_num"
        return "diff_addr"
    if nc in PARTIAL_SAME_NAME:
        return "same" if ac in PARTIAL_SAME_ADDRESS else "unsure_addr"
    if nc in ("add", "swap1", "other", "rm"):
        if added and not all(w in PARTIAL_NOISE for w in added):
            return "diff_typeswap" if removed else "unsure_addword"
        return "same_noise" if ac in PARTIAL_SAME_ADDRESS else "unsure_addr"
    return "unsure_" + nc


def partial_filter(con, new_file, previous_file, scores_file, base_file, norm_glob=None):
    """Merge new and previous France assignments by verdict; returns (pairs, counts)."""
    norm_glob = Path(norm_glob) if norm_glob is not None else NORM_GLOB
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pt_names AS
        SELECT {ID_EXPR.format(col='entity_id')} AS eid, name_full, address
        FROM read_parquet('{norm_glob.as_posix()}') WHERE country = 'france'
    """)
    for table, path in (("pt_new", new_file), ("pt_prev", previous_file)):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {table} AS
            SELECT DISTINCT rid, s1 FROM ({assignment_sql(Path(path))})
            WHERE s1 IN (SELECT eid FROM pt_names WHERE eid // {SOURCE_DIV} = 1)
        """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pt_pairs AS
        SELECT p.rid, p.s1, q.name_full AS qn, s.name_full AS sn,
            {review_key('q.address')} AS qa, {review_key('s.address')} AS sa
        FROM (
            SELECT rid, s1 FROM (SELECT * FROM pt_new ANTI JOIN pt_prev USING (rid, s1))
            UNION ALL
            SELECT rid, s1 FROM (SELECT * FROM pt_prev ANTI JOIN pt_new USING (rid, s1))
        ) p
        JOIN pt_names q ON q.eid = p.rid JOIN pt_names s ON s.eid = p.s1
    """)
    classify(con, "pt_pairs", "pt_cls", review_classes)
    cls = con.execute("SELECT rid, s1, nc, nadd, nrm, ac, delta FROM pt_cls").fetchnumpy()
    verdicts = [review_verdict(*row) for row in zip(
        cls["nc"], cls["nadd"], cls["nrm"], cls["ac"],
        [None if d is None else int(d) for d in cls["delta"].tolist()],
    )]
    con.register("pt_verdict_arrow", pa.table({
        "rid": pa.array(cls["rid"], pa.int64()), "s1": pa.array(cls["s1"], pa.int64()),
        "v": pa.array(verdicts, pa.string()),
    }))
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pt_new_only AS
        SELECT n.rid, n.s1, v.v FROM (SELECT * FROM pt_new ANTI JOIN pt_prev USING (rid, s1)) n
        LEFT JOIN pt_verdict_arrow v USING (rid, s1)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pt_prev_only AS
        SELECT o.rid, o.s1, v.v, CASE
            WHEN EXISTS (SELECT 1 FROM read_parquet('{Path(base_file).as_posix()}') b WHERE b.rid = o.rid AND b.s1 = o.s1)
                THEN 'france_changes'
            WHEN sc.p IS NULL THEN 'not_candidate'
            WHEN sc.p < {PARTIAL_THRESHOLD} THEN 'score'
            WHEN sc.p_guard < {PARTIAL_THRESHOLD} THEN 'guard'
            ELSE 'other' END AS why
        FROM (SELECT * FROM pt_prev ANTI JOIN pt_new USING (rid, s1)) o
        LEFT JOIN pt_verdict_arrow v USING (rid, s1)
        LEFT JOIN (
            SELECT rid, s1, p, p_guard FROM read_parquet('{Path(scores_file).as_posix()}')
            WHERE rid IN (SELECT rid FROM pt_prev)
        ) sc USING (rid, s1)
    """)
    con.unregister("pt_verdict_arrow")
    con.execute("CREATE OR REPLACE TEMP TABLE pt_drop AS SELECT rid, s1 FROM pt_new_only WHERE v = 'diff_addr'")
    con.execute("CREATE OR REPLACE TEMP TABLE pt_kept AS SELECT rid, s1 FROM pt_new ANTI JOIN pt_drop USING (rid, s1)")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pt_restore_all AS
        SELECT rid, s1 FROM pt_prev_only WHERE why <> 'france_changes' AND v IN ('same', 'same_noise')
    """)
    con.execute("CREATE OR REPLACE TEMP TABLE pt_restore AS SELECT * FROM pt_restore_all ANTI JOIN pt_kept USING (rid)")
    result = con.execute("""
        SELECT rid, s1 FROM pt_kept UNION ALL SELECT rid, s1 FROM pt_restore ORDER BY s1, rid
    """).fetchnumpy()
    counts = dict(zip(
        ("new", "previous", "new_only", "previous_only", "dropped", "restore_candidates", "restored"),
        con.execute("""
            SELECT (SELECT count(*) FROM pt_new), (SELECT count(*) FROM pt_prev), (SELECT count(*) FROM pt_new_only),
                (SELECT count(*) FROM pt_prev_only), (SELECT count(*) FROM pt_drop),
                (SELECT count(*) FROM pt_restore_all), (SELECT count(*) FROM pt_restore)
        """).fetchone(),
    ))
    counts["final"] = len(result["rid"])
    counts["duplicate_rids"] = counts["final"] - len(np.unique(result["rid"]))
    counts["previous_only_why"] = dict(con.execute(
        "SELECT why, count(*) FROM pt_prev_only GROUP BY why ORDER BY why"
    ).fetchall())
    for table in PARTIAL_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    return result, counts


def partial_main(argv):
    """Command line entry for the partial filter; writes the kept pairs to parquet."""
    parser = argparse.ArgumentParser(prog="france_rules.py partial")
    parser.add_argument("--new", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--norm-dir", default=str(DATA_DIR / "norm"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    temp = out.parent / f"{out.stem}_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2500MB'")
    con.execute("SET threads = 3")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        result, counts = partial_filter(
            con, args.new, args.previous, args.scores, args.base, Path(args.norm_dir) / "test_s*.parquet",
        )
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    pq.write_table(pa.table({
        "rid": pa.array(result["rid"], pa.int64()), "s1": pa.array(result["s1"], pa.int64()),
    }), out)
    why = ", ".join(f"{k} {v:,}" for k, v in counts.pop("previous_only_why").items())
    print("partial: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"partial: previous-only pairs by reason: {why}")
    print(f"wrote {out} ({counts['final']:,} pairs)")


def main():
    """Command line entry: write France removals and additions, or dispatch to the partial filter."""
    if sys.argv[1:2] == ["partial"]:
        partial_main(sys.argv[2:])
        return
    scores = Path(sys.argv[1]) if len(sys.argv) > 1 else SCORES_FILE
    assignment = Path(sys.argv[2]) if len(sys.argv) > 2 else ASSIGNMENT_FILE
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_DIR
    temp = out_dir / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '1200MB'")
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    try:
        removals, additions = france_changes(con, scores, assignment)
    finally:
        con.close()
        shutil.rmtree(temp, ignore_errors=True)
    for name, rows in (("removals", removals), ("additions", additions)):
        path = out_dir / f"{name}.parquet"
        pq.write_table(pa.table({
            "rid": pa.array(rows["rid"], pa.int64()), "s1": pa.array(rows["s1"], pa.int64()),
            "rule": pa.array(rows["rule"], pa.string()),
        }), path)
        print(f"wrote {path} ({len(rows['rid']):,} pairs)")


if __name__ == "__main__":
    main()
