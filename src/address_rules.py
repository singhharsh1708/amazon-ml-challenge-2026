"""Exact-address addition rule for France.

Finds source 2/3 records that were not blocked with a source 1 entity but share a unique
house-numbered address key with it, and whose name is either an acronym of the source 1 name or
made only of out-of-vocabulary tokens. Library module used by predict_submission.py and
france_rules.py.
"""

from config import DATA_DIR

NORM_DIR = DATA_DIR / "norm"

RULE_COUNTRIES = ("france",)
REGION = r"\b(hauts de france|nouvelle aquitaine|pays de la loire|nord|gironde|loire atlantique|pas de calais)\b"
DROP = [
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "gmbh", "inc", "llc", "ltd", "pvt", "private",
    "limited", "llp", "lp", "corp", "co", "company", "plc", "pllc", "pc", "incorporated",
    "corporation", "de", "du", "des", "la", "le", "les", "et", "and", "the", "of", "d", "l",
]

ID_EXPR = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
KEY = (
    "array_to_string(list_sort(list_filter(string_split("
    f"regexp_replace(coalesce(address, ''), '{REGION}', ' ', 'g'), ' '), x -> x <> '')), ' ')"
)


def exact_address_matches(con, split, cand_file):
    """Return (rid, s1) pairs added by the unique exact-address rule for RULE_COUNTRIES."""
    drop = "[" + ", ".join(f"'{w}'" for w in DROP) + "]"
    countries = "(" + ", ".join(f"'{c}'" for c in RULE_COUNTRIES) + ")"

    def keyed(source):
        """Return SQL that selects a normalised source with its sorted address key."""
        path = (NORM_DIR / f"{split}_s{source}.parquet").as_posix()
        return (
            f"SELECT {ID_EXPR} AS eid, country, name_full, {KEY} AS k FROM read_parquet('{path}') "
            f"WHERE country IN {countries}"
        )

    con.execute(f"""
        CREATE OR REPLACE TABLE rule_s1 AS
        SELECT country, k, any_value(eid) AS s1, any_value(name_full) AS s_name
        FROM ({keyed(1)}) WHERE regexp_matches(k, '[0-9]')
        GROUP BY country, k HAVING count(*) = 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE rule_vocab AS
        SELECT DISTINCT country, t FROM (
            SELECT country, unnest(string_split(name_full, ' ')) AS t FROM ({keyed(1)})
        ) WHERE t <> ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE rule_pairs AS
        SELECT r.eid AS rid, u.s1, r.country,
            list_filter(string_split(r.name_full, ' '), x -> x <> '' AND NOT list_contains({drop}, x)) AS qt,
            array_to_string(list_transform(
                list_filter(string_split(u.s_name, ' '), x -> x <> '' AND NOT list_contains({drop}, x)),
                x -> x[1]), '') AS initials
        FROM (({keyed(2)}) UNION ALL ({keyed(3)})) r
        JOIN rule_s1 u USING (country, k)
        WHERE regexp_matches(r.k, '[0-9]')
          AND NOT EXISTS (
              SELECT 1 FROM read_parquet('{cand_file.as_posix()}') c WHERE c.rid = r.eid AND c.s1 = u.s1
          )
    """)
    return con.execute("""
        WITH oov AS (
            SELECT rid, s1, bool_and(v.t IS NULL) AS all_oov
            FROM (SELECT rid, s1, country, unnest(qt) AS t FROM rule_pairs) p
            LEFT JOIN rule_vocab v USING (country, t)
            GROUP BY rid, s1
        )
        SELECT p.rid, p.s1
        FROM rule_pairs p LEFT JOIN oov o USING (rid, s1)
        WHERE (len(p.qt) = 1 AND regexp_full_match(p.qt[1], '[a-z]{2,4}') AND starts_with(p.initials, p.qt[1]))
           OR coalesce(o.all_oov, false)
    """).fetchnumpy()
