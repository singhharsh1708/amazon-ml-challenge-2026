import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from frc import F, R, SP, ID, con, normdir, work, Timer
import address_rules as AR
import france_rules as FR
from decoy_veto import build_vocab, load_decoy_words, veto_flags

THRESHOLD = 0.85
SIBLING_MAX_P = 0.95
J = f"{SP}/france_audit/judge"


def main(variant, cand):
    T = Timer()
    nd = normdir(variant)
    w = work(variant)
    scores = f"{w}/scores.parquet"
    AR.NORM_DIR = Path(nd)
    FR.NORM_GLOB = Path(nd) / "test_s*.parquet"
    FR.CAND_FILE = Path(cand)
    c = con(f"decide_{variant}")
    c.execute(f"CREATE TABLE s1 AS SELECT entity_id, {ID} AS eid FROM read_parquet('{nd}/test_s1.parquet')")
    c.execute(f"CREATE TABLE names AS SELECT {ID} AS eid, country, name_full, name_core, address FROM read_parquet('{nd}/test_s*.parquet')")
    c.execute(f"""
        CREATE TABLE best AS
        SELECT * FROM (
            SELECT b.rid, b.s1, CASE WHEN b.s1 = b.g_s1 THEN least(b.p, b.g_p) ELSE 0 END AS p, b.p AS p_raw, b.g_s1, b.g_p
            FROM (
                SELECT rid, arg_max(s1, (p, -s1)) FILTER (WHERE p IS NOT NULL) AS s1, max(p) AS p,
                    arg_max(s1, (p_guard, -s1)) FILTER (WHERE p_guard IS NOT NULL) AS g_s1, max(p_guard) AS g_p
                FROM read_parquet('{scores}') GROUP BY rid
            ) b JOIN names q ON q.eid = b.rid
        ) WHERE p >= 0.01
    """)
    pairs = c.execute("""
        SELECT b.rid, b.s1, b.p, q.country, q.name_full AS q_full, s.name_full AS s_full
        FROM best b JOIN names q ON q.eid = b.rid JOIN names s ON s.eid = b.s1
    """).fetchnumpy()
    s1n = c.execute("SELECT country, name_full FROM names WHERE eid // 10000000000 = 1").fetchnumpy()
    decoy_hit, swap_hit = veto_flags(pairs["country"], pairs["q_full"], pairs["s_full"], pairs["p"],
                                     build_vocab(s1n["country"], s1n["name_full"]), load_decoy_words(), set())
    keep = ~(decoy_hit | swap_hit) & (pairs["p"] >= THRESHOLD)
    n_thr = int((pairs["p"] >= THRESHOLD).sum())
    c.register("kept_arrow", pa.table({"rid": pairs["rid"][keep], "s1": pairs["s1"][keep], "p": pairs["p"][keep]}))
    c.execute(f"""CREATE TABLE kept AS SELECT k.rid, k.s1, k.p, f.first_num_equal AS fne
        FROM kept_arrow k LEFT JOIN read_parquet('{w}/pred1.parquet') f USING (rid, s1)""")
    c.execute(f"""CREATE TABLE sibling AS SELECT k.rid FROM kept k
        WHERE k.p < {SIBLING_MAX_P} AND k.fne = 0 AND EXISTS (
            SELECT 1 FROM kept x WHERE x.s1 = k.s1 AND x.rid <> k.rid AND x.rid // 10000000000 = k.rid // 10000000000 AND x.fne = 1)""")
    c.execute("CREATE TABLE after_sibling AS SELECT * FROM kept WHERE rid NOT IN (SELECT rid FROM sibling)")
    additions = AR.exact_address_matches(c, "test", Path(cand))
    c.register("additions_arrow", pa.table({"rid": additions["rid"], "s1": additions["s1"]}))
    c.execute("CREATE TABLE additions AS SELECT rid, s1 FROM additions_arrow WHERE rid NOT IN (SELECT rid FROM kept)")
    n_best, n_sib, n_add, n_as = c.execute("""SELECT (SELECT count(*) FROM best), (SELECT count(*) FROM sibling),
        (SELECT count(*) FROM additions), (SELECT count(*) FROM after_sibling)""").fetchone()
    T(f"{variant}: best {n_best:,}, p>=0.85 {n_thr:,}, decoy vetoed {int((decoy_hit & (pairs['p'] >= THRESHOLD)).sum()):,} "
      f"(all p {int(decoy_hit.sum()):,}), kept {int(keep.sum()):,}, sibling dropped {n_sib:,}, after sibling {n_as:,}, exact-address additions {n_add:,}")
    c.execute(f"COPY (SELECT rid, s1 FROM after_sibling UNION ALL SELECT rid, s1 FROM additions) TO '{w}/base_assign.parquet' (FORMAT parquet)")
    Path(f"{F}/tmp/decide_{variant}").mkdir(parents=True, exist_ok=True)
    removals, proposed = FR.france_changes(c, scores, f"{w}/base_assign.parquet")
    c.register("rem_arrow", pa.table({"rid": pa.array(removals["rid"], pa.int64()), "s1": pa.array(removals["s1"], pa.int64())}))
    c.register("prop_arrow", pa.table({"rid": pa.array(proposed["rid"], pa.int64()), "s1": pa.array(proposed["s1"], pa.int64()),
                                       "rule": pa.array(proposed["rule"], pa.string())}))
    for tb in ("after_sibling", "additions"):
        c.execute(f"DELETE FROM {tb} WHERE (rid, s1) IN (SELECT rid, s1 FROM rem_arrow)")
    c.execute("""CREATE TABLE france AS SELECT rid, min(s1) AS s1 FROM prop_arrow
        WHERE rid NOT IN (SELECT rid FROM after_sibling) AND rid NOT IN (SELECT rid FROM additions) AND s1 IN (SELECT eid FROM s1)
        GROUP BY rid""")
    c.execute(f"""COPY (SELECT rid, s1, 'model' AS how FROM after_sibling UNION ALL SELECT rid, s1, 'exact_address' FROM additions
        UNION ALL SELECT rid, s1, 'france_rules' FROM france) TO '{w}/assign.parquet' (FORMAT parquet)""")
    n_fin, n_dup = c.execute(f"SELECT count(*), count(*) - count(DISTINCT rid) FROM '{w}/assign.parquet'").fetchone()
    T(f"{variant}: france_changes removals {len(removals['rid']):,}, proposed {len(proposed['rid']):,}, added {c.execute('SELECT count(*) FROM france').fetchone()[0]:,}; "
      f"final {n_fin:,} pairs, duplicate rids {n_dup}")
    c.execute(f"CREATE TABLE m AS SELECT rid, s1 FROM '{w}/base_assign.parquet'")
    c.execute(f"DELETE FROM m WHERE (rid, s1) IN (SELECT rid, s1 FROM '{J}/judge_keep_removals.parquet')")
    c.execute(f"""CREATE TABLE ad AS SELECT rid, arg_min(s1, s1) AS s1 FROM (
        SELECT rid, s1 FROM '{J}/judge_keep_adds.parquet' UNION ALL SELECT rid, s1 FROM '{J}/judge_n2_adds.parquet'
        UNION ALL SELECT rid, s1 FROM '{SP}/rescue/test_rescue_adds.parquet') GROUP BY rid""")
    c.execute("DELETE FROM ad WHERE rid IN (SELECT rid FROM m) OR s1 NOT IN (SELECT eid FROM s1)")
    c.execute("INSERT INTO m SELECT rid, s1 FROM ad")
    c.execute(f"COPY m TO '{w}/assign_judge.parquet' (FORMAT parquet)")
    T(f"{variant}: base + v15 judge files -> {c.execute('SELECT count(*) FROM m').fetchone()[0]:,} pairs")
    c.close()
    shutil.rmtree(f"{F}/tmp/decide_{variant}", ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
