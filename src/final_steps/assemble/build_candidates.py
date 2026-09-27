"""Build the packaged candidate_pairs.tsv: exactly the pairs the final models scored.

Inputs (paths relative to REPO_DIR and WORK_DIR):
  data/norm/test_s1.parquet                       test Source 1 records with country
  data/candidates/test.parquet                    blocking candidates (US and India rows are kept)
  WORK_DIR/fr_rerun/out/france_candidates.parquet French candidates from the re-run with fixed normalization
  WORK_DIR/rescue/test_rescue.parquet             rescue pool scored by the cross-encoders (all countries)
  WORK_DIR/top50/errors/test_nm1d_ce2.parquet     name-key rescue pool (nm1d) scored by the cross-encoders
  WORK_DIR/top50/errors/test_hc_ce2.parquet       name-key rescue pool (hc) scored by the cross-encoders
  WORK_DIR/top50/reverse/k20_test_ce2.parquet     reverse-blocking pool scored by the cross-encoders
  <matching>                                      final matching_results.tsv (every match is added)
Output: <out> in the candidate_pairs.tsv format, one row per Source 1 entity, IDs sorted.
Usage: python build_candidates.py <matching.tsv> <out.tsv>
"""

import os
import shutil
import sys

import duckdb

REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
PARTS = 16
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
STR = "'S' || ({c} // 10000000000)::varchar || '-' || ({c} % 10000000000)::varchar"


def build(matching, out):
    """Union every scored candidate source with the final matches and write the TSV."""
    tmp = os.path.join(WORK_DIR, "candtmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='3GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{tmp}'")
    try:
        con.execute(f"""CREATE TABLE s1 AS SELECT entity_id AS sid, {ID.format(c='entity_id')} AS s1, country
            FROM read_parquet('{REPO_DIR}/data/norm/test_s1.parquet')""")
        con.execute(f"""CREATE TABLE m AS SELECT {ID.format(c='source1_entity_id')} s1, {ID.format(c='x')} rid FROM (
            SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) x
            FROM read_csv('{matching}', delim='\t', header=true, all_varchar=true, quote='', escape='')
            WHERE coalesce(matched_entity_ids, '') <> '')""")
        con.execute(f"""CREATE TABLE cand AS SELECT DISTINCT rid, s1 FROM (
            SELECT t.rid, t.s1 FROM read_parquet('{REPO_DIR}/data/candidates/test.parquet') t
                JOIN s1 ON s1.s1 = t.s1 WHERE s1.country <> 'france'
            UNION ALL SELECT rid, s1 FROM '{WORK_DIR}/fr_rerun/out/france_candidates.parquet'
            UNION ALL SELECT rid, s1 FROM '{WORK_DIR}/rescue/test_rescue.parquet'
            UNION ALL SELECT rid, s1 FROM '{WORK_DIR}/top50/errors/test_nm1d_ce2.parquet'
            UNION ALL SELECT rid, s1 FROM '{WORK_DIR}/top50/errors/test_hc_ce2.parquet'
            UNION ALL SELECT rid, s1 FROM '{WORK_DIR}/top50/reverse/k20_test_ce2.parquet'
            UNION ALL SELECT rid, s1 FROM m)""")
        missing = con.execute("SELECT count(*) FROM m ANTI JOIN cand USING (rid, s1)").fetchone()[0]
        lines = []
        for k in range(PARTS):
            rows = con.execute(f"""SELECT s.sid, coalesce(string_agg({STR.format(c='t.rid')}, ',' ORDER BY t.rid), '')
                FROM (SELECT * FROM s1 WHERE s1 % {PARTS} = {k}) s
                LEFT JOIN (SELECT * FROM cand WHERE s1 % {PARTS} = {k}) t ON t.s1 = s.s1 GROUP BY s.sid""").fetchall()
            lines.extend(f"{a}\t{b}\n" for a, b in rows)
        lines.sort(key=lambda x: x.split("\t", 1)[0])
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("source1_entity_id\tcandidate_entity_ids\n")
            fh.writelines(lines)
        total = con.execute("SELECT count(*) FROM cand").fetchone()[0]
        print(f"candidate pairs {total:,}; matches missing from candidates {missing}; rows {len(lines):,}")
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
