import duckdb

from config import OUTPUT_DIR, DB_PATH


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        con.execute("SET threads = 4")

        true_total = con.execute("""
            SELECT SUM(
                CASE
                    WHEN matched_entity_ids IS NULL
                      OR trim(matched_entity_ids) = ''
                    THEN 0
                    ELSE
                        length(matched_entity_ids)
                        - length(
                            replace(
                                matched_entity_ids,
                                ',',
                                ''
                            )
                        )
                        + 1
                END
            )
            FROM validation_gt
        """).fetchone()[0]

        print(f"Total true matches: {true_total:,}")

        # --------------------------------------
        # Stage 1: All original V2 candidates
        # --------------------------------------

        print("\nSTAGE 1: ORIGINAL V2")

        result = con.execute("""
            SELECT COUNT(DISTINCT
                c.source1_entity_id || '|' ||
                c.candidate_entity_id
            )
            FROM all_candidates c
            INNER JOIN validation_gt g
                ON c.source1_entity_id =
                   g.source1_entity_id
            WHERE list_contains(
                string_split(g.matched_entity_ids, ','),
                c.candidate_entity_id
            )
        """).fetchone()[0]

        print(f"Recovered: {result:,}")
        print(f"Recall: {result / true_total:.4%}")

        # --------------------------------------
        # Stage 2: Cheap top-300 shortlist
        # --------------------------------------

        print("\nSTAGE 2: CHEAP SHORTLIST")

        result = con.execute("""
            SELECT COUNT(DISTINCT
                c.source1_entity_id || '|' ||
                c.candidate_entity_id
            )
            FROM pruning_shortlist c
            INNER JOIN validation_gt g
                ON c.source1_entity_id =
                   g.source1_entity_id
            WHERE list_contains(
                string_split(g.matched_entity_ids, ','),
                c.candidate_entity_id
            )
        """).fetchone()[0]

        print(f"Recovered: {result:,}")
        print(f"Recall: {result / true_total:.4%}")

        # --------------------------------------
        # Stage 3: RapidFuzz top K outputs
        # --------------------------------------

        for k in [50, 100, 200]:

            file_path = (
                OUTPUT_DIR /
                f"pruned_candidates_top{k}.tsv"
            )

            print(f"\nSTAGE 3: RAPIDFUZZ TOP {k}")

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE current_topk AS
                SELECT
                    source1_entity_id,
                    trim(candidate_id) AS candidate_id
                FROM read_csv(
                    '{file_path.as_posix()}',
                    delim = '\\t',
                    header = true,
                    all_varchar = true
                ),
                UNNEST(
                    string_split(candidate_entity_ids, ',')
                ) t(candidate_id)
                WHERE trim(candidate_id) <> ''
            """)

            result = con.execute("""
                SELECT COUNT(DISTINCT
                    c.source1_entity_id || '|' ||
                    c.candidate_id
                )
                FROM current_topk c
                INNER JOIN validation_gt g
                    ON c.source1_entity_id =
                       g.source1_entity_id
                WHERE list_contains(
                    string_split(g.matched_entity_ids, ','),
                    c.candidate_id
                )
            """).fetchone()[0]

            print(f"Recovered: {result:,}")
            print(f"Recall: {result / true_total:.4%}")

        print("\nDiagnostic complete.")

    finally:
        con.close()


if __name__ == "__main__":
    main()