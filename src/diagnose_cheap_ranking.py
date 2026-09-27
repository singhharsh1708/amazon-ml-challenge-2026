
import duckdb

from config import DB_PATH, TEMP_DIR


def main():
    con = duckdb.connect(str(DB_PATH))

    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute(
        f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
    )

    try:
        print("Ranking all existing candidate pairs...")

        # Rank candidates within each source1/source pool.
        con.execute("""
            CREATE OR REPLACE TABLE cheap_candidate_ranks AS
            SELECT
                source1_entity_id,
                candidate_entity_id,
                source_label,
                cheap_score,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        source1_entity_id,
                        source_label
                    ORDER BY
                        cheap_score DESC,
                        candidate_entity_id
                ) AS cheap_rank

            FROM pruning_features
        """)

        print("Ranking complete. Evaluating true matches...")

        # Find the cheap-score rank of each recovered true match.
        con.execute("""
            CREATE OR REPLACE TABLE true_match_cheap_ranks AS
            SELECT
                c.source1_entity_id,
                c.candidate_entity_id,
                c.source_label,
                c.cheap_score,
                c.cheap_rank

            FROM cheap_candidate_ranks c

            INNER JOIN validation_gt g
                ON c.source1_entity_id =
                   g.source1_entity_id

            WHERE list_contains(
                string_split(g.matched_entity_ids, ','),
                c.candidate_entity_id
            )
        """)

        print("\n========== CHEAP RANK DIAGNOSTIC ==========")

        total = con.execute("""
            SELECT COUNT(*)
            FROM true_match_cheap_ranks
        """).fetchone()[0]

        print(f"True candidate pairs ranked: {total:,}")

        for cutoff in [100, 300, 500, 1000, 2000]:
            count = con.execute("""
                SELECT COUNT(*)
                FROM true_match_cheap_ranks
                WHERE cheap_rank <= ?
            """, [cutoff]).fetchone()[0]

            recall = count / 17375

            print(
                f"Top {cutoff:>4}: "
                f"{count:>7,} matches | "
                f"{recall:.4%} of all true matches"
            )

        print("\nRank distribution:")

        rows = con.execute("""
            SELECT
                CASE
                    WHEN cheap_rank <= 300
                        THEN '1-300'
                    WHEN cheap_rank <= 500
                        THEN '301-500'
                    WHEN cheap_rank <= 1000
                        THEN '501-1000'
                    WHEN cheap_rank <= 2000
                        THEN '1001-2000'
                    ELSE 'Above 2000'
                END AS rank_range,

                COUNT(*) AS true_matches

            FROM true_match_cheap_ranks

            GROUP BY rank_range

            ORDER BY MIN(cheap_rank)
        """).fetchall()

        for rank_range, count in rows:
            print(f"{rank_range:>12}: {count:,}")

    finally:
        con.close()


if __name__ == "__main__":
    main()