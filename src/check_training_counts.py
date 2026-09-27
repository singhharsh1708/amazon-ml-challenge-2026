import duckdb

DB_PATH = "data/entity_resolution.duckdb"

con = duckdb.connect(DB_PATH, read_only=True)

print("\n--- CANDIDATE COUNTS ---")

query = """
SELECT
    COUNT(*) AS total_pairs,
    COUNT(DISTINCT source1_entity_id) AS source1_groups,
    COUNT(*) FILTER (WHERE cheap_rank <= 300) AS rank_300,
    COUNT(*) FILTER (WHERE cheap_rank <= 1000) AS rank_1000,
    COUNT(*) FILTER (WHERE cheap_rank <= 2000) AS rank_2000
FROM cheap_candidate_ranks
"""

print(con.execute(query).fetchdf().to_string(index=False))

print("\n--- GROUND TRUTH ---")

query = """
SELECT
    COUNT(*) AS validation_groups,
    COUNT(*) FILTER (
        WHERE matched_entity_ids IS NOT NULL
          AND matched_entity_ids != ''
    ) AS groups_with_matches
FROM validation_gt
"""

print(con.execute(query).fetchdf().to_string(index=False))

con.close()

print("\nDiagnostic completed.")