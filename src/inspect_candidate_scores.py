import duckdb

DB_PATH = "data/entity_resolution.duckdb"

con = duckdb.connect(DB_PATH, read_only=True)

print("\n--- CHEAP SCORE DISTRIBUTION ---")

query = """
SELECT
    cheap_score,
    COUNT(*) AS pair_count,
    COUNT(DISTINCT source1_entity_id) AS s1_groups
FROM pruning_features
GROUP BY cheap_score
ORDER BY cheap_score DESC
LIMIT 30
"""

print(con.execute(query).fetchdf().to_string(index=False))

print("\n--- CHEAP RANK DISTRIBUTION ---")

query = """
SELECT
    cheap_rank,
    COUNT(*) AS pair_count
FROM cheap_candidate_ranks
GROUP BY cheap_rank
ORDER BY cheap_rank
LIMIT 15
"""

print(con.execute(query).fetchdf().to_string(index=False))

print("\n--- SAMPLE CANDIDATE FEATURES ---")

query = """
SELECT
    source1_entity_id,
    candidate_entity_id,
    source_label,
    name1,
    name2,
    address1,
    address2,
    cheap_score
FROM pruning_features
LIMIT 10
"""

print(con.execute(query).fetchdf().to_string(index=False))

con.close()

print("\nInspection completed.")