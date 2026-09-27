
import duckdb

DB_PATH = "data/entity_resolution.duckdb"

con = duckdb.connect(DB_PATH)
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='data/temp'")

print("Checking complete candidate coverage for held-out S1 groups...")

query = """
WITH heldout AS (
    SELECT source1_entity_id, matched_entity_ids
    FROM validation_gt
    WHERE hash(source1_entity_id) % 5 = 0
),
candidate_stats AS (
    SELECT
        c.source1_entity_id,
        COUNT(*) AS candidate_count
    FROM cheap_candidate_ranks c
    JOIN heldout h
      ON c.source1_entity_id = h.source1_entity_id
    GROUP BY c.source1_entity_id
),
match_stats AS (
    SELECT
        h.source1_entity_id,
        COUNT(DISTINCT c.candidate_entity_id) AS recovered_matches
    FROM heldout h
    JOIN cheap_candidate_ranks c
      ON c.source1_entity_id = h.source1_entity_id
    WHERE h.matched_entity_ids IS NOT NULL
      AND h.matched_entity_ids != ''
      AND list_contains(
          string_split(h.matched_entity_ids, ','),
          c.candidate_entity_id
      )
    GROUP BY h.source1_entity_id
),
truth_stats AS (
    SELECT
        source1_entity_id,
        CASE
            WHEN matched_entity_ids IS NULL
              OR matched_entity_ids = ''
            THEN 0
            ELSE len(string_split(matched_entity_ids, ','))
        END AS true_match_count
    FROM heldout
)
SELECT
    COUNT(*) AS heldout_s1_groups,
    SUM(COALESCE(cs.candidate_count, 0)) AS full_candidate_pairs,
    SUM(ts.true_match_count) AS total_ground_truth_matches,
    SUM(COALESCE(ms.recovered_matches, 0)) AS recovered_matches,
    ROUND(
        100.0 * SUM(COALESCE(ms.recovered_matches, 0))
        / NULLIF(SUM(ts.true_match_count), 0),
        4
    ) AS candidate_recall_pct,
    ROUND(
        AVG(COALESCE(cs.candidate_count, 0)),
        2
    ) AS avg_candidates_per_s1
FROM heldout h
JOIN truth_stats ts
  ON h.source1_entity_id = ts.source1_entity_id
LEFT JOIN candidate_stats cs
  ON h.source1_entity_id = cs.source1_entity_id
LEFT JOIN match_stats ms
  ON h.source1_entity_id = ms.source1_entity_id
"""

print(con.execute(query).fetchdf().to_string(index=False))

con.close()
print("Coverage check completed.")