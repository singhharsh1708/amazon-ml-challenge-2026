import duckdb
from pathlib import Path

DB = "data/entity_resolution.duckdb"

con = duckdb.connect(DB, read_only=True)
con.execute("SET threads=4")

query = """
WITH ranked AS (
    SELECT
        r.cheap_rank,
        r.source_label,
        r.cheap_score,
        CASE
            WHEN g.matched_entity_ids IS NULL
              OR g.matched_entity_ids = ''
            THEN 0
            WHEN list_contains(
                string_split(g.matched_entity_ids, ','),
                r.candidate_entity_id
            )
            THEN 1
            ELSE 0
        END AS is_match
    FROM cheap_candidate_ranks r
    JOIN validation_gt g
      ON r.source1_entity_id = g.source1_entity_id
    WHERE hash(r.source1_entity_id) % 5 = 0
),
binned AS (
    SELECT
        CASE
            WHEN cheap_rank <= 1 THEN '1'
            WHEN cheap_rank <= 3 THEN '2-3'
            WHEN cheap_rank <= 5 THEN '4-5'
            WHEN cheap_rank <= 10 THEN '6-10'
            WHEN cheap_rank <= 20 THEN '11-20'
            WHEN cheap_rank <= 50 THEN '21-50'
            WHEN cheap_rank <= 100 THEN '51-100'
            WHEN cheap_rank <= 200 THEN '101-200'
            WHEN cheap_rank <= 300 THEN '201-300'
            WHEN cheap_rank <= 500 THEN '301-500'
            WHEN cheap_rank <= 1000 THEN '501-1000'
            WHEN cheap_rank <= 2000 THEN '1001-2000'
            ELSE '2001+'
        END AS rank_bucket,
        is_match
    FROM ranked
)
SELECT
    rank_bucket,
    COUNT(*) AS candidate_pairs,
    SUM(is_match) AS true_matches,
    ROUND(
        100.0 * SUM(is_match) / COUNT(*),
        6
    ) AS match_rate_percent
FROM binned
GROUP BY rank_bucket
ORDER BY
    CASE rank_bucket
        WHEN '1' THEN 1
        WHEN '2-3' THEN 2
        WHEN '4-5' THEN 3
        WHEN '6-10' THEN 4
        WHEN '11-20' THEN 5
        WHEN '21-50' THEN 6
        WHEN '51-100' THEN 7
        WHEN '101-200' THEN 8
        WHEN '201-300' THEN 9
        WHEN '301-500' THEN 10
        WHEN '501-1000' THEN 11
        WHEN '1001-2000' THEN 12
        ELSE 13
    END
"""

print(con.execute(query).fetchdf().to_string(index=False))
con.close()
