
import duckdb

con = duckdb.connect(
    "data/entity_resolution.duckdb",
    read_only=True,
)

con.execute("SET threads=4")

query = """
SELECT
    CASE
        WHEN r.cheap_rank <= 10 THEN '1-10'
        WHEN r.cheap_rank <= 100 THEN '11-100'
        WHEN r.cheap_rank <= 1000 THEN '101-1000'
        WHEN r.cheap_rank <= 2000 THEN '1001-2000'
        ELSE '2001+'
    END AS rank_bucket,
    COUNT(*) AS candidate_pairs,
    SUM(
        CASE
            WHEN g.matched_entity_ids IS NOT NULL
             AND g.matched_entity_ids != ''
             AND list_contains(
                 string_split(g.matched_entity_ids, ','),
                 r.candidate_entity_id
             )
            THEN 1
            ELSE 0
        END
    ) AS true_matches
FROM cheap_candidate_ranks r
JOIN ground_truth g
  ON r.source1_entity_id = g.source1_entity_id
WHERE hash(r.source1_entity_id) % 5 != 0
GROUP BY rank_bucket
ORDER BY
    CASE rank_bucket
        WHEN '1-10' THEN 1
        WHEN '11-100' THEN 2
        WHEN '101-1000' THEN 3
        WHEN '1001-2000' THEN 4
        ELSE 5
    END
"""

print(con.execute(query).fetchdf().to_string(index=False))

con.close()