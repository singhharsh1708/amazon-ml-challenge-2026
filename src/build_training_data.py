
import duckdb
from pathlib import Path

DB_PATH = "data/entity_resolution.duckdb"
OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "labeled_pairs.parquet"

con = duckdb.connect(DB_PATH)

con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")
con.execute("SET temp_directory='data/temp'")

# Keep all positive candidate pairs up to rank 2000.
# Sample at most 100 negative pairs per S1 and source
# from the top 300 cheap-ranked candidates.
query = """
CREATE OR REPLACE TABLE labeled_training_pairs AS
WITH labeled AS (
    SELECT
        f.source1_entity_id,
        f.candidate_entity_id,
        f.source_label,
        f.name1,
        f.name2,
        f.address1,
        f.address2,
        r.cheap_score,
        r.cheap_rank,
        CASE
            WHEN g.matched_entity_ids IS NOT NULL
             AND g.matched_entity_ids != ''
             AND list_contains(
                 string_split(g.matched_entity_ids, ','),
                 f.candidate_entity_id
             )
            THEN 1
            ELSE 0
        END AS label
    FROM pruning_features f
    JOIN cheap_candidate_ranks r
      ON f.source1_entity_id = r.source1_entity_id
     AND f.candidate_entity_id = r.candidate_entity_id
     AND f.source_label = r.source_label
    LEFT JOIN validation_gt g
      ON f.source1_entity_id = g.source1_entity_id
),
positives AS (
    SELECT *
    FROM labeled
    WHERE label = 1
      AND cheap_rank <= 2000
),
ranked_negatives AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY source1_entity_id, source_label
            ORDER BY cheap_rank, candidate_entity_id
        ) AS negative_number
    FROM labeled
    WHERE label = 0
      AND cheap_rank <= 300
),
selected_negatives AS (
    SELECT *
    FROM ranked_negatives
    WHERE negative_number <= 100
)
SELECT
    source1_entity_id,
    candidate_entity_id,
    source_label,
    name1,
    name2,
    address1,
    address2,
    cheap_score,
    cheap_rank,
    label,
    CASE
        WHEN hash(source1_entity_id) % 5 = 0
        THEN 'valid'
        ELSE 'train'
    END AS split
FROM positives

UNION ALL

SELECT
    source1_entity_id,
    candidate_entity_id,
    source_label,
    name1,
    name2,
    address1,
    address2,
    cheap_score,
    cheap_rank,
    label,
    CASE
        WHEN hash(source1_entity_id) % 5 = 0
        THEN 'valid'
        ELSE 'train'
    END AS split
FROM selected_negatives
"""

print("Building labeled pairs...")
con.execute(query)

print("\n--- DATASET SUMMARY ---")

summary = con.execute("""
    SELECT
        split,
        label,
        COUNT(*) AS pairs,
        COUNT(DISTINCT source1_entity_id) AS s1_groups
    FROM labeled_training_pairs
    GROUP BY split, label
    ORDER BY split, label DESC
""").fetchdf()

print(summary.to_string(index=False))

print("\n--- POSITIVE COVERAGE ---")

coverage = con.execute("""
    SELECT
        COUNT(*) AS positive_pairs,
        COUNT(DISTINCT source1_entity_id) AS groups_with_positive_candidates
    FROM labeled_training_pairs
    WHERE label = 1
""").fetchdf()

print(coverage.to_string(index=False))

print("\nExporting Parquet...")
con.execute(f"""
    COPY labeled_training_pairs
    TO '{OUTPUT_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print(f"\nSaved: {OUTPUT_FILE}")

con.close()
print("Completed.")