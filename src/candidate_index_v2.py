import duckdb
import time

from config import OUTPUT_DIR, DB_PATH, TEMP_DIR

# ==========================================
# CONFIGURATION
# ==========================================

V1_CANDIDATES_FILE = OUTPUT_DIR / "validation_candidates.tsv"

SAMPLE_SIZE = 5000
MAX_TOKEN_FREQUENCY = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DATABASE
# ==========================================

def connect_db():
    con = duckdb.connect(str(DB_PATH))

    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")
    con.execute(
        f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
    )

    return con


# ==========================================
# VALIDATION SAMPLE
# ==========================================

def create_validation_sample(con):
    print("\nCreating controlled validation sample...")

    if not V1_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"V1 validation file not found: "
            f"{V1_CANDIDATES_FILE}"
        )

    # Read the exact V1 validation IDs.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE v1_validation_ids AS
        SELECT DISTINCT
            source1_entity_id
        FROM read_csv(
            '{V1_CANDIDATES_FILE.as_posix()}',
            delim = '\\t',
            header = true,
            all_varchar = true
        )
        WHERE source1_entity_id IS NOT NULL
          AND source1_entity_id <> ''
    """)

    v1_count = con.execute("""
        SELECT COUNT(*)
        FROM v1_validation_ids
    """).fetchone()[0]

    if v1_count != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE:,} V1 IDs, "
            f"but found {v1_count:,}. "
            "Check validation_candidates.tsv."
        )

    # Use precisely those IDs from the normalized source1 table.
    con.execute("""
        CREATE OR REPLACE TABLE validation_s1 AS
        SELECT
            s.entity_id,
            s.business_name,
            s.business_address,
            s.country,
            s.normalized_name,
            s.normalized_address,
            s.normalized_country
        FROM train_source1_norm s
        INNER JOIN v1_validation_ids v
            ON s.entity_id = v.source1_entity_id
    """)

    # Ground truth for exactly the same validation businesses.
    con.execute("""
        CREATE OR REPLACE TABLE validation_gt AS
        SELECT
            g.source1_entity_id,
            g.matched_entity_ids
        FROM ground_truth g
        INNER JOIN validation_s1 v
            ON g.source1_entity_id = v.entity_id
    """)

    actual_count = con.execute("""
        SELECT COUNT(*)
        FROM validation_s1
    """).fetchone()[0]

    gt_count = con.execute("""
        SELECT COUNT(*)
        FROM validation_gt
    """).fetchone()[0]

    if actual_count != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE:,} validation businesses "
            f"in source1, but found {actual_count:,}."
        )

    if gt_count != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE:,} ground-truth rows, "
            f"but found {gt_count:,}."
        )

    print(f"V1 validation IDs: {v1_count:,}")
    print(f"Matched source1 records: {actual_count:,}")
    print(f"Ground-truth records: {gt_count:,}")
    print("Validation sample matches V1 exactly.")


# ==========================================
# CANDIDATE GENERATION
# ==========================================

def create_candidates_for_source(
    con,
    source_table,
    label
):
    print(f"\nGenerating candidates from {source_table}...")

    start = time.time()

    # Initialize the candidate table.
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates_{label} AS
        SELECT
            v.entity_id AS source1_entity_id,
            s.entity_id AS candidate_entity_id
        FROM validation_s1 v
        CROSS JOIN {source_table} s
        WHERE FALSE
    """)

    # --------------------------------------
    # BLOCK 1: Exact normalized name prefix
    # --------------------------------------

    print("Block 1: Name prefix")

    con.execute(f"""
        INSERT INTO candidates_{label}
        SELECT DISTINCT
            v.entity_id AS source1_entity_id,
            s.entity_id AS candidate_entity_id
        FROM validation_s1 v
        INNER JOIN {source_table} s
            ON v.normalized_country = s.normalized_country
            AND length(v.normalized_name) >= 4
            AND substr(v.normalized_name, 1, 4)
                = substr(s.normalized_name, 1, 4)
        WHERE v.normalized_name <> ''
          AND s.normalized_name <> ''
    """)

    # --------------------------------------
    # BLOCK 2: Rare name tokens
    # --------------------------------------

    print("Block 2: Rare name tokens")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE val_name_tokens AS
        SELECT DISTINCT
            v.entity_id AS source1_entity_id,
            trim(token) AS token,
            v.normalized_country
        FROM validation_s1 v,
        UNNEST(
            string_split(v.normalized_name, ' ')
        ) t(token)
        WHERE length(trim(token)) >= 4
          AND trim(token) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE source_name_tokens AS
        SELECT DISTINCT
            s.entity_id AS candidate_entity_id,
            trim(token) AS token,
            s.normalized_country
        FROM {source_table} s,
        UNNEST(
            string_split(s.normalized_name, ' ')
        ) t(token)
        WHERE length(trim(token)) >= 4
          AND trim(token) <> ''
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE source_token_frequency AS
        SELECT
            token,
            COUNT(*) AS frequency
        FROM source_name_tokens
        GROUP BY token
    """)

    con.execute(f"""
        INSERT INTO candidates_{label}
        SELECT DISTINCT
            v.source1_entity_id,
            s.candidate_entity_id
        FROM val_name_tokens v
        INNER JOIN source_name_tokens s
            ON v.token = s.token
            AND v.normalized_country = s.normalized_country
        INNER JOIN source_token_frequency f
            ON s.token = f.token
        WHERE f.frequency <= {MAX_TOKEN_FREQUENCY}
    """)

    # --------------------------------------
    # BLOCK 3: Rare address tokens
    # --------------------------------------

    print("Block 3: Rare address tokens")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE val_address_tokens AS
        SELECT DISTINCT
            v.entity_id AS source1_entity_id,
            trim(token) AS token,
            v.normalized_country
        FROM validation_s1 v,
        UNNEST(
            string_split(v.normalized_address, ' ')
        ) t(token)
        WHERE length(trim(token)) >= 4
          AND trim(token) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE source_address_tokens AS
        SELECT DISTINCT
            s.entity_id AS candidate_entity_id,
            trim(token) AS token,
            s.normalized_country
        FROM {source_table} s,
        UNNEST(
            string_split(s.normalized_address, ' ')
        ) t(token)
        WHERE length(trim(token)) >= 4
          AND trim(token) <> ''
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE source_address_frequency AS
        SELECT
            token,
            COUNT(*) AS frequency
        FROM source_address_tokens
        GROUP BY token
    """)

    con.execute(f"""
        INSERT INTO candidates_{label}
        SELECT DISTINCT
            v.source1_entity_id,
            s.candidate_entity_id
        FROM val_address_tokens v
        INNER JOIN source_address_tokens s
            ON v.token = s.token
            AND v.normalized_country = s.normalized_country
        INNER JOIN source_address_frequency f
            ON s.token = f.token
        WHERE f.frequency <= {MAX_TOKEN_FREQUENCY}
    """)

    # --------------------------------------
    # Deduplicate final candidate pairs
    # --------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TABLE candidates_{label}_final AS
        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id
        FROM candidates_{label}
    """)

    count = con.execute(f"""
        SELECT COUNT(*)
        FROM candidates_{label}_final
    """).fetchone()[0]

    print(f"Unique candidate pairs: {count:,}")
    print(f"Elapsed: {time.time() - start:.1f}s")


# ==========================================
# EVALUATION
# ==========================================

def evaluate(con):
    print("\nEvaluating candidate recall...")

    # UNION (not UNION ALL) removes duplicate pairs
    # across source2 and source3.
    con.execute("""
        CREATE OR REPLACE TABLE all_candidates AS
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidates_s2_final

        UNION

        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidates_s3_final
    """)

    con.execute("""
        CREATE OR REPLACE TABLE candidate_evaluation AS
        SELECT
            v.entity_id AS source1_entity_id,

            COUNT(DISTINCT c.candidate_entity_id)
                AS candidate_count,

            COUNT(DISTINCT CASE
                WHEN list_contains(
                    string_split(g.matched_entity_ids, ','),
                    c.candidate_entity_id
                )
                THEN c.candidate_entity_id
            END) AS recovered_matches,

            CASE
                WHEN g.matched_entity_ids IS NULL
                  OR trim(g.matched_entity_ids) = ''
                THEN 0
                ELSE
                    length(g.matched_entity_ids)
                    - length(
                        replace(
                            g.matched_entity_ids,
                            ',',
                            ''
                        )
                    )
                    + 1
            END AS true_matches

        FROM validation_s1 v

        INNER JOIN validation_gt g
            ON v.entity_id = g.source1_entity_id

        LEFT JOIN all_candidates c
            ON v.entity_id = c.source1_entity_id

        GROUP BY
            v.entity_id,
            g.matched_entity_ids
    """)

    result = con.execute("""
        SELECT
            SUM(true_matches),
            SUM(recovered_matches),
            SUM(candidate_count),
            AVG(candidate_count)
        FROM candidate_evaluation
    """).fetchone()

    true_matches, recovered, total_candidates, avg_candidates = result

    recall = (
        recovered / true_matches
        if true_matches
        else 0
    )

    print("\n========== V2 VALIDATION RESULTS ==========")
    print(f"True matches: {true_matches:,}")
    print(f"Recovered true matches: {recovered:,}")
    print(f"Candidate recall: {recall:.4%}")
    print(f"Total candidate pairs: {total_candidates:,}")
    print(
        f"Average candidates per business: "
        f"{avg_candidates:.2f}"
    )

    # Save validation candidates.
    output_file = OUTPUT_DIR / "validation_candidates_v2.tsv"

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                string_agg(
                    candidate_entity_id,
                    ','
                    ORDER BY candidate_entity_id
                ) AS candidate_entity_ids
            FROM all_candidates
            GROUP BY source1_entity_id
        )
        TO '{output_file.as_posix()}'
        (HEADER, DELIMITER '\\t')
    """)

    print(f"\nSaved candidates: {output_file}")


# ==========================================
# MAIN
# ==========================================

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    con = connect_db()

    try:
        create_validation_sample(con)

        create_candidates_for_source(
            con,
            "train_source2_norm",
            "s2"
        )

        create_candidates_for_source(
            con,
            "train_source3_norm",
            "s3"
        )

        evaluate(con)

    finally:
        con.close()


if __name__ == "__main__":
    main()