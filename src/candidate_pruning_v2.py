
import duckdb
from collections import defaultdict
from rapidfuzz import fuzz
import csv
import time

from config import OUTPUT_DIR, DB_PATH, TEMP_DIR

# ==========================================
# CONFIGURATION
# ==========================================

SHORTLIST_SIZE = 300
TOP_K_VALUES = [50, 100, 200]

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
# BUILD A CHEAP SHORTLIST
# ==========================================

def build_shortlist(con):
    print("\nBuilding cheap candidate shortlist...")
    start = time.time()

    # Preserve the source label because source2 and source3
    # are separate candidate pools.
    con.execute("""
        CREATE OR REPLACE TABLE pruning_candidates AS
        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id,
            's2' AS source_label
        FROM candidates_s2_final

        UNION

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id,
            's3' AS source_label
        FROM candidates_s3_final
    """)

    # Attach normalized names and addresses.
    con.execute("""
        CREATE OR REPLACE TABLE pruning_features AS
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.source_label,

            v.normalized_name AS name1,
            s.normalized_name AS name2,

            v.normalized_address AS address1,
            s.normalized_address AS address2,

            CASE
                WHEN v.normalized_name = s.normalized_name
                    AND v.normalized_name <> ''
                THEN 1000
                ELSE 0
            END
            +
            CASE
                WHEN v.normalized_address = s.normalized_address
                    AND v.normalized_address <> ''
                THEN 800
                ELSE 0
            END
            +
            CASE
                WHEN length(v.normalized_name) >= 4
                 AND length(s.normalized_name) >= 4
                 AND substr(v.normalized_name, 1, 4)
                     = substr(s.normalized_name, 1, 4)
                THEN 100
                ELSE 0
            END
            +
            CASE
                WHEN length(v.normalized_address) >= 5
                 AND length(s.normalized_address) >= 5
                 AND substr(v.normalized_address, 1, 5)
                     = substr(s.normalized_address, 1, 5)
                THEN 50
                ELSE 0
            END AS cheap_score

        FROM pruning_candidates c

        INNER JOIN validation_s1 v
            ON c.source1_entity_id = v.entity_id

        INNER JOIN train_source2_norm s
            ON c.source_label = 's2'
            AND c.candidate_entity_id = s.entity_id

        UNION ALL

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.source_label,

            v.normalized_name AS name1,
            s.normalized_name AS name2,

            v.normalized_address AS address1,
            s.normalized_address AS address2,

            CASE
                WHEN v.normalized_name = s.normalized_name
                    AND v.normalized_name <> ''
                THEN 1000
                ELSE 0
            END
            +
            CASE
                WHEN v.normalized_address = s.normalized_address
                    AND v.normalized_address <> ''
                THEN 800
                ELSE 0
            END
            +
            CASE
                WHEN length(v.normalized_name) >= 4
                 AND length(s.normalized_name) >= 4
                 AND substr(v.normalized_name, 1, 4)
                     = substr(s.normalized_name, 1, 4)
                THEN 100
                ELSE 0
            END
            +
            CASE
                WHEN length(v.normalized_address) >= 5
                 AND length(s.normalized_address) >= 5
                 AND substr(v.normalized_address, 1, 5)
                     = substr(s.normalized_address, 1, 5)
                THEN 50
                ELSE 0
            END AS cheap_score

        FROM pruning_candidates c

        INNER JOIN validation_s1 v
            ON c.source1_entity_id = v.entity_id

        INNER JOIN train_source3_norm s
            ON c.source_label = 's3'
            AND c.candidate_entity_id = s.entity_id
    """)

    # Rank candidates separately within each source pool.
    con.execute("""
        CREATE OR REPLACE TABLE pruning_shortlist AS
        SELECT *
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        source1_entity_id,
                        source_label
                    ORDER BY
                        cheap_score DESC,
                        candidate_entity_id
                ) AS cheap_rank
            FROM pruning_features
        )
        WHERE cheap_rank <= ?
    """, [SHORTLIST_SIZE])

    count = con.execute("""
        SELECT COUNT(*)
        FROM pruning_shortlist
    """).fetchone()[0]

    print(f"Shortlist rows: {count:,}")
    print(f"Elapsed: {time.time() - start:.1f}s")


# ==========================================
# RAPIDFUZZ RERANKING
# ==========================================

def rerank_candidates(con):
    print("\nReranking shortlist with RapidFuzz...")
    start = time.time()

    # Use a cursor and fetchmany so the full shortlist
    # is not loaded into RAM at once.
    cursor = con.execute("""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            source_label,
            name1,
            name2,
            address1,
            address2
        FROM pruning_shortlist
        ORDER BY
            source1_entity_id,
            source_label,
            cheap_rank
    """)

    ranked = defaultdict(list)

    while True:
        batch = cursor.fetchmany(50000)

        if not batch:
            break

        for row in batch:
            (
                source1_id,
                candidate_id,
                source_label,
                name1,
                name2,
                address1,
                address2
            ) = row

            name1 = name1 or ""
            name2 = name2 or ""
            address1 = address1 or ""
            address2 = address2 or ""

            name_score = (
                fuzz.ratio(name1, name2)
                if name1 and name2
                else 0
            )

            address_score = (
                fuzz.ratio(address1, address2)
                if address1 and address2
                else 0
            )

            # Name receives higher weight than address.
            final_score = (
                0.70 * name_score
                + 0.30 * address_score
            )

            ranked[(source1_id, source_label)].append(
                (
                    candidate_id,
                    final_score
                )
            )

    # Sort each source-specific candidate list.
    for key in ranked:
        ranked[key].sort(
            key=lambda x: (-x[1], x[0])
        )

    print(f"Reranked groups: {len(ranked):,}")
    print(f"Elapsed: {time.time() - start:.1f}s")

    return ranked


# ==========================================
# EVALUATION
# ==========================================

def evaluate_pruning(con, ranked):
    print("\nEvaluating pruning thresholds...")

    # Ground truth lookup for the validation sample.
    gt_rows = con.execute("""
        SELECT
            source1_entity_id,
            matched_entity_ids
        FROM validation_gt
    """).fetchall()

    ground_truth = {}

    for source1_id, matches in gt_rows:
        if matches:
            ground_truth[source1_id] = set(
                x.strip()
                for x in matches.split(",")
                if x.strip()
            )
        else:
            ground_truth[source1_id] = set()

    # Combine the source2 and source3 candidate pools
    # and deduplicate IDs per source1 business.
    combined = defaultdict(dict)

    for (source1_id, source_label), candidates in ranked.items():
        for candidate_id, score in candidates:
            old = combined[source1_id].get(candidate_id)

            if old is None or score > old:
                combined[source1_id][candidate_id] = score

    ordered = {}

    for source1_id, candidates in combined.items():
        ordered[source1_id] = sorted(
            candidates.items(),
            key=lambda x: (-x[1], x[0])
        )

    true_total = sum(
        len(matches)
        for matches in ground_truth.values()
    )

    print(f"Ground-truth matches: {true_total:,}")

    for k in TOP_K_VALUES:
        recovered = 0
        total_candidates = 0

        output_file = (
            OUTPUT_DIR / f"pruned_candidates_top{k}.tsv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(
                f,
                delimiter="\t"
            )

            writer.writerow([
                "source1_entity_id",
                "candidate_entity_ids"
            ])

            for source1_id in ground_truth:
                candidates = ordered.get(
                    source1_id,
                    []
                )[:k]

                candidate_ids = [
                    candidate_id
                    for candidate_id, score in candidates
                ]

                total_candidates += len(candidate_ids)

                recovered += len(
                    set(candidate_ids)
                    & ground_truth[source1_id]
                )

                writer.writerow([
                    source1_id,
                    ",".join(candidate_ids)
                ])

        recall = (
            recovered / true_total
            if true_total
            else 0
        )

        average = (
            total_candidates / len(ground_truth)
            if ground_truth
            else 0
        )

        print(f"\nTOP {k}")
        print(f"Recovered matches: {recovered:,}")
        print(f"Candidate recall: {recall:.4%}")
        print(f"Total candidates: {total_candidates:,}")
        print(f"Average candidates/business: {average:.2f}")
        print(f"Saved: {output_file}")


# ==========================================
# MAIN
# ==========================================

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    con = connect_db()

    try:
        build_shortlist(con)

        ranked = rerank_candidates(con)

        evaluate_pruning(con, ranked)

    finally:
        con.close()


if __name__ == "__main__":
    main()