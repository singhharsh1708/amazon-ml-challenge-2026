
import csv
import duckdb
from collections import defaultdict
from rapidfuzz import fuzz
import time

from config import OUTPUT_DIR, DB_PATH, TEMP_DIR

# ==========================================
# CONFIGURATION
# ==========================================

CHEAP_RANK_LIMIT = 2000
BATCH_SIZE = 50
TOP_K_VALUES = [200, 500, 1000]

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
# GROUND TRUTH
# ==========================================

def load_ground_truth(con):
    rows = con.execute("""
        SELECT
            source1_entity_id,
            matched_entity_ids
        FROM validation_gt
    """).fetchall()

    ground_truth = {}

    for source1_id, matches in rows:
        if matches and matches.strip():
            ground_truth[source1_id] = {
                x.strip()
                for x in matches.split(",")
                if x.strip()
            }
        else:
            ground_truth[source1_id] = set()

    return ground_truth


# ==========================================
# INITIALIZE OUTPUTS
# ==========================================

def initialize_outputs():
    for k in TOP_K_VALUES:
        path = (
            OUTPUT_DIR
            / f"expanded_rerank_top{k}.tsv"
        )

        # Start fresh for this experiment.
        if path.exists():
            path.unlink()

        with open(
            path,
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


# ==========================================
# RAPIDFUZZ SCORING
# ==========================================

def similarity_score(name1, name2, address1, address2):
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

    return (
        0.70 * name_score
        + 0.30 * address_score
    )


# ==========================================
# PROCESS ONE BATCH
# ==========================================

def process_batch(con, batch_ids):
    con.execute("""
        CREATE OR REPLACE TEMP TABLE current_batch (
            entity_id VARCHAR
        )
    """)

    con.executemany(
        "INSERT INTO current_batch VALUES (?)",
        [(x,) for x in batch_ids]
    )

    # Only candidates within the expanded cheap-rank
    # limit are fetched for fuzzy reranking.
    rows = con.execute("""
        SELECT
            r.source1_entity_id,
            r.candidate_entity_id,
            r.source_label,
            f.name1,
            f.name2,
            f.address1,
            f.address2

        FROM cheap_candidate_ranks r

        INNER JOIN current_batch b
            ON r.source1_entity_id = b.entity_id

        INNER JOIN pruning_features f
            ON r.source1_entity_id =
               f.source1_entity_id
            AND r.candidate_entity_id =
                f.candidate_entity_id
            AND r.source_label = f.source_label

        WHERE r.cheap_rank <= ?

    """, [CHEAP_RANK_LIMIT]).fetchall()

    # Deduplicate candidate IDs across source2/source3
    # while keeping the highest score.
    scored = defaultdict(dict)

    for row in rows:
        (
            source1_id,
            candidate_id,
            source_label,
            name1,
            name2,
            address1,
            address2
        ) = row

        score = similarity_score(
            name1,
            name2,
            address1,
            address2
        )

        old_score = scored[source1_id].get(
            candidate_id
        )

        if old_score is None or score > old_score:
            scored[source1_id][candidate_id] = score

    # Sort candidates per source1 business.
    ordered = {}

    for source1_id in batch_ids:
        candidates = scored.get(
            source1_id,
            {}
        )

        ordered[source1_id] = sorted(
            candidates.items(),
            key=lambda x: (-x[1], x[0])
        )

    return ordered


# ==========================================
# WRITE OUTPUTS
# ==========================================

def write_batch_outputs(batch_ids, ordered):
    for k in TOP_K_VALUES:
        path = (
            OUTPUT_DIR
            / f"expanded_rerank_top{k}.tsv"
        )

        with open(
            path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(
                f,
                delimiter="\t"
            )

            for source1_id in batch_ids:
                selected = ordered.get(
                    source1_id,
                    []
                )[:k]

                candidate_ids = [
                    candidate_id
                    for candidate_id, score in selected
                ]

                writer.writerow([
                    source1_id,
                    ",".join(candidate_ids)
                ])


# ==========================================
# EVALUATION
# ==========================================

def evaluate_results(ground_truth):
    true_total = sum(
        len(matches)
        for matches in ground_truth.values()
    )

    print("\n========== EXPANDED RERANK RESULTS ==========")
    print(f"Ground-truth matches: {true_total:,}")

    for k in TOP_K_VALUES:
        path = (
            OUTPUT_DIR
            / f"expanded_rerank_top{k}.tsv"
        )

        recovered = 0
        total_candidates = 0
        row_count = 0

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t"
            )

            for row in reader:
                source1_id = row["source1_entity_id"]

                candidate_string = (
                    row["candidate_entity_ids"] or ""
                )

                candidate_ids = {
                    x.strip()
                    for x in candidate_string.split(",")
                    if x.strip()
                }

                total_candidates += len(candidate_ids)
                row_count += 1

                recovered += len(
                    candidate_ids
                    & ground_truth.get(
                        source1_id,
                        set()
                    )
                )

        recall = (
            recovered / true_total
            if true_total
            else 0
        )

        average = (
            total_candidates / row_count
            if row_count
            else 0
        )

        print(f"\nTOP {k}")
        print(f"Recovered matches: {recovered:,}")
        print(f"Candidate recall: {recall:.4%}")
        print(f"Total candidates: {total_candidates:,}")
        print(
            f"Average candidates/business: "
            f"{average:.2f}"
        )
        print(f"Saved: {path}")


# ==========================================
# MAIN
# ==========================================

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    con = connect_db()

    try:
        # Verify required tables exist.
        required_tables = [
            "cheap_candidate_ranks",
            "pruning_features",
            "validation_s1",
            "validation_gt"
        ]

        existing_tables = {
            row[0]
            for row in con.execute("""
                SHOW TABLES
            """).fetchall()
        }

        missing = [
            table
            for table in required_tables
            if table not in existing_tables
        ]

        if missing:
            raise RuntimeError(
                "Missing required DuckDB tables: "
                + ", ".join(missing)
                + ". Run the previous diagnostic "
                  "scripts first."
            )

        ground_truth = load_ground_truth(con)

        initialize_outputs()

        all_ids = [
            row[0]
            for row in con.execute("""
                SELECT entity_id
                FROM validation_s1
                ORDER BY entity_id
            """).fetchall()
        ]

        print(
            f"Validation businesses: {len(all_ids):,}"
        )

        print(
            f"Cheap rank cutoff per source: "
            f"{CHEAP_RANK_LIMIT:,}"
        )

        start = time.time()

        for offset in range(
            0,
            len(all_ids),
            BATCH_SIZE
        ):
            batch_ids = all_ids[
                offset:offset + BATCH_SIZE
            ]

            ordered = process_batch(
                con,
                batch_ids
            )

            write_batch_outputs(
                batch_ids,
                ordered
            )

            processed = min(
                offset + len(batch_ids),
                len(all_ids)
            )

            print(
                f"Processed {processed:,}/"
                f"{len(all_ids):,} businesses "
                f"({time.time() - start:.1f}s)"
            )

        evaluate_results(ground_truth)

    finally:
        con.close()


if __name__ == "__main__":
    main()