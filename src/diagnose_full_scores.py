
from pathlib import Path
import gc

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

DB_PATH = "data/entity_resolution.duckdb"
MODEL_PATH = "models/lightgbm_entity_matcher.joblib"

BATCH_SIZE = 25_000

THRESHOLDS = [0.50, 0.90, 0.95, 0.99, 0.999]
TOP_K = [1, 3, 5, 10, 20, 50, 100, 200]

OUTPUT_PATH = Path("output/full_score_diagnostics.txt")
SCORED_PATH = Path("output/heldout_scored_temp.parquet")


def safe_text(value):
    if value is None:
        return ""
    return str(value)


def token_overlap(a, b):
    ta = set(a.split())
    tb = set(b.split())

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


# Exact same feature logic and ordering as the existing evaluator.
def make_features(df):
    features = pd.DataFrame(index=df.index)

    name1 = df["name1"].map(safe_text)
    name2 = df["name2"].map(safe_text)
    address1 = df["address1"].map(safe_text)
    address2 = df["address2"].map(safe_text)

    features["cheap_score"] = df["cheap_score"].fillna(0)
    features["cheap_rank"] = df["cheap_rank"].fillna(999999)

    features["source_s2"] = (
        df["source_label"] == "s2"
    ).astype("int8")

    features["source_s3"] = (
        df["source_label"] == "s3"
    ).astype("int8")

    features["name_ratio"] = [
        fuzz.ratio(a, b) for a, b in zip(name1, name2)
    ]
    features["name_token_sort"] = [
        fuzz.token_sort_ratio(a, b) for a, b in zip(name1, name2)
    ]
    features["name_token_set"] = [
        fuzz.token_set_ratio(a, b) for a, b in zip(name1, name2)
    ]
    features["name_partial"] = [
        fuzz.partial_ratio(a, b) for a, b in zip(name1, name2)
    ]

    features["address_ratio"] = [
        fuzz.ratio(a, b) for a, b in zip(address1, address2)
    ]
    features["address_token_sort"] = [
        fuzz.token_sort_ratio(a, b)
        for a, b in zip(address1, address2)
    ]
    features["address_token_set"] = [
        fuzz.token_set_ratio(a, b)
        for a, b in zip(address1, address2)
    ]
    features["address_partial"] = [
        fuzz.partial_ratio(a, b)
        for a, b in zip(address1, address2)
    ]

    features["name_exact"] = (
        (name1 != "") & (name1 == name2)
    ).astype("int8")

    features["address_exact"] = (
        (address1 != "") & (address1 == address2)
    ).astype("int8")

    features["name_contains"] = [
        int(bool(a and b and (a in b or b in a)))
        for a, b in zip(name1, name2)
    ]

    features["address_contains"] = [
        int(bool(a and b and (a in b or b in a)))
        for a, b in zip(address1, address2)
    ]

    features["name1_length"] = name1.str.len()
    features["name2_length"] = name2.str.len()
    features["address1_length"] = address1.str.len()
    features["address2_length"] = address2.str.len()

    features["name_length_diff"] = (
        name1.str.len() - name2.str.len()
    ).abs()

    features["address_length_diff"] = (
        address1.str.len() - address2.str.len()
    ).abs()

    features["name_token_overlap"] = [
        token_overlap(a, b) for a, b in zip(name1, name2)
    ]

    features["address_token_overlap"] = [
        token_overlap(a, b) for a, b in zip(address1, address2)
    ]

    return features


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading saved model...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_list = saved["features"]

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='data/temp'")

    print("Loading held-out ground truth...")

    gt_rows = con.execute("""
        SELECT source1_entity_id, matched_entity_ids
        FROM validation_gt
        WHERE hash(source1_entity_id) % 5 = 0
    """).fetchall()

    truth = {}

    for s1_id, ids in gt_rows:
        truth[s1_id] = (
            set(ids.split(","))
            if ids else set()
        )

    total_true = sum(len(v) for v in truth.values())
    positive_groups = sum(bool(v) for v in truth.values())

    print(f"Held-out groups: {len(truth):,}")
    print(f"Groups with matches: {positive_groups:,}")
    print(f"Total ground-truth matches: {total_true:,}")

    # Remove only the temporary output from a previous run.
    if SCORED_PATH.exists():
        SCORED_PATH.unlink()

    query = """
        SELECT
            f.source1_entity_id,
            f.candidate_entity_id,
            f.source_label,
            f.name1,
            f.name2,
            f.address1,
            f.address2,
            r.cheap_score,
            r.cheap_rank
        FROM pruning_features f
        JOIN cheap_candidate_ranks r
          ON f.source1_entity_id = r.source1_entity_id
         AND f.candidate_entity_id = r.candidate_entity_id
         AND f.source_label = r.source_label
        WHERE hash(f.source1_entity_id) % 5 = 0
        ORDER BY f.source1_entity_id,
                 f.source_label,
                 r.cheap_rank
    """

    cursor = con.execute(query)
    columns = [desc[0] for desc in cursor.description]

    positive_scores = []
    negative_scores = []

    threshold_positive = {t: 0 for t in THRESHOLDS}
    threshold_negative = {t: 0 for t in THRESHOLDS}

    processed = 0
    writer = None

    print("\nScoring full held-out candidate set...")

    try:
        while True:
            rows = cursor.fetchmany(BATCH_SIZE)

            if not rows:
                break

            df = pd.DataFrame(rows, columns=columns)

            X = make_features(df)
            X = X[feature_list].astype("float32")

            probabilities = model.predict_proba(
                X,
                num_iteration=model.best_iteration_,
            )[:, 1]

            labels = np.fromiter(
                (
                    int(candidate_id in truth[s1_id])
                    for s1_id, candidate_id
                    in zip(
                        df["source1_entity_id"],
                        df["candidate_entity_id"],
                    )
                ),
                dtype=np.int8,
                count=len(df),
            )

            pos_mask = labels == 1
            neg_mask = ~pos_mask

            positive_scores.extend(
                probabilities[pos_mask].tolist()
            )
            negative_scores.extend(
                probabilities[neg_mask].tolist()
            )

            for threshold in THRESHOLDS:
                threshold_positive[threshold] += int(
                    np.sum(
                        probabilities[pos_mask] >= threshold
                    )
                )

                threshold_negative[threshold] += int(
                    np.sum(
                        probabilities[neg_mask] >= threshold
                    )
                )

            # Save scored candidates temporarily for exact
            # within-group ranking using DuckDB.
            result = pd.DataFrame({
                "source1_entity_id": df["source1_entity_id"],
                "candidate_entity_id": df["candidate_entity_id"],
                "probability": probabilities,
                "label": labels,
            })

            table = pa.Table.from_pandas(
                result,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    SCORED_PATH,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

            processed += len(df)

            if processed % 250_000 == 0:
                print(f"Scored {processed:,} pairs...")

            del df, X, rows, result, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print(f"\nTotal pairs scored: {processed:,}")

    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)

    print(f"Positive candidates: {len(pos):,}")
    print(f"Negative candidates: {len(neg):,}")

    lines = []

    def add(text=""):
        print(text)
        lines.append(text)

    add("=" * 75)
    add("FULL HELD-OUT SCORE DIAGNOSTICS")
    add("=" * 75)

    add(f"Held-out source1 groups: {len(truth):,}")
    add(f"Groups with at least one true match: {positive_groups:,}")
    add(f"Total ground-truth matches: {total_true:,}")
    add(f"Total candidate pairs scored: {processed:,}")
    add(f"Positive candidate pairs: {len(pos):,}")
    add(f"Negative candidate pairs: {len(neg):,}")

    def score_summary(name, values, percentiles):
        add(f"\n--- {name} SCORE DISTRIBUTION ---")

        if len(values) == 0:
            add("No candidates.")
            return

        qs = np.percentile(values, percentiles)

        add(f"Minimum: {np.min(values):.8f}")
        add(f"Median:  {np.median(values):.8f}")
        add(f"Mean:    {np.mean(values):.8f}")

        for percentile, value in zip(percentiles, qs):
            add(f"{percentile}th percentile: {value:.8f}")

        add(f"Maximum: {np.max(values):.8f}")

    score_summary(
        "POSITIVE",
        pos,
        [90, 95, 99],
    )

    score_summary(
        "NEGATIVE",
        neg,
        [90, 95, 99, 99.9],
    )

    add("\n--- THRESHOLD ANALYSIS ---")
    add(
        "threshold\tpositive_above\tnegative_above"
        "\tpositive_recall"
    )

    for threshold in THRESHOLDS:
        tp = threshold_positive[threshold]
        fp = threshold_negative[threshold]

        recall = tp / total_true if total_true else 0.0

        add(
            f"{threshold:.3f}\t{tp:,}\t{fp:,}\t{recall:.6f}"
        )

    # Ground-truth matches per source1 group.
    group_match_counts = np.array(
        [len(ids) for ids in truth.values()],
        dtype=np.int64,
    )

    add("\n--- GROUND-TRUTH MATCHES PER SOURCE1 GROUP ---")
    add(f"Groups with zero matches: {np.sum(group_match_counts == 0):,}")
    add(f"Groups with one match: {np.sum(group_match_counts == 1):,}")
    add(f"Groups with 2-5 matches: {np.sum((group_match_counts >= 2) & (group_match_counts <= 5)):,}")
    add(f"Groups with 6-10 matches: {np.sum((group_match_counts >= 6) & (group_match_counts <= 10)):,}")
    add(f"Groups with more than 10 matches: {np.sum(group_match_counts > 10):,}")
    add(f"Mean matches per group: {np.mean(group_match_counts):.4f}")
    add(f"Median matches per group: {np.median(group_match_counts):.0f}")
    add(f"Maximum matches in a group: {np.max(group_match_counts)}")

    # Compute exact model ranks for true matches.
    add("\n--- TRUE-MATCH RECALL AT TOP K ---")
    add("Computing within-group candidate ranks with DuckDB...")

    rank_query = f"""
        WITH ranked AS (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                label,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY probability DESC,
                             candidate_entity_id
                ) AS model_rank
            FROM read_parquet('{SCORED_PATH.as_posix()}')
        )
        SELECT
            model_rank
        FROM ranked
        WHERE label = 1
    """

    true_ranks = con.execute(rank_query).fetchdf()[
        "model_rank"
    ].to_numpy(dtype=np.int64)

    for k in TOP_K:
        recovered = int(np.sum(true_ranks <= k))
        recall = recovered / total_true if total_true else 0.0

        add(
            f"Top {k:>3}: {recovered:,} / {total_true:,}"
            f" true matches | recall = {recall:.6f}"
        )

    # Save report.
    OUTPUT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {OUTPUT_PATH}")

    # Delete temporary scored parquet after ranking.
    con.close()

    if SCORED_PATH.exists():
        SCORED_PATH.unlink()

    print("Temporary scored parquet removed.")
    print("Existing model and database tables were not modified.")


if __name__ == "__main__":
    main()