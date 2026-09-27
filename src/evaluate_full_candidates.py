
from pathlib import Path
from collections import defaultdict

import duckdb
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

DB_PATH = "data/entity_resolution.duckdb"
MODEL_PATH = "models/lightgbm_entity_matcher.joblib"

BATCH_SIZE = 25_000

THRESHOLDS = [
    0.50, 0.60, 0.70, 0.80, 0.85,
    0.90, 0.95, 0.97, 0.99,
]


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


def make_features(df):
    features = pd.DataFrame(index=df.index)

    name1 = df["name1"].map(safe_text)
    name2 = df["name2"].map(safe_text)
    address1 = df["address1"].map(safe_text)
    address2 = df["address2"].map(safe_text)

    features["cheap_score"] = (
        df["cheap_score"].fillna(0)
    )
    features["cheap_rank"] = (
        df["cheap_rank"].fillna(999999)
    )

    features["source_s2"] = (
        df["source_label"] == "s2"
    ).astype("int8")

    features["source_s3"] = (
        df["source_label"] == "s3"
    ).astype("int8")

    features["name_ratio"] = [
        fuzz.ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    features["name_token_sort"] = [
        fuzz.token_sort_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    features["name_token_set"] = [
        fuzz.token_set_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    features["name_partial"] = [
        fuzz.partial_ratio(a, b)
        for a, b in zip(name1, name2)
    ]

    features["address_ratio"] = [
        fuzz.ratio(a, b)
        for a, b in zip(address1, address2)
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
        token_overlap(a, b)
        for a, b in zip(name1, name2)
    ]

    features["address_token_overlap"] = [
        token_overlap(a, b)
        for a, b in zip(address1, address2)
    ]

    return features


def group_f05(tp, fp, fn):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    denominator = 1.25 * tp + 0.25 * fn + fp

    if denominator == 0:
        return 0.0

    return 1.25 * tp / denominator


def main():
    print("Loading trained model...")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features_list = saved["features"]

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='data/temp'")

    # Full ground truth for the held-out 1,000 S1 groups.
    gt_rows = con.execute("""
        SELECT source1_entity_id, matched_entity_ids
        FROM validation_gt
        WHERE hash(source1_entity_id) % 5 = 0
    """).fetchall()

    truth = {}

    for s1_id, ids in gt_rows:
        if ids is None or ids == "":
            truth[s1_id] = set()
        else:
            truth[s1_id] = set(ids.split(","))

    print(f"Held-out S1 groups: {len(truth):,}")

    # Store TP, FP, and FN per threshold and S1 group.
    tp = {
        t: defaultdict(int)
        for t in THRESHOLDS
    }
    fp = {
        t: defaultdict(int)
        for t in THRESHOLDS
    }
    fn = {
        t: defaultdict(int)
        for t in THRESHOLDS
    }

    # Every candidate in the full candidate table for
    # held-out S1 groups is scored, not just sampled negatives.
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

    columns = [
        desc[0] for desc in cursor.description
    ]

    processed = 0

    print("\nScoring complete held-out candidate set...")

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)

        if not rows:
            break

        df = pd.DataFrame(
            rows,
            columns=columns,
        )

        X = make_features(df)
        X = X[features_list].astype("float32")

        probabilities = model.predict_proba(
            X,
            num_iteration=model.best_iteration_,
        )[:, 1]

        df["probability"] = probabilities

        for row in df.itertuples(index=False):
            s1_id = row.source1_entity_id
            candidate_id = row.candidate_entity_id
            probability = row.probability

            is_true = candidate_id in truth[s1_id]

            for threshold in THRESHOLDS:
                if probability >= threshold:
                    if is_true:
                        tp[threshold][s1_id] += 1
                    else:
                        fp[threshold][s1_id] += 1

        processed += len(df)

        if processed % 250_000 == 0:
            print(f"Scored {processed:,} pairs...")

    print(f"\nTotal pairs scored: {processed:,}")

    # Calculate false negatives per group.
    for threshold in THRESHOLDS:
        for s1_id, true_ids in truth.items():
            recovered = tp[threshold][s1_id]
            fn[threshold][s1_id] = (
                len(true_ids) - recovered
            )

    print("\n--- FULL HELD-OUT METRICS ---")

    results = []

    for threshold in THRESHOLDS:
        group_scores = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_predicted = 0

        for s1_id in truth:
            a = tp[threshold][s1_id]
            b = fp[threshold][s1_id]
            c = fn[threshold][s1_id]

            group_scores.append(group_f05(a, b, c))

            total_tp += a
            total_fp += b
            total_fn += c
            total_predicted += a + b

        macro_f05 = float(np.mean(group_scores))

        precision = (
            total_tp / total_predicted
            if total_predicted else 0.0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 0.0
        )

        results.append({
            "threshold": threshold,
            "macro_f05": macro_f05,
            "precision": precision,
            "recall": recall,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "predicted_matches": total_predicted,
        })

    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    output_path = Path(
        "output/full_candidate_threshold_results.tsv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"\nSaved threshold results: {output_path}")

    con.close()


if __name__ == "__main__":
    main()