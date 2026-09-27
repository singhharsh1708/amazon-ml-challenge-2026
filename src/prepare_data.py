import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz
from collections import defaultdict
import random
import re
import time

from config import TRAIN_DIR, OUTPUT_DIR

# =========================
# CONFIGURATION
# =========================

SAMPLE_SIZE = 5000
RANDOM_SEED = 42

SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
SOURCE2_FILE = TRAIN_DIR / "train_source2.tsv"
SOURCE3_FILE = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# NORMALIZATION
# =========================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


# =========================
# LOAD DATA
# =========================

def load_data():
    print("Loading training data...")

    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    source2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    source3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Loaded:")
    print("Source1:", len(source1))
    print("Source2:", len(source2))
    print("Source3:", len(source3))
    print("Ground truth:", len(ground_truth))

    return source1, source2, source3, ground_truth


# =========================
# SAMPLE VALIDATION DATA
# =========================

def create_validation_sample(source1, ground_truth):
    print("\nCreating validation sample...")

    random.seed(RANDOM_SEED)

    available_ids = set(source1["entity_id"])

    valid_gt = ground_truth[
        ground_truth["source1_entity_id"].isin(available_ids)
    ]

    sample_ids = random.sample(
        list(valid_gt["source1_entity_id"]),
        min(SAMPLE_SIZE, len(valid_gt))
    )

    validation_s1 = source1[
        source1["entity_id"].isin(sample_ids)
    ].copy()

    validation_gt = valid_gt[
        valid_gt["source1_entity_id"].isin(sample_ids)
    ].copy()

    print("Validation Source1:", len(validation_s1))

    return validation_s1, validation_gt


# =========================
# PREPARE CANDIDATE INDEX
# =========================

def prepare_candidates(source):
    print("Preparing candidate index...")

    source = source.copy()

    source["normalized_name"] = source["business_name"].map(
        normalize_text
    )

    source["normalized_address"] = source["business_address"].map(
        normalize_text
    )

    source["normalized_country"] = source["country"].map(
        normalize_text
    )

    # Index records by country and first characters of name
    index = defaultdict(list)

    for row in source.itertuples(index=False):
        name = row.normalized_name
        country = row.normalized_country

        if not name:
            continue

        key = (country, name[:3])
        index[key].append(row)

    print("Index groups:", len(index))

    return source, index


# =========================
# GENERATE CANDIDATES
# =========================

def generate_candidates(query, source, index):
    name = normalize_text(query["business_name"])
    country = normalize_text(query["country"])

    if not name:
        return []

    candidates = []

    # Blocking on country + first 3 normalized name characters
    key = (country, name[:3])

    candidates.extend(index.get(key, []))

    # If the block is empty, fall back to same-country records
    if not candidates:
        country_records = source[
            source["normalized_country"] == country
        ]

        candidates = list(
            country_records.itertuples(index=False)
        )

    if not candidates:
        return []

    candidate_names = [
        row.normalized_name for row in candidates
    ]

    matches = process.extract(
        name,
        candidate_names,
        scorer=fuzz.ratio,
        limit=10,
        score_cutoff=40
    )

    results = []

    for _, score, idx in matches:
        row = candidates[idx]

        results.append({
            "entity_id": row.entity_id,
            "score": score
        })

    return results


# =========================
# EVALUATION
# =========================

def evaluate_candidates(validation_s1, validation_gt,
                        source2, source3):

    print("\nBuilding candidate indexes...")

    source2, index2 = prepare_candidates(source2)
    source3, index3 = prepare_candidates(source3)

    gt_map = {}

    for row in validation_gt.itertuples(index=False):
        ids = [
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        ]

        gt_map[row.source1_entity_id] = set(ids)

    total_true = 0
    total_recovered = 0
    candidate_count = 0

    results = []

    start = time.time()

    print("\nGenerating candidates...")

    for i, row in enumerate(
        validation_s1.itertuples(index=False), start=1
    ):

        query = {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country
        }

        candidates2 = generate_candidates(
            query, source2, index2
        )

        candidates3 = generate_candidates(
            query, source3, index3
        )

        candidate_ids = {
            c["entity_id"] for c in candidates2
        } | {
            c["entity_id"] for c in candidates3
        }

        true_ids = gt_map.get(row.entity_id, set())

        recovered = candidate_ids & true_ids

        total_true += len(true_ids)
        total_recovered += len(recovered)
        candidate_count += len(candidate_ids)

        results.append({
            "source1_entity_id": row.entity_id,
            "candidate_entity_ids": ",".join(
                sorted(candidate_ids)
            ),
            "true_match_count": len(true_ids),
            "recovered_match_count": len(recovered)
        })

        if i % 500 == 0:
            print(
                f"Processed {i}/{len(validation_s1)} "
                f"| Candidates: {candidate_count} "
                f"| Elapsed: {time.time() - start:.1f}s"
            )

    recall = (
        total_recovered / total_true
        if total_true else 0
    )

    avg_candidates = (
        candidate_count / len(validation_s1)
        if len(validation_s1) else 0
    )

    output_file = OUTPUT_DIR / "validation_candidates.tsv"

    pd.DataFrame(results).to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print("\n========== VALIDATION RESULTS ==========")
    print("Validation businesses:", len(validation_s1))
    print("True matches:", total_true)
    print("Recovered true matches:", total_recovered)
    print(f"Candidate recall: {recall:.4f}")
    print(f"Average candidates per business: {avg_candidates:.2f}")
    print("Saved:", output_file)


# =========================
# MAIN
# =========================

def main():
    source1, source2, source3, ground_truth = load_data()

    validation_s1, validation_gt = create_validation_sample(
        source1, ground_truth
    )

    evaluate_candidates(
        validation_s1,
        validation_gt,
        source2,
        source3
    )


if __name__ == "__main__":
    main()