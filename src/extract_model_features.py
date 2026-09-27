
from pathlib import Path

import duckdb
import pandas as pd
from rapidfuzz import fuzz
import pyarrow.parquet as pq

INPUT_FILE = Path("data/training/labeled_pairs.parquet")
OUTPUT_DIR = Path("data/training/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 25_000

def similarity_features(df):
    features = pd.DataFrame(index=df.index)

    # Basic candidate-ranking features
    features["cheap_score"] = df["cheap_score"].fillna(0)
    features["cheap_rank"] = df["cheap_rank"].fillna(999999)
    features["source_s2"] = (df["source_label"] == "s2").astype("int8")
    features["source_s3"] = (df["source_label"] == "s3").astype("int8")

    # Text normalization and missing-value handling
    name1 = df["name1"].fillna("").astype(str)
    name2 = df["name2"].fillna("").astype(str)
    address1 = df["address1"].fillna("").astype(str)
    address2 = df["address2"].fillna("").astype(str)

    # Name similarity
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

    # Address similarity
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

    # Exact matches and containment
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

    # Text lengths and token overlap
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

    def token_overlap(a, b):
        ta = set(a.split())
        tb = set(b.split())

        if not ta or not tb:
            return 0.0

        return len(ta & tb) / len(ta | tb)

    features["name_token_overlap"] = [
        token_overlap(a, b)
        for a, b in zip(name1, name2)
    ]

    features["address_token_overlap"] = [
        token_overlap(a, b)
        for a, b in zip(address1, address2)
    ]

    return features


def main():
    parquet = pq.ParquetFile(INPUT_FILE)

    total_rows = parquet.metadata.num_rows
    print(f"Input rows: {total_rows:,}")

    # Create separate output files for train and validation.
    train_file = OUTPUT_DIR / "train_features.parquet"
    valid_file = OUTPUT_DIR / "valid_features.parquet"

    for path in [train_file, valid_file]:
        if path.exists():
            path.unlink()

    train_writer = None
    valid_writer = None

    try:
        for batch_number, batch in enumerate(
            parquet.iter_batches(batch_size=BATCH_SIZE),
            start=1
        ):
            df = batch.to_pandas()

            X = similarity_features(df)

            # Preserve identifiers and labels for model training.
            output = pd.concat(
                [
                    df[
                        [
                            "source1_entity_id",
                            "candidate_entity_id",
                            "source_label",
                            "label",
                            "split",
                        ]
                    ].reset_index(drop=True),
                    X.reset_index(drop=True),
                ],
                axis=1,
            )

            train = output[output["split"] == "train"]
            valid = output[output["split"] == "valid"]

            if not train.empty:
                table = __import__("pyarrow").Table.from_pandas(
                    train, preserve_index=False
                )

                if train_writer is None:
                    train_writer = __import__(
                        "pyarrow.parquet", fromlist=["ParquetWriter"]
                    ).ParquetWriter(
                        train_file,
                        table.schema,
                        compression="zstd",
                    )

                train_writer.write_table(table)

            if not valid.empty:
                table = __import__("pyarrow").Table.from_pandas(
                    valid, preserve_index=False
                )

                if valid_writer is None:
                    valid_writer = __import__(
                        "pyarrow.parquet", fromlist=["ParquetWriter"]
                    ).ParquetWriter(
                        valid_file,
                        table.schema,
                        compression="zstd",
                    )

                valid_writer.write_table(table)

            print(
                f"Batch {batch_number}: "
                f"{len(df):,} processed | "
                f"{len(train):,} train | "
                f"{len(valid):,} valid"
            )

    finally:
        if train_writer is not None:
            train_writer.close()

        if valid_writer is not None:
            valid_writer.close()

    print("\nFeature extraction completed.")
    print(f"Train features: {train_file}")
    print(f"Validation features: {valid_file}")


if __name__ == "__main__":
    main()