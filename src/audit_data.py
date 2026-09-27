import pandas as pd
from config import TRAIN_DIR, TEST_DIR

FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}

for name, path in FILES.items():
    print(f"\n--- {name} ---")
    print(f"Path: {path}")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype=str,
        keep_default_na=False
    )

    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))