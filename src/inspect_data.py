import pandas as pd

from config import TEST_DIR

file_path = TEST_DIR / "test_source3.tsv"

# Read only the first 10 rows
df = pd.read_csv(
    file_path,
    sep="\t",
    nrows=10
)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.to_string(index=False))

# Get total number of rows without loading everything
with open(file_path, "r", encoding="utf-8") as f:
    row_count = sum(1 for _ in f) - 1

print("\nTotal records:", row_count)