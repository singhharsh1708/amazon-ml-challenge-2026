import duckdb
import time

from config import TRAIN_DIR, TEST_DIR, DATA_DIR, DB_PATH

# =========================
# CONFIGURATION
# =========================

DATA_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 250_000

# =========================
# DATABASE CONNECTION
# =========================

def connect_db():
    con = duckdb.connect(str(DB_PATH))

    # Keep memory use controlled
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")

    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"SET temp_directory = '{temp_dir.as_posix()}'"
    )

    return con


# =========================
# IMPORT TSV FILE
# =========================

def import_tsv(con, table_name, file_path):
    print(f"\nImporting {table_name}")
    print(f"File: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    start = time.time()

    # Read all columns as strings to preserve entity IDs
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *
        FROM read_csv(
            '{file_path.as_posix()}',
            delim = '\\t',
            header = true,
            all_varchar = true,
            quote = '',
            escape = ''
        )
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0]

    print(f"Rows imported: {count:,}")
    print(f"Time: {time.time() - start:.1f}s")


# =========================
# CREATE NORMALIZED TABLES
# =========================

def create_normalized_table(con, source_table):
    normalized_table = f"{source_table}_norm"

    print(f"\nNormalizing {source_table}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {normalized_table} AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,

            lower(
                trim(
                    regexp_replace(
                        business_name,
                        '[^[:alnum:]]+',
                        ' ',
                        'g'
                    )
                )
            ) AS normalized_name,

            lower(
                trim(
                    regexp_replace(
                        business_address,
                        '[^[:alnum:]]+',
                        ' ',
                        'g'
                    )
                )
            ) AS normalized_address,

            lower(trim(country)) AS normalized_country

        FROM {source_table}
    """)

    print(f"Created: {normalized_table}")


# =========================
# MAIN
# =========================

def main():
    print("=" * 55)
    print("AMAZON ML CHALLENGE - CANDIDATE GENERATOR V2")
    print("=" * 55)

    con = connect_db()

    try:
        # Import reference data
        import_tsv(
            con,
            "train_source1",
            TRAIN_DIR / "train_source1.tsv"
        )

        # Import candidate sources
        import_tsv(
            con,
            "train_source2",
            TRAIN_DIR / "train_source2.tsv"
        )

        import_tsv(
            con,
            "train_source3",
            TRAIN_DIR / "train_source3.tsv"
        )

        # Import labels
        import_tsv(
            con,
            "ground_truth",
            TRAIN_DIR / "train_ground_truth.tsv"
        )

        # Create normalized tables
        for table in [
            "train_source1",
            "train_source2",
            "train_source3"
        ]:
            create_normalized_table(con, table)

        # Basic integrity check
        print("\n========== DATABASE SUMMARY ==========")

        tables = con.execute("""
            SHOW TABLES
        """).fetchall()

        for (table,) in tables:
            count = con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            print(f"{table}: {count:,}")

        print("\nDatabase saved at:")
        print(DB_PATH)

        print("\nDatabase preparation completed!")

    finally:
        con.close()


if __name__ == "__main__":
    main()