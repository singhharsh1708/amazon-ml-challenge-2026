"""France re-run step 1: normalizes France records with the fixed address normalization.

Step context: Step 4 input (France re-run). Re-runs the France slice of the pipeline with the fixed address normalization and writes france_assign_partial.parquet.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {R}/data/norm/test_s{s}.parquet

Outputs:
    {F}/data/norm/test_s{s}.parquet
    {F}/data/fr_new/test_s{s}.parquet
    {F}/data/fr_old/test_s{s}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from frc import F, R, SP, ID, FIXES, con, raw, Timer

sys.path.insert(0, f"{FINAL_STEPS}/france_rerun/norm")
from norm_v2 import address_v2


def work_chunk(addrs):
    """Normalize a chunk of addresses with and without the fixes."""
    return [address_v2(a, set()) for a in addrs], [address_v2(a, FIXES) for a in addrs]


def main():
    """Parse the command line and run the step."""
    T = Timer()
    os.makedirs(f"{F}/data/fr_new", exist_ok=True)
    os.makedirs(f"{F}/data/fr_old", exist_ok=True)
    c = con("norm")
    with Pool(3) as pool:
        for s in (1, 2, 3):
            old = f"{R}/data/norm/test_s{s}.parquet"
            rows = c.execute(f"""
                SELECT r.entity_id, r.business_address FROM {raw(s)} r WHERE lower(r.country) = 'france'
            """).fetchall()
            ids = [x[0] for x in rows]
            addrs = [x[1] for x in rows]
            chunks = [addrs[i:i + 50000] for i in range(0, len(addrs), 50000)]
            base, new = [], []
            for b, n in pool.imap(work_chunk, chunks):
                base.extend(b)
                new.extend(n)
            c.register("na", pa.table({"entity_id": ids, "addr_base": base, "addr_new": new}))
            chk = c.execute(f"""
                SELECT count(*), count(*) FILTER (WHERE o.address IS DISTINCT FROM na.addr_base),
                    count(*) FILTER (WHERE o.address IS DISTINCT FROM na.addr_new),
                    count(*) FILTER (WHERE o.country <> 'france')
                FROM na JOIN read_parquet('{old}') o USING (entity_id)
            """).fetchone()
            nfr = c.execute(f"SELECT count(*) FROM read_parquet('{old}') WHERE country = 'france'").fetchone()[0]
            T(f"s{s}: raw france {len(ids):,}, norm france {nfr:,}, joined {chk[0]:,}, base mismatches {chk[1]:,}, "
              f"changed by fixes {chk[2]:,}, country mismatch {chk[3]:,}")
            cols = "entity_id, country, name_full, name_core, address, name_translit, address_missing"
            newsel = f"""
                SELECT o.entity_id, o.country, o.name_full, o.name_core,
                    CASE WHEN o.country = 'france' THEN na.addr_new ELSE o.address END AS address,
                    o.name_translit,
                    CASE WHEN o.country = 'france' THEN na.addr_new = '' ELSE o.address_missing END AS address_missing
                FROM read_parquet('{old}') o LEFT JOIN na USING (entity_id)
            """
            c.execute(f"COPY ({newsel}) TO '{F}/data/norm/test_s{s}.parquet' (FORMAT parquet, COMPRESSION zstd)")
            c.execute(f"COPY (SELECT {cols} FROM ({newsel}) WHERE country = 'france') TO '{F}/data/fr_new/test_s{s}.parquet' (FORMAT parquet, COMPRESSION zstd)")
            c.execute(f"COPY (SELECT {cols} FROM read_parquet('{old}') WHERE country = 'france') TO '{F}/data/fr_old/test_s{s}.parquet' (FORMAT parquet, COMPRESSION zstd)")
            c.unregister("na")
            v = c.execute(f"""
                SELECT count(*), count(*) FILTER (WHERE n.country <> 'france' AND (n.address IS DISTINCT FROM o.address
                        OR n.address_missing IS DISTINCT FROM o.address_missing OR n.name_full IS DISTINCT FROM o.name_full
                        OR n.name_core IS DISTINCT FROM o.name_core)),
                    count(*) FILTER (WHERE n.country = 'france' AND n.address IS DISTINCT FROM o.address),
                    count(*) FILTER (WHERE n.country = 'france' AND n.address IS NULL),
                    (SELECT count(*) FROM read_parquet('{old}'))
                FROM read_parquet('{F}/data/norm/test_s{s}.parquet') n JOIN read_parquet('{old}') o USING (entity_id)
            """).fetchone()
            T(f"s{s}: wrote {v[0]:,} rows (old file {v[4]:,}); non-France rows differing {v[1]:,}; France address changed {v[2]:,}; France null address {v[3]:,}")
    c.close()


if __name__ == "__main__":
    main()
