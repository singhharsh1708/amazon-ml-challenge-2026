"""France re-run step 4: stacking of the France scores.

Step context: Step 4 input (France re-run). Re-runs the France slice of the pipeline with the fixed address normalization and writes france_assign_partial.parquet.

Command-line arguments used: argv[1].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {CE}/{f}.parquet
    {w}/pred2.parquet
    {w}/pred1.parquet
    {CE}/test_stacked_v15_stacker.txt
    {w}/scores.parquet

Outputs:
    {w}/stacked.parquet
"""
import sys

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from frc import F, SP, con, work, Timer

CE = f"{SP}/ce"
CES = ["test_ce1", "test_ce2", "test_kbase"]


def lg(p):
    """Return the logit of a clipped probability."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(variant):
    """Parse the command line and run the step."""
    T = Timer()
    w = work(variant)
    c = con(f"stack_{variant}")
    tsel = ", ".join(f"e{i}.ce AS ce_{i}" for i in range(len(CES)))
    tjoin = " ".join(f"JOIN '{CE}/{f}.parquet' e{i} USING (rid, s1)" for i, f in enumerate(CES))
    c.execute(f"""CREATE TABLE band AS SELECT rid, s1, p FROM (
        SELECT rid, s1, p, row_number() OVER (PARTITION BY rid ORDER BY p DESC) rk FROM read_parquet('{w}/pred2.parquet'))
        WHERE rk <= 2 AND p BETWEEN 0.01 AND 0.99""")
    t = c.execute(f"""SELECT b.rid, b.s1, b.p AS p2, {tsel}, f.p AS p1, f.address_missing::int AS am, f.source, f.rk
        FROM band b {tjoin} JOIN read_parquet('{w}/pred1.parquet') f USING (rid, s1)""").fetchnumpy()
    nband = c.execute("SELECT count(*) FROM band").fetchone()[0]
    m = lgb.Booster(model_file=f"{CE}/test_stacked_v15_stacker.txt")
    Xt = np.column_stack([lg(t["p2"])] + [t[f"ce_{i}"] for i in range(len(CES))] + [lg(t["p1"]), t["am"], t["source"], t["rk"]]).astype(np.float32)
    ps = m.predict(Xt).astype(np.float32)
    pq.write_table(pa.table({"rid": t["rid"], "s1": t["s1"], "ps": ps}), f"{w}/stacked.parquet")
    T(f"{variant}: band {nband:,} pairs, {len(ps):,} with all three CE scores; mean p2 {t['p2'].mean():.4f} -> ps {ps.mean():.4f}; "
      f"cross 0.85 up {int(((t['p2'] < 0.85) & (ps >= 0.85)).sum()):,} down {int(((t['p2'] >= 0.85) & (ps < 0.85)).sum()):,}")
    c.execute(f"""COPY (SELECT n.rid, n.s1, least(n.p, coalesce(st.ps, n.p))::float AS p, n.p_guard::float AS p_guard,
            n.p AS p_s2, st.ps, n.p1, n.fne
        FROM read_parquet('{w}/pred2.parquet') n LEFT JOIN read_parquet('{w}/stacked.parquet') st USING (rid, s1))
        TO '{w}/scores.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    T(f"{variant}: scores " + str(c.execute(f"SELECT count(*), count(ps), avg(p) FROM '{w}/scores.parquet'").fetchone()))
    c.close()


if __name__ == "__main__":
    main(sys.argv[1])
