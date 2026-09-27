"""Held-out evaluation helpers: builds the validation split, loads scores, assigns matches and computes F.

Step context: Shared helpers for the validation scripts (DuckDB connection, held-out split loading, F-score).

Command-line arguments used: argv[1], argv[2].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {SCR}/truth.parquet
    {SCR}/valid_s1.parquet
    {SCR}/valid_s2.parquet
    {SCR}/runs/{tag}_oof.parquet
"""
import sys
from common import *
W = 1.887
THR = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def setup(c):
    """Create the ground-truth and validation-scope tables."""
    c.execute(f"CREATE OR REPLACE TABLE truth AS SELECT * FROM '{SCR}/truth.parquet'")
    c.execute(f"CREATE OR REPLACE TABLE vs1 AS SELECT * FROM '{SCR}/valid_s1.parquet'")
    c.execute(f"CREATE OR REPLACE TABLE vx AS SELECT rid, s1, p AS p1, coalesce(o_veto, 0) AS veto, first_num_equal AS fne FROM '{SCR}/valid_s2.parquet'")

def load(c, tag, src=None):
    """Load the out-of-fold scores of a run into a table."""
    src = src or f"'{SCR}/runs/{tag}_oof.parquet'"
    c.execute(f"""CREATE OR REPLACE TABLE best_{tag} AS SELECT b.rid, b.s1, b.p, x.p1, x.veto, x.fne, t.s1 AS true_s1
        FROM (SELECT rid, arg_max(s1, p2) AS s1, max(p2) AS p FROM {src} GROUP BY rid) b
        JOIN vx x USING (rid, s1) LEFT JOIN truth t ON t.rid = b.rid WHERE hash(b.s1) % 20 = 0""")

def assigned(c, tag, T, sib=False, extra_drop="false"):
    """Assign each record its best candidate above threshold T."""
    c.execute(f"CREATE OR REPLACE TEMP TABLE A AS SELECT * FROM best_{tag} WHERE p >= {T} AND veto = 0 AND NOT ({extra_drop})")
    if sib:
        c.execute("""DELETE FROM A k WHERE k.p < 0.95 AND k.fne = 0 AND EXISTS (SELECT 1 FROM A x WHERE x.s1 = k.s1 AND x.rid <> k.rid
            AND x.rid // 10000000000 = k.rid // 10000000000 AND x.fne = 1)""")

def fscore(c):
    """Return the held-out macro F score of the current assignment."""
    q = f"""WITH per AS (SELECT s.s1, s.country, s.n_true, count(a.rid) AS n_pred, count(a.rid) FILTER (WHERE a.true_s1 = a.s1) AS tp,
        count(a.rid) FILTER (WHERE a.true_s1 IS NULL) AS fpd, count(a.rid) FILTER (WHERE a.true_s1 <> a.s1) AS fpw
        FROM vs1 s LEFT JOIN A a ON a.s1 = s.s1 GROUP BY ALL)
    SELECT coalesce(country, 'ALL'),
      avg(CASE WHEN n_true=0 AND n_pred=0 THEN 1.0 WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+(n_pred-tp)) END),
      avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END),
      sum(n_pred), sum(tp), sum(fpd), sum(fpw)
    FROM per GROUP BY ROLLUP(country)"""
    return {r[0]: r[1:] for r in c.execute(q).fetchall()}

def table(c, tags, sib=False, thr=THR):
    """Return a table of held-out F for several runs."""
    rows = {}
    for T in thr:
        for tag in tags:
            assigned(c, tag, T, sib)
            rows[(T, tag)] = fscore(c)
    return rows

if __name__ == "__main__":
    tags = sys.argv[1].split(',')
    sib = len(sys.argv) > 2 and sys.argv[2] == '1'
    c = con('1500MB'); setup(c)
    for tag in tags:
        load(c, tag)
    res = table(c, tags, sib)
    ref = tags[0]
    print(f"sibling rule: {sib}; delta vs {ref}")
    for T in THR:
        for tag in tags:
            r = res[(T, tag)]; b = res[(T, ref)]
            a = r['ALL']
            print(f"T={T:.2f} {tag:18} exact {a[0]:.5f} ({a[0]-b['ALL'][0]:+.5f}) mix {a[1]:.5f} ({a[1]-b['ALL'][1]:+.5f}) | US {r['us'][0]:.5f}/{r['us'][1]:.5f} IN {r['india'][0]:.5f}/{r['india'][1]:.5f} | pred {a[2]} tp {a[3]} fpd {a[4]} fpw {a[5]}")
