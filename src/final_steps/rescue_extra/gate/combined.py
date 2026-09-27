"""Step 6 gate. Experiment/validation script combined.py for this step.

Step context: Step 6 gate. Validation of the combined rescue families on the held-out split and the final nm1d addition set (nm1d_adds.parquet).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/top50/gate
    {G}/k20_cr.parquet
    {G}/oof_{t}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
G = f'{WORK_DIR}/top50/gate'
c = duckdb.connect(f'{G}/err.duckdb', read_only=True)
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{G}/tmp'")
W = 1.887
c.execute(f"CREATE TEMP TABLE rv AS SELECT rid, s1, ts FROM '{G}/k20_cr.parquet' WHERE p >= 0.6 AND nsmax < 0.2")
def oof(t, th):
    """Return SQL selecting the best out-of-fold pair per record above a threshold."""
    return f"SELECT rid, arg_max(s1, ps) s1, arg_max(ts, ps) ts FROM '{G}/oof_{t}.parquet' GROUP BY rid HAVING max(ps) >= {th}"
c.execute(f"CREATE TEMP TABLE nm AS {oof('nm1d', 0.7)}")
c.execute(f"CREATE TEMP TABLE hc AS {oof('hc', 0.7)}")
print('reverse valid adds', c.execute("SELECT count(*) FROM rv").fetchone(), 'rid overlap with nm1d', c.execute("SELECT count(*), count(*) FILTER (WHERE rv.s1 = nm.s1) FROM rv JOIN nm USING (rid)").fetchone(),
      'with hc', c.execute("SELECT count(*), count(*) FILTER (WHERE rv.s1 = hc.s1) FROM rv JOIN hc USING (rid)").fetchone(),
      'rv rid in fa', c.execute("SELECT count(*) FROM rv SEMI JOIN fa USING (rid)").fetchone())
halves = {f"q{i}": f"hash(v.s1 * 7 + 3) % 4 = {i}" for i in range(4)}
def F(parts, half):
    """Return held-out precision, recall and F when the given extra pairs are added."""
    u = " UNION ALL ".join(f"SELECT rid, s1, ts, {i} pr FROM {p}" for i, p in enumerate(parts))
    add = f"SELECT rid, arg_min(s1, pr) s1, arg_min(ts, pr) ts FROM ({u}) GROUP BY rid" if parts else "SELECT NULL::bigint rid, NULL::bigint s1, NULL::bigint ts WHERE false"
    return c.execute(f"""WITH AA AS (SELECT rid, s1, ts FROM fa UNION ALL SELECT * FROM ({add})),
      per AS (SELECT v.s1, v.n_true, count(a.rid) FILTER (WHERE a.ts = a.s1) tp, count(a.rid) FILTER (WHERE a.ts IS NULL) fpd,
        count(a.rid) FILTER (WHERE a.ts <> a.s1) fpw FROM vs1 v LEFT JOIN AA a USING (s1) WHERE {half} GROUP BY ALL)
      SELECT avg(CASE WHEN n_true=0 THEN greatest(0, 1-fpw-{W}*fpd) WHEN tp=0 THEN 0.0 ELSE 1.25*tp/(1.25*tp+0.25*(n_true-tp)+fpw+{W}*fpd) END) FROM per""").fetchone()[0]
cfg = {'base': [], 'nm1d': ['nm'], 'nm1d+hc': ['nm', 'hc'], 'nm1d+rev': ['nm', 'rv'], 'nm1d+hc+rev': ['nm', 'hc', 'rv']}
res = {k: {h: F(p, q) for h, q in halves.items()} for k, p in cfg.items()}
for k in cfg:
    print(f"{k:14}", ' '.join(f"{h} {res[k][h]:.6f} ({res[k][h]-res['base'][h]:+.6f})" for h in halves))
for a, b in [('nm1d+hc', 'nm1d'), ('nm1d+rev', 'nm1d'), ('nm1d+hc+rev', 'nm1d+hc')]:
    print(f"increment {a} over {b}:", ' '.join(f"{h} {res[a][h]-res[b][h]:+.6f}" for h in halves))
for grp in ["hash(s1 * 7 + 3) % 4", "country"]:
    pass
