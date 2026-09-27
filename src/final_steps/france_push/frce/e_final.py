"""Final evaluation of the French cross-encoder and writing of the removal set.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $REPO_DIR/student_resource/dataset/test
    {W}/frce_hold.parquet
    {W}/score_h.parquet
    {SC}/sc_h.parquet
    {SC}/hfr_ce1.parquet
    {W}/score_hfr.parquet
    {W}/labels60.parquet
    {SC}/lab_frce.parquet
    {SP}/ce/test_{m}.parquet
    {R}/test_source1.tsv
    {R}/test_source2.tsv
    {R}/test_source3.tsv
"""
import os
import sys
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
PYTHON = os.environ.get("PYTHON", sys.executable)
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys, random
import numpy as np, pyarrow.parquet as pq, duckdb
from sklearn.metrics import roc_auc_score
SP = f"{WORK_DIR}"
W = f"{SP}/push/frce"
R = f"{REPO_DIR}/student_resource/dataset/test"
ID = "cast(substr(entity_id, 2, 1) as bigint) * 10000000000 + cast(substr(entity_id, 4) as bigint)"
WRITE = "--write" in sys.argv
SC = os.environ.get("SCDIR", W)
c = duckdb.connect()
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp/spill2'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
have = lambda f: os.path.exists(f"{SC}/{f}")
parts = [f"'{SC}/{f}'" for f in ("sc_a.parquet", "sc_b.parquet") if have(f)]
c.execute(f"create table fr as select rid, s1, any_value(ce) ce from read_parquet([{','.join(parts)}]) group by all")
print("frCE scored pairs", c.execute("select count(*) from fr").fetchone()[0])
def aucs(tag, kind, y, s, groups):
    """Print AUC and accuracy of the scores for each group of pairs."""
    for name, m in groups:
        if m.sum() and len(set(y[m])) == 2:
            print(f"  {tag} {name}: n={int(m.sum())} auc={roc_auc_score(y[m], s[m]):.4f} acc={np.mean((s[m] > 0) == (y[m] == 1)):.4f}")
    print(f"  {tag} per kind acc:", [(k, int((kind == k).sum()), round(float(np.mean((s[kind == k] > 0) == (y[kind == k] == 1))), 3)) for k in sorted(set(kind))])
if have("sc_h.parquet"):
    print("== gate 1: holdout AUC")
    hold = pq.read_table(f"{W}/frce_hold.parquet"); sh = pq.read_table(f"{W}/score_h.parquet"); ch = pq.read_table(f"{SC}/sc_h.parquet")
    assert hold.column("q_text").to_pylist() == sh.column("q_text").to_pylist() and sh.column("s1").to_pylist() == ch.column("s1").to_pylist()
    kind = np.array(hold.column("kind").to_pylist()); y = hold.column("label").to_numpy(); s = ch.column("ce").to_numpy()
    us = np.char.startswith(kind.astype(str), "us_"); ps = np.char.startswith(kind.astype(str), "ps_")
    aucs("frCE", kind, y, s, (("all", np.ones(len(y), bool)), ("france", ~us), ("fr_pseudo", ps), ("fr_synth", ~us & ~ps), ("us_india", us)))
    if have("hfr_ce1.parquet"):
        h1 = pq.read_table(f"{SC}/hfr_ce1.parquet"); sf = pq.read_table(f"{W}/score_hfr.parquet")
        m = ~us
        assert sf.column("q_text").to_pylist() == [q for q, k in zip(hold.column("q_text").to_pylist(), m) if k]
        s1 = h1.column("ce").to_numpy()
        aucs("ce1", kind[m], y[m], s1, (("france", np.ones(m.sum(), bool)), ("fr_pseudo", ps[m]), ("fr_synth", ~ps[m])))
print("== gate 2: 60 hand labels")
c.execute(f"create table labf as select l.rid, l.s1, f.ce from '{W}/labels60.parquet' l join fr f using (rid, s1)")
c.execute(f"copy labf to '{SC}/lab_frce.parquet'")
os.system(f"cd {W} && {PYTHON} {HERE}/e_lab.py lab_ce1.parquet lab_ce2.parquet {SC}/lab_frce.parquet")
print("== gate 3: CE disagreement on French band (sign(ce>0) vs stage-2 p>=0.5)")
c.execute("create table bd as select b.rid, b.s1, b.p, b.ps, b.v, b.rid_asg, b.pair_asg, f.ce from w.bandv b left join fr f using (rid, s1)")
print("  frCE:", c.execute("select count(*), count(ce), avg(((ce > 0) <> (p >= 0.5))::int), avg(((ce > 0) <> (ps >= 0.5))::int) from bd where ce is not null").fetchall())
for m in ("ce1", "ce2"):
    print(f"  {m} (test_band overlap):", c.execute(f"""select count(*), avg(((t.ce > 0) <> (b.p >= 0.5))::int), avg(((b.ce > 0) <> (b.p >= 0.5))::int)
      from bd b join '{SP}/ce/test_{m}.parquet' t using (rid, s1) where b.ce is not null""").fetchall())
print("  frCE disagreement by verdict:", c.execute("select v, count(*), round(avg(((ce > 0) <> (p >= 0.5))::int), 3), round(avg((ce > 0)::int), 3) from bd where ce is not null group by 1 order by 2 desc").fetchall())
c.execute(f"""create table raw as select {ID} eid, business_name bn, business_address ba,
  (business_name = lower(business_name) and regexp_matches(business_name, '[a-z]'))::int lc, contains(business_name, '.')::int dot
  from read_csv(['{R}/test_source1.tsv','{R}/test_source2.tsv','{R}/test_source3.tsv'], delim='\\t', header=true, all_varchar=true, quote='', escape='')
  where lower(country)='france'""")
print("== rule")
c.execute("""create table adds as select rid, arg_max(s1, ce) s1, max(ce) ce, arg_max(p, ce) p, arg_max(v, ce) v from bd
  where p >= 0.85 and ce >= 0 and not rid_asg group by rid""")
c.execute("create table rems as select v.rid, v.s1, f.ce, v.v from w.v17v v join fr f using (rid, s1) where f.ce <= -2 and v.v like 'diff%'")
base = c.execute("select count(*), avg(r.lc), avg(r.dot) from w.v17fr v join raw r on r.eid = v.rid").fetchone()
print(f"  v17 French pairs: {base[0]} lc={base[1]*100:.2f}% dot={base[2]*100:.2f}%")
print("  v17 French lc/dot by verdict:", [(a, n, round(l*100, 2), round(d*100, 2)) for a, n, l, d in c.execute("select v.v, count(*), avg(r.lc), avg(r.dot) from w.v17v v join raw r on r.eid=v.rid group by 1 order by 2 desc").fetchall()])
res = {}
for name, t, goodv in (("additions", "adds", "('same','same_noise')"), ("removals", "rems", None)):
    n, lc, dot = c.execute(f"select count(*), avg(r.lc), avg(r.dot) from {t} x join raw r on r.eid = x.rid").fetchone()
    vs = c.execute(f"select v, count(*) from {t} group by 1 order by 2 desc").fetchall()
    frac_same = c.execute(f"select avg((v in ('same','same_noise'))::int), avg((v like 'diff%')::int) from {t}").fetchone()
    print(f"  {name}: n={n} lc={100*(lc or 0):.2f}% dot={100*(dot or 0):.2f}% same/same_noise={100*(frac_same[0] or 0):.1f}% diff_*={100*(frac_same[1] or 0):.1f}%")
    print(f"    verdicts: {vs}")
    print(f"    lc/dot by verdict:", [(a, k, round(l*100, 2), round(d*100, 2)) for a, k, l, d in c.execute(f"select x.v, count(*), avg(r.lc), avg(r.dot) from {t} x join raw r on r.eid=x.rid group by 1 order by 2 desc").fetchall()])
    res[name] = (n, lc or 0, frac_same)
    print(f"    20 random examples ({name}):")
    for r in c.execute(f"""select x.v, round(x.ce, 2), q.bn, q.ba, s.bn, s.ba from {t} x join raw q on q.eid = x.rid join raw s on s.eid = x.s1
        order by hash(x.rid, 4242) limit 20""").fetchall():
        print(f"      [{r[0]} ce={r[1]}] Q: {r[2]} | {r[3]}  ||  S1: {r[4]} | {r[5]}")
sys.path.insert(0, W)
from vd import review_address, review_key
DEL = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
for name, t in (("additions", "adds"), ("removals", "rems")):
    rows = c.execute(f"""select x.v, {review_key('q.address')}, {review_key('s.address')}, r.lc from {t} x join w.nm q on q.eid = x.rid join w.nm s on s.eid = x.s1 join raw r on r.eid = x.rid""").fetchall()
    d = [(v, review_address(a, b)[1], lc) for v, a, b, lc in rows]
    dd = [x[1] for x in d if x[1] is not None]
    if dd:
        pos = sum(1 for x in dd if x > 0 and x in DEL); neg = sum(1 for x in dd if x < 0)
        print(f"  {name} house-number deltas (q - s1): n={len(dd)} positive-offset-set={pos} ({100*pos/len(dd):.1f}%) negative={neg} ({100*neg/len(dd):.1f}%)")
    wc = [x for x in d if x[0] in ("diff_typeswap", "same_noise", "unsure_addword")]
    if wc:
        print(f"  {name} word-changed subset (diff_typeswap/same_noise/unsure_addword): n={len(wc)} lc={100*sum(x[2] for x in wc)/len(wc):.2f}%")
na, lca, fa = res["additions"]; nr, lcr, fr_ = res["removals"]
ga = lca <= 0.003 and (fa[0] or 0) >= 0.70
gr = lcr >= 0.02 and (fr_[1] or 0) >= 0.70
print(f"GATE additions: {'PASS' if ga else 'FAIL'} (lc {lca*100:.2f}% <= 0.3%, same/same_noise {100*(fa[0] or 0):.1f}% >= 70%)")
print(f"GATE removals: {'PASS' if gr else 'FAIL'} (lc {lcr*100:.2f}% >= 2%, diff_* {100*(fr_[1] or 0):.1f}% >= 70%)")
print("rescue (French, unassigned in v17):", c.execute("""select count(*), count(f.ce), avg((f.ce >= 0)::int) from w.resc r left join fr f using (rid, s1) where not r.rid_asg""").fetchall())
if WRITE:
    if ga:
        c.execute(f"copy (select rid, s1 from adds order by rid) to '{W}/additions.parquet'")
    if gr:
        c.execute(f"copy (select rid, s1 from rems order by rid) to '{W}/removals.parquet'")
    print("written:", [f for f in ("additions.parquet", "removals.parquet") if os.path.exists(f"{W}/{f}")])
