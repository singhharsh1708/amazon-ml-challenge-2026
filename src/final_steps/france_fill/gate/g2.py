"""Step 7 gate. Experiment/validation script g2.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {S}/fill/additions.parquet
    {G}/j_p07.parquet
    {S}/fill/frfill/pool_p05.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, re, random
sys.dont_write_bytecode = True
import duckdb
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
c = duckdb.connect()
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
rows = c.execute(f"""select j.*, p.p, p.ce, p.v, p.nc, p.ac, (j.rid in (select rid from '{S}/fill/additions.parquet')) in08
  from '{G}/j_p07.parquet' j left join '{S}/fill/frfill/pool_p05.parquet' p using (rid, s1)""").fetchall()
cols = [d[0] for d in c.description]
R = [dict(zip(cols, r)) for r in rows]
OFF = {1,2,3,4,5,7,9,11,13,21}
def num(a):
    """Return the leading street number of an address, or None."""
    if not a: return None
    m = re.match(r"\s*(\d+)", a)
    return int(m.group(1)) if m else None
def fp(rs):
    """Print the false-positive profile of a list of candidate rows."""
    n = len(rs)
    lc = [r for r in rs if r["qn"] and r["qn"] == r["qn"].lower() and re.search(r"[a-z]", r["qn"])]
    lc_ne = [r for r in lc if r["qn"] != r["sn"].lower()]
    dq = sum(r["qn"].count(".") for r in rs); ds = sum(r["sn"].count(".") for r in rs)
    dotrec = sum(1 for r in rs if "." in r["qn"]); dots1 = sum(1 for r in rs if "." in r["sn"])
    exact = sum(1 for r in rs if r["qn"] == r["sn"])
    exact_ci = sum(1 for r in rs if r["qn"].lower() == r["sn"].lower())
    both = [(num(r["qa"]), num(r["sa"])) for r in rs if num(r["qa"]) is not None and num(r["sa"]) is not None]
    up = sum(1 for a, b in both if a - b in OFF); dn = sum(1 for a, b in both if b - a in OFF); eq = sum(1 for a, b in both if a == b)
    qmiss = sum(1 for r in rs if not (r["qa"] or "").strip())
    ce = [r["ce"] for r in rs if r["ce"] is not None]
    print(f"  n={n} lowercase rec names={len(lc)} ({100*len(lc)/n:.2f}%) lowercase non-exact={len(lc_ne)} | rec names with dot={dotrec} ({100*dotrec/n:.2f}%) vs S1 with dot={dots1} | exact name={exact} ci-exact={exact_ci}")
    print(f"  rec addr missing={qmiss} | both numbered={len(both)} equal={eq} rec=S1+offset={up} rec=S1-offset={dn} | frCE scored={len(ce)} >=0={sum(1 for x in ce if x>=0)} min={min(ce):.2f}")
print("p>=0.8 (shipped):"); fp([r for r in R if r["in08"]])
print("0.7<=p<0.8 (extra in alt):"); fp([r for r in R if not r["in08"]])
print("p>=0.7 (alt):"); fp(R)
random.seed(20260927_2121)
samp = random.sample(R, 30)
print("\nFRESH RANDOM 30 from p>=0.7 alt set (seed 202609272121):")
for i, r in enumerate(sorted(samp, key=lambda r: -r["p"])):
    print(f"{i+1:2d} {'*' if r['in08'] else ' '} p={r['p']:.2f} ce={r['ce'] if r['ce'] is None else round(r['ce'],2)} {r['v']}/{r['nc']}/{r['ac']}\n    S1 {r['sid']}: {r['sn']} | {r['sa']}\n    RC {r['qid']}: {r['qn']} | {r['qa']}")
rest = [r for r in R if r["in08"] and r not in samp]
print("\nREMAINING p>=0.8 pairs not in the random 30:")
for r in rest:
    print(f"  p={r['p']:.2f} ce={r['ce'] if r['ce'] is None else round(r['ce'],2)} {r['v']}/{r['nc']}/{r['ac']}\n    S1 {r['sid']}: {r['sn']} | {r['sa']}\n    RC {r['qid']}: {r['qn']} | {r['qa']}")
