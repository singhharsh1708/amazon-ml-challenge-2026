"""Step 5 (France push). Experiment/validation script s3_cls.py for this step.

Step context: Step 5 (France push). France rescue additions for records left without a match.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/push/frrescue
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.path.insert(0, f"{REPO_DIR}/src")
import duckdb
import numpy as np
import pyarrow as pa
from france_rules import review_name, review_address, review_key, review_verdict, PARTIAL_NOISE
from decoy_veto import build_vocab, load_decoy_words, type_swap_flags, type_vocab, veto_flags
W = f"{WORK_DIR}/push/frrescue"
c = duckdb.connect(f"{W}/work.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp'")
NUM = "nullif(regexp_extract({a}, '\\b([0-9]+)\\b', 1), '')"
c.execute(f"""create or replace table pq as
  select p.rid, p.s1, p.ce, p.ce1, p.ce2, p.family,
    q.name_full qn, s.name_full sn, q.name_core qc, s.name_core sc, q.address qa, s.address sa,
    coalesce(q.address_missing, false) or coalesce(trim(q.address), '') = '' as q_noaddr,
    {NUM.format(a='q.address')} qnum, {NUM.format(a='s.address')} snum,
    {review_key('q.address')} qk, {review_key('s.address')} sk
  from pool2 p join nm q on q.eid = p.rid join nm s on s.eid = p.s1""")
print("joined", c.execute("select count(*) from pq").fetchone())
t = c.execute("select * from pq").fetchnumpy()
n = len(t["rid"])
rows = [review_name(a, b) + review_address(x, y) for a, b, x, y in zip(t["qn"], t["sn"], t["qk"], t["sk"])]
verd = [review_verdict(*r) for r in rows]
s1n = c.execute("select country, name_full, name_core, address from nm where eid // 10000000000 = 1").fetchnumpy()
tv = type_vocab(s1n["country"], s1n["name_core"], s1n["address"])
print("type vocab france size", len(tv.get("france", ())), sorted(tv.get("france", ()))[:80])
countries = np.array(["france"] * n)
th = type_swap_flags(countries, t["qc"], t["sc"], tv, {"france"})
dw = load_decoy_words()
print("decoy words countries", {k: len(v) for k, v in dw.items()})
dh, sh = veto_flags(countries, t["qn"], t["sn"], np.full(n, 0.5), build_vocab(s1n["country"], s1n["name_full"]), dw, set())
out = pa.table({"rid": t["rid"], "s1": t["s1"],
    "nc": [r[0] for r in rows], "nadd": [r[1] for r in rows], "nrm": [r[2] for r in rows],
    "ac": [r[3] for r in rows], "delta": pa.array([r[4] for r in rows], pa.int64()), "v": verd,
    "type_hit": th, "decoy_hit": dh, "swap_hit": sh})
c.register("o", out)
c.execute("create or replace table cls as select pq.*, o.* exclude (rid, s1) from pq join o using (rid, s1)")
print(c.execute("""select count(*), sum(q_noaddr::int), sum((qnum = snum)::int), sum((qnum is null and snum is null and not q_noaddr)::int),
  sum(type_hit::int), sum(decoy_hit::int), sum(swap_hit::int) from cls""").fetchone())
print("verdicts", c.execute("select v, count(*) from cls group by 1 order by 2 desc").fetchall())
