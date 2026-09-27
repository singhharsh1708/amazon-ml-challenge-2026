"""Step 7 gate. Experiment/validation script g4.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, re
sys.dont_write_bytecode = True
import duckdb
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
c = duckdb.connect(f"{G}/tmp/g.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
c.execute("create or replace table s1a as select eid, bn, ba, is_emp, lower(strip_accents(coalesce(ba,''))) la from frs1")
STOP = {"rue","avenue","boulevard","place","route","allee","hauts","france","nouvelle","aquitaine","pays","loire","lille","nantes","de","du","des","la","le","les","bis","square","cour","tourcoing","roubaix","dunkerque","bordeaux","calais","merignac","nord","gironde"}
ship = c.execute("select a.s1, a.rid, s.bn, s.ba, q.bn, q.ba from a_p08 a join frs1 s on s.eid=a.s1 join frrec q on q.eid=a.rid order by a.s1").fetchall()
for s1, rid, sn, sa, qn, qa in ship:
    la = (sa or "").lower()
    import unicodedata
    la = unicodedata.normalize("NFKD", la).encode("ascii","ignore").decode()
    m = re.search(r"\b(\d+)\b", la)
    words = [w for w in re.findall(r"[a-z]{4,}", la) if w not in STOP]
    if not m or not words:
        city = [w for w in re.findall(r"[a-z]{4,}", la)]
        hits = c.execute(f"select eid, bn, ba, is_emp from s1a where eid <> {s1} and lower(strip_accents(bn)) = lower(strip_accents(?))", [sn]).fetchall()
        print(f"## {sn} | {sa}  <- {qn} | {qa}\n   no number/street: other same-name S1 = {len(hits)}")
        for h in hits: print("     ", h)
        continue
    w = max(words, key=len)
    hits = c.execute(f"select eid, bn, ba, is_emp from s1a where eid <> {s1} and regexp_matches(la, '\\b{m.group(1)}\\b') and la like '%{w}%'").fetchall()
    print(f"## {sn} | {sa}  <- {qn} | {qa}\n   other S1 at number {m.group(1)} + '{w}': {len(hits)}")
    for h in hits: print("     ", h)
