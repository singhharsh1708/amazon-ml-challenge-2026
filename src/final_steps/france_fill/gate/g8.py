"""Step 7 gate. Experiment/validation script g8.py for this step.

Step context: Step 7 gate. Strict filters on the France fill pool; writes additions_strict.parquet (24 pairs).
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys
sys.dont_write_bytecode = True
import duckdb
S = f"{WORK_DIR}"
G = f"{S}/fill/gate"
c = duckdb.connect(f"{G}/tmp/g.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{G}/tmp/spill'")
print("group: n, lowercase %, dot %, all-caps %, exact-case %, double-space %, accent-added %")
q = """select {g} grp, count(*), round(100*avg((r.bn = lower(r.bn) and regexp_matches(r.bn,'[a-z]'))::int),2), round(100*avg((r.bn like '%.%')::int),2),
  round(100*avg((r.bn = upper(r.bn) and regexp_matches(r.bn,'[A-Z]'))::int),2), round(100*avg((r.bn = s.bn)::int),2), round(100*avg((r.bn like '%  %')::int),2),
  round(100*avg((strip_accents(r.bn) <> r.bn and strip_accents(s.bn) = s.bn)::int),2)
  from mrp m join frrec r on r.eid = m.rid join frs1 s on s.eid = m.s1 {w} group by all order by 1"""
for r in c.execute(q.format(g="case when m.is_asg then 'A assigned' when m.emp then 'C unassigned, S1 empty' else 'B unassigned, S1 non-empty' end", w="")).fetchall(): print("  ", r)
print("unassigned by p band (all S1 states):")
for r in c.execute(q.format(g="case when m.p >= 0.8 then 'p>=0.8' when m.p >= 0.5 then '0.5-0.8' when m.p is null then 'unscored' else '<0.5' end", w="where not m.is_asg")).fetchall(): print("  ", r)
print("assigned by p band:")
for r in c.execute(q.format(g="case when m.p >= 0.95 then 'p>=0.95' when m.p >= 0.85 then '0.85-0.95' else '<0.85' end", w="where m.is_asg")).fetchall(): print("  ", r)
