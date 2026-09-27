"""Step 7 (France empty-S1 fill). Experiment/validation script b6.py for this step.

Step context: Step 7 (France empty-S1 fill). Candidate pool for France source-1 records that ended with no match.
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
import duckdb, pyarrow as pa
S = f"{WORK_DIR}"
W = f"{S}/fill/frfill"
sys.path.insert(0, f"{FINAL_STEPS}/france_push/frce")
from vd import review_name, review_address, review_verdict
ID = "cast(substr(x, 2, 1) as bigint) * 10000000000 + cast(substr(x, 4) as bigint)"
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET threads=3; SET memory_limit='2GB'; SET temp_directory='{W}/tmp/spill'")
c.execute(f"create or replace table vp as select {ID.replace('x','source1_entity_id')} s1, {ID} rid from (select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) x from v20 where matched_entity_ids <> '')")
c.execute("create or replace table vce as select v.rid, v.s1, ce.ce, sc.p from vp v join ce using (rid, s1) left join sc using (rid, s1)")
print("v20 pairs with frCE", c.execute("select count(*), sum((ce>=0)::int) from vce").fetchone())
t = c.execute("select b.rid, b.s1, q.name_full qn, s.name_full sn, q.ak qa, s.ak sa from vce b join nm q on q.eid = b.rid join nm s on s.eid = b.s1").to_arrow_table()
cols = [t.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")]
rows = [review_name(a, b) + review_address(x, y) for a, b, x, y in zip(*cols)]
o = pa.table({"rid": t.column("rid"), "s1": t.column("s1"), "nc": [r[0] for r in rows], "ac": [r[3] for r in rows], "v": [review_verdict(*r) for r in rows]})
c.register("o", o)
print("v20 French pairs with frCE, verdict same/same_noise, by nc x ac: n, frCE>=0 share, p quartiles")
for r in c.execute("""select o.nc, o.ac, count(*) n, round(100*avg((ce>=0)::int),1), round(median(p),3) from vce join o using (rid, s1)
    where o.v in ('same','same_noise') group by all having count(*) >= 20 order by n desc limit 14""").fetchall(): print("  ", r)
print("same/eq/missing by p band:")
for r in c.execute("""select case when p>=0.95 then '0.95+' when p>=0.85 then '0.85-0.95' else '<0.85' end b, count(*), round(100*avg((ce>=0)::int),1) from vce join o using (rid, s1)
    where o.v='same' and o.nc='eq' and o.ac='missing' group by 1 order by 1""").fetchall(): print("  ", r)
c.execute("create or replace table base as select o.nc, o.ac, count(*) bn, avg((ce>=0)::int) bs from vce join o using (rid, s1) where o.v in ('same','same_noise') group by all")
for th in (0.5, 0.6, 0.7, 0.8):
    r = c.execute(f"""select count(g.ce), sum((g.ce>=0)::int), sum(coalesce(b.bs, 0.84)), sum((b.bs is null)::int) from g4 g left join base b using (nc, ac) where g.p >= {th} and g.ce is not null""").fetchone()
    print(th, "covered", r[0], "observed ce>=0", r[1], round(100*r[1]/r[0],1), "expected from v20 same-type pairs", round(r[2],1), round(100*r[2]/r[0],1), "cells w/o baseline", r[3])
