"""Step 5 (France push). Experiment/validation script e_dis.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {W}/sc_a.parquet
    {W}/sc_b.parquet
    {SP}/ce/test_{m}.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import duckdb
SP = f"{WORK_DIR}"
W = f"{SP}/push/frce"
c = duckdb.connect(); c.execute(f"SET threads=3; SET memory_limit='2GB'")
c.execute(f"ATTACH '{W}/tmp/w.db' AS w (READ_ONLY)")
c.execute(f"create table fr as select rid, s1, any_value(ce) ce from read_parquet(['{W}/sc_a.parquet', '{W}/sc_b.parquet']) group by all")
for m in ("ce1", "ce2"):
    c.execute(f"create table {m} as select rid, s1, any_value(ce) ce from '{SP}/ce/test_{m}.parquet' group by all")
print("band pairs with ce1/ce2 score:", c.execute("select count(*), count(c1.ce), count(c2.ce) from w.bandv b left join ce1 c1 using (rid, s1) left join ce2 c2 using (rid, s1)").fetchall())
c.execute("""create table rec as
  select b.rid, bool_or(b.pair_asg) asg_in_band, bool_or(b.rid_asg) rid_asg,
    any_value(b.s1) filter (where b.pair_asg) s1_asg, arg_max(b.s1, b.p) s1a
  from w.bandv b group by 1""")
for m in ("fr", "ce1", "ce2"):
    r = c.execute(f"""select count(*) filter (where r.asg_in_band and a.ce < -3), count(*) filter (where r.asg_in_band and a.ce is not null),
        count(*) filter (where not r.rid_asg and t.ce > 3), count(*) filter (where not r.rid_asg and t.ce is not null)
      from rec r left join {m} a on a.rid = r.rid and a.s1 = r.s1_asg left join {m} t on t.rid = r.rid and t.s1 = r.s1a""").fetchone()
    tot = (r[0] + r[2]) / max(1, r[1] + r[3])
    print(f"{m}: assigned-in-band with ce<-3 {r[0]}/{r[1]} ({100*r[0]/max(1,r[1]):.1f}%), unassigned top-p ce>3 {r[2]}/{r[3]} ({100*r[2]/max(1,r[3]):.1f}%), combined {100*tot:.1f}%")
