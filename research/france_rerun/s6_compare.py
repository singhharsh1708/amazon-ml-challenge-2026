import sys

from frc import F, R, SP, ID, con, raw

variant = sys.argv[1] if len(sys.argv) > 1 else "new"
assign_file = sys.argv[2] if len(sys.argv) > 2 else f"{F}/work/{variant}/assign.parquet"
out_txt = sys.argv[3] if len(sys.argv) > 3 else f"{F}/out/france_compare.txt"
write_outputs = len(sys.argv) <= 2
w = f"{F}/work/{variant}"
c = con("cmp")
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


c.execute(f"CREATE TABLE fr AS SELECT {ID} AS eid FROM read_parquet('{F}/data/fr_old/test_s*.parquet')")
c.execute(f"""CREATE TABLE v15 AS SELECT {ID.replace('entity_id', 'm')} AS rid, {ID.replace('entity_id', 'source1_entity_id')} AS s1 FROM (
    SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS m
    FROM read_csv('{R}/output/v15/matching_results_v15.tsv', delim='\\t', header=true, all_varchar=true, quote='', escape='')
    WHERE coalesce(matched_entity_ids, '') <> '') WHERE rid IN (SELECT eid FROM fr)""")
c.execute(f"CREATE TABLE nw AS SELECT rid, s1 FROM '{assign_file}'")
has_how = "how" in [r[0] for r in c.execute(f"DESCRIBE SELECT * FROM '{assign_file}'").fetchall()]
c.execute(f"CREATE TABLE how AS SELECT rid, s1, {'how' if has_how else chr(39) + 'n/a' + chr(39)} AS how FROM '{assign_file}'")
c.execute(f"CREATE TABLE cn AS SELECT rid, s1, rk FROM '{w}/cand.parquet'" if variant == "new" else f"CREATE TABLE cn AS SELECT rid, s1, rk FROM '{w}/cand_orig.parquet'")
c.execute(f"CREATE TABLE sc AS SELECT rid, s1, p, p_guard, p_s2, ps, p1 FROM '{w}/scores.parquet'")
c.execute(f"CREATE TABLE vsc AS SELECT rid, s1, p, p_guard FROM '{SP}/v11/pred_v15.parquet' WHERE rid IN (SELECT eid FROM fr)")
c.execute(f"""CREATE TABLE rawt AS SELECT {ID} AS eid, business_name AS name, business_address AS addr FROM (
    SELECT * FROM {raw(1)} UNION ALL SELECT * FROM {raw(2)} UNION ALL SELECT * FROM {raw(3)}) WHERE lower(country) = 'france'""")
if write_outputs:
    c.execute(f"COPY (SELECT rid, s1 FROM nw ORDER BY rid) TO '{F}/out/france_assign.parquet' (FORMAT parquet)")
    c.execute(f"COPY (SELECT rid, s1 FROM cn UNION SELECT rid, s1 FROM nw) TO '{F}/out/france_candidates.parquet' (FORMAT parquet)")
    say(f"wrote {F}/out/france_assign.parquet ({c.execute(f'SELECT count(*) FROM read_parquet(' + chr(39) + F + '/out/france_assign.parquet' + chr(39) + ')').fetchone()[0]:,} pairs) and "
        f"{F}/out/france_candidates.parquet ({c.execute(f'SELECT count(*) FROM read_parquet(' + chr(39) + F + '/out/france_candidates.parquet' + chr(39) + ')').fetchone()[0]:,} pairs)")
nv, nn = c.execute("SELECT (SELECT count(*) FROM v15), (SELECT count(*) FROM nw)").fetchone()
both = c.execute("SELECT count(*) FROM v15 JOIN nw USING (rid, s1)").fetchone()[0]
say(f"variant {variant}: assignment file {assign_file}")
say(f"French assigned pairs: v15 {nv:,}, new {nn:,}, in both {both:,}, only in new {nn - both:,}, only in v15 {nv - both:,}")
say(f"duplicate rids in new: {c.execute('SELECT count(*) - count(DISTINCT rid) FROM nw').fetchone()[0]}")
nfs1 = c.execute("SELECT count(*) FROM fr WHERE eid // 10000000000 = 1").fetchone()[0]
say(f"French S1 entities: {nfs1:,}")
say("assigned pairs per French S1 (number of matched records -> number of S1), v15 vs new:")
dv = dict(c.execute("""SELECT least(coalesce(n, 0), 6) k, count(*) FROM (SELECT eid AS s1 FROM fr WHERE eid // 10000000000 = 1) s
    LEFT JOIN (SELECT s1, count(*) n FROM v15 GROUP BY s1) USING (s1) GROUP BY 1""").fetchall())
dn = dict(c.execute("""SELECT least(coalesce(n, 0), 6) k, count(*) FROM (SELECT eid AS s1 FROM fr WHERE eid // 10000000000 = 1) s
    LEFT JOIN (SELECT s1, count(*) n FROM nw GROUP BY s1) USING (s1) GROUP BY 1""").fetchall())
for k in range(7):
    say(f"  {k if k < 6 else '6+'}: v15 {dv.get(k, 0):,}  new {dn.get(k, 0):,}")
say(f"  mean pairs per S1: v15 {nv / nfs1:.4f}  new {nn / nfs1:.4f}")
say(f"S1 whose matched set differs: {c.execute('SELECT count(DISTINCT s1) FROM (SELECT * FROM (SELECT rid, s1 FROM v15 EXCEPT SELECT rid, s1 FROM nw) UNION ALL SELECT * FROM (SELECT rid, s1 FROM nw EXCEPT SELECT rid, s1 FROM v15))').fetchone()[0]:,}")
say()
c.execute("CREATE TABLE onew AS SELECT n.rid, n.s1 FROM nw n WHERE (rid, s1) NOT IN (SELECT rid, s1 FROM v15)")
c.execute("CREATE TABLE oold AS SELECT v.rid, v.s1 FROM v15 v WHERE (rid, s1) NOT IN (SELECT rid, s1 FROM nw)")
say("only-in-new pairs by source of decision:")
for r in c.execute("SELECT h.how, count(*) FROM onew o JOIN how h USING (rid, s1) GROUP BY 1 ORDER BY 2 DESC").fetchall():
    say(f"  {r[0]}: {r[1]:,}")
say("only-in-new pairs: record status in v15:")
for r in c.execute("""SELECT CASE WHEN v.rid IS NULL THEN 'record unassigned in v15' ELSE 'record assigned to another S1 in v15' END, count(*)
        FROM onew o LEFT JOIN (SELECT DISTINCT rid FROM v15) v USING (rid) GROUP BY 1 ORDER BY 2 DESC""").fetchall():
    say(f"  {r[0]}: {r[1]:,}")
say("only-in-v15 pairs: record status in new:")
for r in c.execute("""SELECT CASE WHEN n.rid IS NULL THEN 'record unassigned in new' ELSE 'record assigned to another S1 in new' END,
        CASE WHEN (o.rid, o.s1) IN (SELECT rid, s1 FROM cn) THEN 'pair still a candidate' ELSE 'pair not a candidate' END, count(*)
        FROM oold o LEFT JOIN (SELECT DISTINCT rid FROM nw) n USING (rid) GROUP BY 1, 2 ORDER BY 3 DESC""").fetchall():
    say(f"  {r[0]}, {r[1]}: {r[2]:,}")
say()


def examples(table, title):
    say(f"15 random {title} pairs (rid -> s1; new score p / guard / stage-2 / stacked; v15 score p / guard):")
    rows = c.execute(f"""SELECT t.rid, t.s1, h.how, s.p, s.p_guard, s.p_s2, s.ps, v.p, v.p_guard,
            q.name, q.addr, x.name, x.addr
        FROM (SELECT * FROM {table} USING SAMPLE 15 ROWS (reservoir, 7)) t
        LEFT JOIN how h USING (rid, s1) LEFT JOIN sc s USING (rid, s1) LEFT JOIN vsc v USING (rid, s1)
        LEFT JOIN rawt q ON q.eid = t.rid LEFT JOIN rawt x ON x.eid = t.s1 ORDER BY t.rid""").fetchall()
    f = lambda v: "-" if v is None else f"{v:.3f}"
    for r in rows:
        sid = lambda e: f"S{e // 10000000000}-{e % 10000000000}"
        say(f"  {sid(r[0])} -> {sid(r[1])}  [{r[2] or '-'}]  new p {f(r[3])} g {f(r[4])} s2 {f(r[5])} st {f(r[6])} | v15 p {f(r[7])} g {f(r[8])}")
        say(f"      rec: {r[9]} | {r[10]}")
        say(f"      S1 : {r[11]} | {r[12]}")
    say()


examples("onew", "only-in-new")
examples("oold", "only-in-v15")
open(out_txt, "w").write("\n".join(lines) + "\n")
c.close()
