import sys, duckdb
base_m, base_c, out_m, out_c = sys.argv[1:5]
removals = [x for x in sys.argv[5].split(',') if x]
adds = [x for x in sys.argv[6].split(',') if x]
ID = "cast(substr({c}, 2, 1) as bigint) * 10000000000 + cast(substr({c}, 4) as bigint)"
STR = "'S' || ({c} // 10000000000)::varchar || '-' || ({c} % 10000000000)::varchar"
import os, shutil
tmp = os.path.join(os.path.dirname(out_m), 'apply_tmp'); os.makedirs(tmp, exist_ok=True)
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=2; SET preserve_insertion_order=false; SET temp_directory='{tmp}'")
rd = lambda f: f"read_csv('{f}', delim='\t', header=true, all_varchar=true, quote='', escape='')"
c.execute(f"CREATE TABLE s1 AS SELECT source1_entity_id AS sid, {ID.format(c='source1_entity_id')} AS s1 FROM {rd(base_m)}")
c.execute(f"""CREATE TABLE m AS SELECT {ID.format(c='source1_entity_id')} AS s1, {ID.format(c='x')} AS rid FROM (
    SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS x FROM {rd(base_m)} WHERE coalesce(matched_entity_ids, '') <> '')""")
c.execute(f"""CREATE TABLE cd AS SELECT {ID.format(c='source1_entity_id')} AS s1, {ID.format(c='x')} AS rid FROM (
    SELECT source1_entity_id, trim(unnest(string_split(candidate_entity_ids, ','))) AS x FROM {rd(base_c)} WHERE coalesce(candidate_entity_ids, '') <> '')""")
n0 = c.execute("SELECT count(*) FROM m").fetchone()[0]
c.execute("CREATE TABLE rem AS SELECT DISTINCT rid, s1 FROM (" + (" UNION ALL ".join(f"SELECT rid, s1 FROM '{f}'" for f in removals) or "SELECT NULL::bigint rid, NULL::bigint s1 WHERE false") + ")")
c.execute("DELETE FROM m WHERE (rid, s1) IN (SELECT rid, s1 FROM rem)")
n1 = c.execute("SELECT count(*) FROM m").fetchone()[0]
c.execute("CREATE TABLE ad AS SELECT rid, arg_min(s1, s1) AS s1 FROM (" + (" UNION ALL ".join(f"SELECT rid, s1 FROM '{f}'" for f in adds) or "SELECT NULL::bigint rid, NULL::bigint s1 WHERE false") + ") GROUP BY rid")
c.execute("DELETE FROM ad WHERE rid IN (SELECT rid FROM m) OR s1 NOT IN (SELECT s1 FROM s1)")
c.execute("INSERT INTO m SELECT s1, rid FROM ad")
n2 = c.execute("SELECT count(*) FROM m").fetchone()[0]
c.execute("INSERT INTO cd SELECT s1, rid FROM ad WHERE (rid, s1) NOT IN (SELECT rid, s1 FROM cd)")
assert c.execute("SELECT count(*) FROM (SELECT rid FROM m GROUP BY rid HAVING count(*) > 1)").fetchone()[0] == 0
PARTS = 16
for table, out, col in (('m', out_m, 'matched_entity_ids'), ('cd', out_c, 'candidate_entity_ids')):
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(f"source1_entity_id\t{col}\n")
    lines = []
    for k in range(PARTS):
        rows = c.execute(f"""SELECT s.sid, coalesce(string_agg({STR.format(c='t.rid')}, ',' ORDER BY t.rid), '')
            FROM (SELECT * FROM s1 WHERE s1 % {PARTS} = {k}) s LEFT JOIN (SELECT * FROM {table} WHERE s1 % {PARTS} = {k}) t ON t.s1 = s.s1
            GROUP BY s.sid""").fetchall()
        lines.extend(f"{a}\t{b}\n" for a, b in rows)
    lines.sort(key=lambda x: x.split('\t', 1)[0])
    with open(out, 'a', encoding='utf-8') as fh:
        fh.writelines(lines)
    del lines
shutil.rmtree(tmp, ignore_errors=True)
print(f"base {n0:,} -> after removals {n1:,} (-{n0-n1:,}) -> after adds {n2:,} (+{n2-n1:,})")
