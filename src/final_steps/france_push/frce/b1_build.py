"""Builds synthetic French training pairs by perturbing names and addresses (typos, legal-form drops, number shifts, type swaps).

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    {F}/chk/labels.csv
    {F}/chk/cls.parquet
    {W}/labels60.parquet
    {F}/work/new/pred2.parquet
    {SP}/ce/train_pairs.parquet
    {W}/frce_all.parquet
    {W}/frce_train.parquet
    {W}/frce_hold.parquet
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time, random
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, duckdb
SP = f"{WORK_DIR}"
F = f"{SP}/fr_rerun"
W = f"{SP}/push/frce"
sys.path.insert(0, W)
from vd import review_name, review_address, review_key, review_verdict, KEEP_NOISE
t0 = time.time()
rng = np.random.default_rng(11); random.seed(11)
c = duckdb.connect(f"{W}/tmp/w.db")
c.execute(f"SET memory_limit='2GB'; SET threads=3; SET preserve_insertion_order=false; SET temp_directory='{W}/tmp/spill'")
lab = c.execute(f"""select l.eid, l.lab, x.rid, x.s1 from read_csv('{F}/chk/labels.csv') l
  join '{F}/chk/cls.parquet' x on x.grp = l.grp and x.rid = cast(substr(l.eid, 2, 1) as bigint) * 10000000000 + cast(substr(l.eid, 4) as bigint)""").fetchall()
print("labelled pairs matched", len(lab), flush=True)
c.execute("create or replace table labp as select * from (values " + ",".join(f"({r[2]},{r[3]},'{r[1]}')" for r in lab) + ") t(rid, s1, lab)")
c.execute(f"copy (select l.rid, l.s1, l.lab, q.name_full || ' | ' || q.address q_text, s.name_full || ' | ' || s.address s_text from labp l join nm q on q.eid=l.rid join nm s on s.eid=l.s1) to '{W}/labels60.parquet'")
types = [r[0] for r in c.execute("""select w from swp where n >= 150 and lc >= 0.02 and p >= 0.05 and regexp_full_match(w, '[a-z]{3,}')
  and w not in ('fils','cie','groupe','services','developpement','associes','france','frs','freres','st','compagnie','cb') order by n desc""").fetchall()]
print("type vocab", len(types), types, flush=True)
DEC_ADD = ["participations", "distribution", "holding", "international"]
DEC_DUAL = ["developpement", "groupe", "france"]
COPY_SUF = ["fils", "services", "associes", "cie", "frs", "developpement", "groupe", "france", "freres"]
LEGAL = ["sa", "sas", "sasu", "sarl", "eurl", "sci", "snc"]
OFFS = [3, 4, 5, 7, 9, 11, 13, 21]
NPOS = NNEG = 23000
c.execute(f"""create or replace table pos0 as select p.rid, p.s1, q.name_full qn, s.name_full sn, q.address qa0, s.address sa0,
  {review_key('q.address')} qa, {review_key('s.address')} sa from read_parquet('{F}/work/new/pred2.parquet') p
  join nm q on q.eid=p.rid join nm s on s.eid=p.s1
  where p.p >= 0.98 and p.p_guard >= 0.98 and p.rid not in (select rid from labp) and p.s1 not in (select s1 from labp)
  order by hash(p.rid, 77) limit 90000""")
t = c.execute("select rid, s1, qn, sn, qa, sa, qn || ' | ' || qa0 q_text, sn || ' | ' || sa0 s_text from pos0").to_arrow_table()
cols = [t.column(n).to_pylist() for n in ("qn", "sn", "qa", "sa")]
v = [review_verdict(*(review_name(a, b) + review_address(x, y))) for a, b, x, y in zip(*cols)]
v = np.array(v)
qt, st = np.array(t.column("q_text").to_pylist(), dtype=object), np.array(t.column("s_text").to_pylist(), dtype=object)
ok = np.isin(v, ["same", "same_noise"])
print("positives classified", len(v), dict(zip(*np.unique(v, return_counts=True))), flush=True)
ident = qt == st
i_diff = np.where(ok & ~ident)[0]; i_same = np.where(ok & ident)[0]
nd = min(len(i_diff), int(NPOS * 0.7)); pick = np.concatenate([rng.choice(i_diff, nd, replace=False), rng.choice(i_same, NPOS - nd, replace=False)])
pos = pa.table({"rid": t.column("rid").take(pick), "s1": t.column("s1").take(pick), "q_text": qt[pick].tolist(), "s_text": st[pick].tolist(),
    "label": np.ones(len(pick), np.int32), "kind": ["ps_pos"] * len(pick)})
print("pseudo pos", len(pick), "non-identical", nd, flush=True)
c.execute(f"""create or replace table neg0 as select p.rid, p.s1, p.p1, q.name_full || ' | ' || q.address q_text, s.name_full || ' | ' || s.address s_text
  from read_parquet('{F}/work/new/pred2.parquet') p join nm q on q.eid=p.rid join nm s on s.eid=p.s1
  where p.p <= 0.02 and p.rid not in (select rid from labp)""")
print("neg pool", c.execute("select count(*), count(*) filter (where p1 >= 0.3), count(*) filter (where p1 >= 0.05) from neg0").fetchall(), flush=True)
neg = c.execute(f"""select rid, s1, q_text, s_text, 0::int as label, 'ps_neg' as kind from (
   (select * from neg0 where p1 >= 0.05 order by hash(rid, s1, 5) limit {NNEG // 2})
   union all (select * from neg0 where p1 < 0.05 and p1 >= 0.005 order by hash(rid, s1, 6) limit {NNEG // 4})
   union all (select * from neg0 where p1 < 0.005 order by hash(rid, s1, 7) limit {NNEG - NNEG // 2 - NNEG // 4}))""").to_arrow_table()
print("pseudo neg", neg.num_rows, flush=True)
s1 = c.execute("select eid, name_full, address from nm where eid < 20000000000 and eid not in (select s1 from labp) order by hash(eid, 3) limit 160000").fetchall()
TYPES = set(types)
def typo(w):
    """Return the word with one random character edit."""
    if len(w) < 4: return w
    i = random.randrange(1, len(w) - 1); k = random.random()
    a = "abcdefghijklmnopqrstuvwxyz"
    if k < 0.3: return w[:i] + w[i + 1:]
    if k < 0.6: return w[:i] + random.choice(a) + w[i + 1:]
    if k < 0.8: return w[:i] + random.choice(a) + w[i:]
    return w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
def typo_str(s):
    """Return the string with a typo in one of its words."""
    t = s.split()
    idx = [i for i, w in enumerate(t) if len(w) >= 4 and not w.isdigit()]
    if not idx: return None
    i = random.choice(idx); t[i] = typo(t[i]); return " ".join(t)
def shift(a):
    """Shift the street number of an address, or return None if it has none."""
    t = a.split()
    idx = [i for i, w in enumerate(t) if w.isdigit()]
    if not idx: return None
    i = idx[0]; t[i] = str(int(t[i]) + random.choice(OFFS)); return " ".join(t)
def insert_word(n, w):
    """Insert a word before the legal form of a name (or append it)."""
    t = n.split()
    if t and t[-1] in LEGAL: t.insert(len(t) - 1, w)
    else: t.append(w)
    return " ".join(t)
def d_typeswap(n, a):
    """Make a negative pair by swapping the business type word."""
    t = n.split(); idx = [i for i, w in enumerate(t) if w in TYPES]
    if not idx: return None
    i = random.choice(idx); t[i] = random.choice([x for x in types if x != t[i]])
    a2 = shift(a) if random.random() < 0.5 else a
    return " ".join(t), a2 or a
def d_shift(n, a):
    """Make a negative pair by shifting the street number."""
    a2 = shift(a); return None if a2 is None else (n, a2)
def d_legal(n, a):
    """Make a negative pair by changing the legal form."""
    t = n.split(); idx = [i for i, w in enumerate(t) if w in LEGAL]
    if not idx: return None
    i = idx[0]; t[i] = random.choice([x for x in LEGAL if x != t[i]]); return " ".join(t), a
def d_addword(n, a):
    """Make a negative pair by adding a distinguishing word to the name."""
    if random.random() < 0.5:
        w = random.choice(DEC_ADD); a2 = shift(a) if random.random() < 0.5 else a
    else:
        w = random.choice(DEC_DUAL); a2 = shift(a)
    if a2 is None or w in n.split(): return None
    return insert_word(n, w), a2
def c_typo(n, a):
    """Make a positive pair with a typo in the name."""
    if random.random() < 0.6:
        n2 = typo_str(n); return None if n2 is None else (n2, a)
    a2 = typo_str(a); return None if a2 is None else (n, a2)
def c_drop(n, a):
    """Make a positive pair with one name word dropped."""
    t = n.split(); idx = [i for i, w in enumerate(t) if w not in LEGAL]
    if len(idx) < 2: return None
    del t[random.choice(idx)]; return " ".join(t), a
def c_suffix(n, a):
    """Make a positive pair with a name suffix added or removed."""
    w = random.choice(COPY_SUF)
    if w in n.split(): return None
    if random.random() < 0.3 and w in ("fils", "cie", "associes"): w = "and " + w
    return insert_word(n, w), a
def c_legaldrop(n, a):
    """Make a positive pair with the legal form dropped."""
    t = n.split(); idx = [i for i, w in enumerate(t) if w in LEGAL]
    if not idx or len(t) < 2: return None
    del t[idx[0]]; return " ".join(t), a
def c_ndeg(n, a):
    """Make a positive pair with the street number degree marker changed."""
    t = a.split()
    if "bis" in t or "ter" in t:
        return n, " ".join(w for w in t if w not in ("bis", "ter"))
    idx = [i for i, w in enumerate(t) if w.isdigit()]
    if not idx: return None
    t.insert(idx[0], "ndeg"); return n, " ".join(t)
def c_citymove(n, a):
    """Make a positive pair where the city moves within the address."""
    t = a.split()
    if len(t) < 3: return None
    return n, " ".join([t[-1]] + t[:-1])
def c_noaddr(n, a):
    """Make a positive pair with the address removed."""
    return n, ""
TSUF = ["fils", "developpement", "cie", "groupe", "france", "associes", "services", "et fils", "and fils", "and associes", "and cie"]
def c_typesuffix(n, a):
    """Make a positive pair with the business type word moved or dropped."""
    t = n.split(); idx = [i for i, w in enumerate(t) if w in TYPES]
    if not idx: return None
    i = random.choice(idx); w = random.choice(TSUF)
    if random.random() < 0.5:
        t[i] = w
    else:
        del t[i]; t.append(w)
    return " ".join(t), a
def c_glue(n, a):
    """Make a positive pair with two name words glued together."""
    t = n.split()
    if len(t) < 2: return None
    g = "".join(t)
    if random.random() < 0.4: g = typo(g)
    return g, a
def c_acronym(n, a):
    """Make a positive pair where the name is replaced by its acronym."""
    t = [w for w in n.split() if w not in LEGAL and w not in ("de", "du", "des", "la", "le", "les", "et", "and")]
    if len(t) < 2: return None
    k = random.random()
    ac = "".join(w[0] for w in t) if k < 0.5 else (t[0][0] + t[-1][0] if k < 0.8 else t[0][:2])
    return ac, a
def c_numdrop(n, a):
    """Make a positive pair with the street number dropped."""
    t = a.split(); idx = [i for i, w in enumerate(t) if w.isdigit()]
    if not idx: return None
    del t[idx[0]]; return n, " ".join(t)
def copy_noise(n, a):
    """Apply random copy noise (typos, drops) to a name and address."""
    k = random.random()
    if k < 0.35:
        x = typo_str(n) if random.random() < 0.6 else None
        if x: return x, a
        y = typo_str(a) if a else None
        return n, (y or a)
    if k < 0.6:
        r = c_citymove(n, a); return r if r else (n, a)
    if k < 0.8:
        r = c_numdrop(n, a); return r if r else (n, a)
    r = c_legaldrop(n, a); return r if r else (n, a)
DEC = [(d_typeswap, 6500), (d_shift, 6500), (d_legal, 1500), (d_addword, 8500)]
COP = [(c_typo, 3500), (c_drop, 2500), (c_suffix, 3000), (c_typesuffix, 5500), (c_legaldrop, 2500), (c_glue, 1500), (c_acronym, 1000), (c_numdrop, 1500), (c_ndeg, 500), (c_citymove, 1000), (c_noaddr, 2000)]
rows = {"rid": [], "s1": [], "q_text": [], "s_text": [], "label": [], "kind": []}
ptr = 0
def gen(fn, k, label, noise):
    """Generate k synthetic pairs with the given perturbation function and label."""
    global ptr
    got = 0; tries = 0
    while got < k and tries < k * 30:
        tries += 1
        eid, n, a = s1[ptr % len(s1)]; ptr += 1
        a = a or ""
        r = fn(n, a)
        if r is None: continue
        n2, a2 = r
        if noise and label == 0 and random.random() < 0.35:
            kk = random.random()
            if fn is d_legal or kk < 0.45:
                x = typo_str(n2); n2 = x or n2
            elif kk < 0.75:
                x = c_legaldrop(n2, a2); n2 = x[0] if x else n2
            else:
                x = c_citymove(n2, a2); a2 = x[1] if x else a2
        if noise and label == 1 and random.random() < 0.3:
            n2, a2 = copy_noise(n2, a2)
        if n2 == n and a2 == a: continue
        rows["rid"].append(-1); rows["s1"].append(eid); rows["q_text"].append(n2 + " | " + a2); rows["s_text"].append(n + " | " + a)
        rows["label"].append(label); rows["kind"].append(("sd_" if label == 0 else "sc_") + fn.__name__[2:]); got += 1
    print(fn.__name__, got, "tries", tries, flush=True)
for fn, k in DEC: gen(fn, k, 0, True)
for fn, k in COP: gen(fn, k, 1, True)
syn = pa.table({"rid": pa.array(rows["rid"], pa.int64()), "s1": pa.array(rows["s1"], pa.int64()), "q_text": rows["q_text"], "s_text": rows["s_text"],
    "label": pa.array(rows["label"], pa.int32()), "kind": rows["kind"]})
fr = pa.concat_tables([pos, neg.cast(pos.schema), syn])
us = c.execute(f"""select rid, s1, q_text, s_text, label::int as label, 'us_' || country as kind from read_parquet('{SP}/ce/train_pairs.parquet')
  offset 1300000""").to_arrow_table().cast(pos.schema)
print("fr", fr.num_rows, "us/india", us.num_rows, flush=True)
allt = pa.concat_tables([fr, us])
c.register("allt", allt)
c.execute(f"""copy (select *, (hash(q_text, s_text, 91) % 5 = 0) as hold from allt order by hash(q_text, s_text, rid, 13)) to '{W}/frce_all.parquet'""")
print(c.execute(f"select hold, count(*), avg(label) from '{W}/frce_all.parquet' group by 1").fetchall())
c.execute(f"copy (select q_text, s_text, label from '{W}/frce_all.parquet' where not hold order by hash(q_text, s_text, 29)) to '{W}/frce_train.parquet'")
c.execute(f"copy (select rid, s1, q_text, s_text, label, kind from '{W}/frce_all.parquet' where hold) to '{W}/frce_hold.parquet'")
print(c.execute(f"select kind, label, count(*) from '{W}/frce_all.parquet' group by all order by 1").fetchall())
print(c.execute(f"select count(*) from '{W}/frce_train.parquet'").fetchall(), f"{time.time()-t0:.0f}s")
for k in ["sc_typesuffix", "sc_glue", "sc_acronym", "sc_numdrop", "sd_typeswap", "sd_addword"]:
    print(k, c.execute(f"select q_text, s_text from '{W}/frce_all.parquet' where kind='{k}' limit 2").fetchall())
