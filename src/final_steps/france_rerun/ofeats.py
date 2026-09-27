"""Odd-one-out pair features used by the France re-run (france_rerun/s3_score.py).

This is the feature code the stage 2 v10 model (data/v10/model/stage2_odd.txt) was trained with.
For each record/candidate pair it compares the pair with the other records that point at the same
Source 1 entity (the siblings): house numbers, number shifts, suffixed numbers, name words added or
removed, the learned decoy words (data/decoy_words.json) and the level-2 decoy words (data/v10/l2_words.json).

Inputs: $REPO_DIR/data/decoy_words.json, $REPO_DIR/data/v10/l2_words.json.
Run as a script (python ofeats.py <split>) it reads $WORK_DIR/wf/hunt-odd-one-out-model/<split>_sib.parquet
and <split>_text.parquet and writes <split>_ofeat.parquet in the same folder.
"""
import re, json, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from rapidfuzz import fuzz
import os
REPO_DIR = os.path.abspath(os.environ.get('REPO_DIR', '.'))
WORK_DIR = os.path.abspath(os.environ.get('WORK_DIR', 'work'))

DECOY = {k: set(v) for k, v in json.load(open(f'{REPO_DIR}/data/decoy_words.json')).items()}
L2 = {k: set(v) for k, v in json.load(open(f'{REPO_DIR}/data/v10/l2_words.json')).items()}
LEGAL = {"pvt","private","ltd","limited","llp","llc","inc","incorporated","corp","corporation","co","company",
         "lp","plc","pllc","pc","sarl","sas","sasu","eurl","sa","sci","snc","gmbh"}
STOP = {"the","and","of","et","de","du","des","la","le","les","ms","m","s","shri","sri","smt","mr","mrs","dr",
        "dba","formerly","known","as","com","www","d","l","a"}
D8 = {3, 4, 5, 7, 9, 11, 13, 21}
NUM = re.compile(r"\d+")
SFX = re.compile(r"[a-z]?\d+[a-z]")
ORD = re.compile(r"\d+(st|nd|rd|th)")
NAMES = ["o_n5", "o_n9", "o_n9_x", "o_name_sib", "o_name_rel", "o_addr_sib", "o_addr_rel", "o_name_eq_sib", "o_addr_eq_sib",
         "o_d1", "o_d1_cls", "o_shift_d8", "o_shift_p12", "o_shift_n12", "o_q1_sib", "o_s1_sib", "o_odd_num",
         "o_qx_n", "o_qx_sib", "o_qx_unexpl", "o_sfx_new", "o_sfx_sib", "o_add_n", "o_add_unsup", "o_rem_n", "o_rem_kept",
         "o_veto", "o_l2"]


def nums(addr):
    """Return the numbers in an address, leading zeros stripped."""
    out = []
    for tok in (addr or "").split():
        for m in NUM.findall(tok):
            out.append(m.lstrip("0") or "0")
    return out


def sfx(addr):
    """Return the address tokens that are a number with a letter suffix (not ordinals)."""
    return {t for t in (addr or "").split() if SFX.fullmatch(t) and not ORD.fullmatch(t)}


def real(t):
    """Return True for a name word that is not a digit, legal form or stop word."""
    return len(t) >= 2 and not t.isdigit() and t not in LEGAL and t not in STOP


def matched(t, other_list, other_joined):
    """Return True if a word appears in, or fuzzily matches, the other name."""
    if len(t) >= 4 and t in other_joined:
        return True
    return any(fuzz.ratio(t, o) >= 80 for o in other_list)


def unexplained(qn, sn, sib_nums):
    """Count record numbers not found in the candidate, its siblings or a split/joined form."""
    sset = set(sn)
    bad = 0
    for i, a in enumerate(qn):
        if a in sset or a in sib_nums:
            continue
        if any(a in b and len(a) < len(b) for b in sn):
            continue
        if i + 1 < len(qn) and any((a + qn[i + 1]) in b for b in sn):
            continue
        if i > 0 and any((qn[i - 1] + a) in b for b in sn):
            continue
        bad += 1
    return bad


def ival(x):
    """Return the integer value of a number string (first 9 digits)."""
    return int(x[:9])


def pair_features(rid, country, q_name, q_addr, s_name, s_addr, sibs):
    """Return the odd-one-out feature dict (keys in NAMES) for one record/candidate pair."""
    nan = float("nan")
    f = dict.fromkeys(NAMES, nan)
    q_name, s_name, q_addr, s_addr = q_name or "", s_name or "", q_addr or "", s_addr or ""
    S = [x for x in sibs if x[0] != rid]
    f["o_n5"] = len(S)
    src = rid // 10000000000
    S9 = [x for x in S if x[1] >= 0.9]
    f["o_n9"] = len(S9)
    f["o_n9_x"] = sum(1 for x in S9 if x[2] != src)
    rn = fuzz.ratio(q_name, s_name)
    ra = fuzz.ratio(q_addr, s_addr)
    if S:
        f["o_name_sib"] = max(fuzz.ratio(q_name, x[3]) for x in S)
        f["o_name_rel"] = f["o_name_sib"] - rn
        f["o_addr_sib"] = max(fuzz.ratio(q_addr, x[4]) for x in S)
        f["o_addr_rel"] = f["o_addr_sib"] - ra
        f["o_name_eq_sib"] = sum(1 for x in S if x[3] == q_name)
        f["o_addr_eq_sib"] = sum(1 for x in S if x[4] == q_addr)
    qn, sn = nums(q_addr), nums(s_addr)
    sib_nums = set()
    for x in S:
        sib_nums |= x[5]
    if qn and sn:
        d = ival(qn[0]) - ival(sn[0])
        f["o_d1"] = max(-100, min(100, d))
        f["o_d1_cls"] = 0 if d == 0 else 1 if d in D8 else 2 if d in (1, 2) else 3 if d in (-1, -2) else 4
    qset, sset = set(qn), set(sn)
    qx = [n for n in dict.fromkeys(qn) if n not in sset]
    sx = [n for n in dict.fromkeys(sn) if n not in qset]
    ds = {ival(a) - ival(b) for a in qx for b in sx}
    f["o_shift_d8"] = int(bool(ds & D8))
    f["o_shift_p12"] = int(bool(ds & {1, 2}))
    f["o_shift_n12"] = int(bool(ds & {-1, -2}))
    if qn:
        f["o_q1_sib"] = sum(1 for x in S if qn[0] in x[5])
    if sn:
        f["o_s1_sib"] = sum(1 for x in S if sn[0] in x[5])
    if qn and sn:
        f["o_odd_num"] = int(qn[0] != sn[0] and f["o_s1_sib"] > 0 and f["o_q1_sib"] == 0)
    f["o_qx_n"] = len(qx)
    f["o_qx_sib"] = sum(1 for n in qx if n in sib_nums)
    f["o_qx_unexpl"] = unexplained(qn, sn, sib_nums) if qx else 0
    new_sfx = sfx(q_addr) - set(s_addr.split())
    f["o_sfx_new"] = int(bool(new_sfx))
    f["o_sfx_sib"] = int(any(any(t in x[4].split() for x in S) for t in new_sfx)) if new_sfx else 0
    q_tok, s_tok = q_name.split(), s_name.split()
    qs, ss = set(q_tok), set(s_tok)
    s_joined, q_joined = "".join(s_tok), "".join(q_tok)
    added = [t for t in dict.fromkeys(q_tok) if t not in ss and real(t) and not matched(t, s_tok, s_joined)]
    removed = [t for t in dict.fromkeys(s_tok) if t not in qs and real(t) and not matched(t, q_tok, q_joined)]
    sib_tok = set()
    for x in S:
        sib_tok |= x[6]
    f["o_add_n"] = len(added)
    f["o_add_unsup"] = sum(1 for t in added if t not in sib_tok)
    f["o_rem_n"] = len(removed)
    if S:
        f["o_rem_kept"] = sum(1 for t in removed if all(t in x[6] for x in S))
    dec = DECOY.get(country, set())
    f["o_veto"] = int(any(t in dec and t not in ss for t in q_tok if len(t) >= 3 and t.isalpha()))
    l2 = L2.get(country, set())
    f["o_l2"] = int(any(t in l2 for t in added if len(t) >= 3))
    return f


def sibling_index(sib_table):
    """Index the sibling table by Source 1 id: (sid, p1, source, name, addr, numbers, name words)."""
    idx = {}
    cols = {n: sib_table.column(n).to_pylist() for n in ["s1", "sid", "sp1", "src", "name", "addr"]}
    for s1, sid, sp1, src, name, addr in zip(cols["s1"], cols["sid"], cols["sp1"], cols["src"], cols["name"], cols["addr"]):
        name, addr = name or "", addr or ""
        idx.setdefault(s1, []).append((sid, sp1, src, name, addr, set(nums(addr)), set(name.split())))
    return idx


if __name__ == "__main__":
    split = sys.argv[1]
    S = f'{WORK_DIR}/wf/hunt-odd-one-out-model'
    t = time.time()
    idx = sibling_index(pq.read_table(f"{S}/{split}_sib.parquet"))
    pf = pq.ParquetFile(f"{S}/{split}_text.parquet")
    writer = None
    n = 0
    for batch in pf.iter_batches(batch_size=200_000):
        b = {k: batch.column(k).to_pylist() for k in ["rid", "s1", "country", "q_name", "q_addr", "s_name", "s_addr"]}
        rows = [pair_features(r, c, qn, qa, sn, sa, idx.get(s, ())) for r, s, c, qn, qa, sn, sa in
                zip(b["rid"], b["s1"], b["country"], b["q_name"], b["q_addr"], b["s_name"], b["s_addr"])]
        out = pa.table({"rid": batch.column("rid"), "s1": batch.column("s1")} |
                       {k: pa.array(np.array([r[k] for r in rows], dtype=np.float32)) for k in NAMES})
        writer = writer or pq.ParquetWriter(f"{S}/{split}_ofeat.parquet", out.schema)
        writer.write_table(out)
        n += len(rows)
        print(f"{split} {n:,} pairs ({time.time()-t:.0f}s)", flush=True)
    writer.close()
