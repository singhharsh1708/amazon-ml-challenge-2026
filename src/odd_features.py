"""Sibling-aware "odd" features for the stage 2 model.

For each pair the features compare the query record with the other records (siblings) that
also picked the same source 1 entity as their best match: counts of siblings, best sibling
name/address similarity, house-number differences and shifts, numbers and suffixes explained by
siblings, added and removed name words, and decoy/replacement word hits. Library module used by
build_odd_features.py.
"""

import json
import re

import numpy as np
import pyarrow as pa
from rapidfuzz import fuzz

from config import DATA_DIR

DECOY_FILE = DATA_DIR / "decoy_words.json"
REPLACEMENT_FILE = DATA_DIR / "replacement_words.json"
SIBLING_MIN_P = 0.5
TEXT_MIN_P = 0.001

LEGAL = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "lp", "plc", "pllc", "pc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "gmbh",
}
STOP = {
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "ms", "m", "s", "shri", "sri", "smt", "mr", "mrs",
    "dr", "dba", "formerly", "known", "as", "com", "www", "d", "l", "a",
}
SHIFT_DECOY = {3, 4, 5, 7, 9, 11, 13, 21}
NUM = re.compile(r"\d+")
SUFFIX = re.compile(r"[a-z]?\d+[a-z]")
ORDINAL = re.compile(r"\d+(st|nd|rd|th)")
NAMES = [
    "o_n5", "o_n9", "o_n9_x", "o_name_sib", "o_name_rel", "o_addr_sib", "o_addr_rel", "o_name_eq_sib", "o_addr_eq_sib",
    "o_d1", "o_d1_cls", "o_shift_d8", "o_shift_p12", "o_shift_n12", "o_q1_sib", "o_s1_sib", "o_odd_num",
    "o_qx_n", "o_qx_sib", "o_qx_unexpl", "o_sfx_new", "o_sfx_sib", "o_add_n", "o_add_unsup", "o_rem_n", "o_rem_kept",
    "o_veto", "o_l2",
]


def load_word_lists():
    """Load the decoy and replacement word lists as per-country sets."""
    def load(path):
        """Read one per-country word list JSON file into sets, or return {} if it is missing."""
        if not path.exists():
            return {}
        return {country: set(words) for country, words in json.loads(path.read_text()).items()}

    return load(DECOY_FILE), load(REPLACEMENT_FILE)


def numbers(addr):
    """Return all digit runs in an address, without leading zeros."""
    out = []
    for tok in (addr or "").split():
        for m in NUM.findall(tok):
            out.append(m.lstrip("0") or "0")
    return out


def suffixed(addr):
    """Return address tokens that are a number with a letter suffix (ordinals excluded)."""
    return {t for t in (addr or "").split() if SUFFIX.fullmatch(t) and not ORDINAL.fullmatch(t)}


def real(t):
    """Return True for a name token that is not a digit, legal form or stop word."""
    return len(t) >= 2 and not t.isdigit() and t not in LEGAL and t not in STOP


def matched(t, other_list, other_joined):
    """Return True if a token appears in the joined other name or fuzzily matches one of its tokens."""
    if len(t) >= 4 and t in other_joined:
        return True
    return any(fuzz.ratio(t, o) >= 80 for o in other_list)


def unexplained(qn, sn, sib_nums):
    """Count query house numbers not present in, contained in or concatenated into the source 1 numbers."""
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
    """Convert a digit string to int, using at most its first 9 digits."""
    return int(x[:9])


def pair_features(rid, country, q_name, q_addr, s_name, s_addr, sibs, decoys, replacements):
    """Compute the dict of odd features for one pair given its sibling list and word lists."""
    f = dict.fromkeys(NAMES, float("nan"))
    q_name, s_name, q_addr, s_addr = q_name or "", s_name or "", q_addr or "", s_addr or ""
    sib = [x for x in sibs if x[0] != rid]
    f["o_n5"] = len(sib)
    src = rid // 10000000000
    sib9 = [x for x in sib if x[1] >= 0.9]
    f["o_n9"] = len(sib9)
    f["o_n9_x"] = sum(1 for x in sib9 if x[2] != src)
    rn = fuzz.ratio(q_name, s_name)
    ra = fuzz.ratio(q_addr, s_addr)
    if sib:
        f["o_name_sib"] = max(fuzz.ratio(q_name, x[3]) for x in sib)
        f["o_name_rel"] = f["o_name_sib"] - rn
        f["o_addr_sib"] = max(fuzz.ratio(q_addr, x[4]) for x in sib)
        f["o_addr_rel"] = f["o_addr_sib"] - ra
        f["o_name_eq_sib"] = sum(1 for x in sib if x[3] == q_name)
        f["o_addr_eq_sib"] = sum(1 for x in sib if x[4] == q_addr)
    qn, sn = numbers(q_addr), numbers(s_addr)
    sib_nums = set()
    for x in sib:
        sib_nums |= x[5]
    if qn and sn:
        d = ival(qn[0]) - ival(sn[0])
        f["o_d1"] = max(-100, min(100, d))
        f["o_d1_cls"] = 0 if d == 0 else 1 if d in SHIFT_DECOY else 2 if d in (1, 2) else 3 if d in (-1, -2) else 4
    qset, sset = set(qn), set(sn)
    qx = [n for n in dict.fromkeys(qn) if n not in sset]
    sx = [n for n in dict.fromkeys(sn) if n not in qset]
    ds = {ival(a) - ival(b) for a in qx for b in sx}
    f["o_shift_d8"] = int(bool(ds & SHIFT_DECOY))
    f["o_shift_p12"] = int(bool(ds & {1, 2}))
    f["o_shift_n12"] = int(bool(ds & {-1, -2}))
    if qn:
        f["o_q1_sib"] = sum(1 for x in sib if qn[0] in x[5])
    if sn:
        f["o_s1_sib"] = sum(1 for x in sib if sn[0] in x[5])
    if qn and sn:
        f["o_odd_num"] = int(qn[0] != sn[0] and f["o_s1_sib"] > 0 and f["o_q1_sib"] == 0)
    f["o_qx_n"] = len(qx)
    f["o_qx_sib"] = sum(1 for n in qx if n in sib_nums)
    f["o_qx_unexpl"] = unexplained(qn, sn, sib_nums) if qx else 0
    new_sfx = suffixed(q_addr) - set(s_addr.split())
    f["o_sfx_new"] = int(bool(new_sfx))
    f["o_sfx_sib"] = int(any(any(t in x[4].split() for x in sib) for t in new_sfx)) if new_sfx else 0
    q_tok, s_tok = q_name.split(), s_name.split()
    qs, ss = set(q_tok), set(s_tok)
    s_joined, q_joined = "".join(s_tok), "".join(q_tok)
    added = [t for t in dict.fromkeys(q_tok) if t not in ss and real(t) and not matched(t, s_tok, s_joined)]
    removed = [t for t in dict.fromkeys(s_tok) if t not in qs and real(t) and not matched(t, q_tok, q_joined)]
    sib_tok = set()
    for x in sib:
        sib_tok |= x[6]
    f["o_add_n"] = len(added)
    f["o_add_unsup"] = sum(1 for t in added if t not in sib_tok)
    f["o_rem_n"] = len(removed)
    if sib:
        f["o_rem_kept"] = sum(1 for t in removed if all(t in x[6] for x in sib))
    dec = decoys.get(country, set())
    f["o_veto"] = int(any(t in dec and t not in ss for t in q_tok if len(t) >= 3 and t.isalpha()))
    rep = replacements.get(country, set())
    f["o_l2"] = int(any(t in rep for t in added if len(t) >= 3))
    return f


def sibling_index(table):
    """Group sibling rows by source 1 id into tuples used by pair_features."""
    idx = {}
    cols = {n: table.column(n).to_pylist() for n in ["s1", "sid", "sp1", "src", "name", "addr"]}
    for s1, sid, sp1, src, name, addr in zip(cols["s1"], cols["sid"], cols["sp1"], cols["src"], cols["name"], cols["addr"]):
        name, addr = name or "", addr or ""
        idx.setdefault(s1, []).append((sid, sp1, src, name, addr, set(numbers(addr)), set(name.split())))
    return idx


def feature_table(text, idx, decoys, replacements):
    """Compute odd features for every pair in an Arrow table and return them as an Arrow table."""
    cols = {n: text.column(n).to_pylist() for n in ["rid", "s1", "country", "q_name", "q_addr", "s_name", "s_addr"]}
    rows = [
        pair_features(r, co, qn, qa, sn, sa, idx.get(s, ()), decoys, replacements)
        for r, s, co, qn, qa, sn, sa in zip(
            cols["rid"], cols["s1"], cols["country"], cols["q_name"], cols["q_addr"], cols["s_name"], cols["s_addr"]
        )
    ]
    return pa.table(
        {"rid": text.column("rid"), "s1": text.column("s1")}
        | {n: pa.array(np.array([r[n] for r in rows], dtype=np.float32)) for n in NAMES}
    )


def sibling_sql(best, names, where="true"):
    """Return SQL selecting best-match siblings with probability at least SIBLING_MIN_P."""
    return f"""
        SELECT b.s1, b.rid AS sid, b.p AS sp1, b.rid // 10000000000 AS src, q.name_full AS name, q.address AS addr
        FROM {best} b JOIN {names} q ON q.eid = b.rid WHERE b.p >= {SIBLING_MIN_P} AND {where}
    """


def text_sql(pairs, names, where="true"):
    """Return SQL joining pairs above TEXT_MIN_P with the query and source 1 text."""
    return f"""
        SELECT p.rid, p.s1, q.country, q.name_full AS q_name, q.address AS q_addr, s.name_full AS s_name, s.address AS s_addr
        FROM {pairs} p JOIN {names} q ON q.eid = p.rid JOIN {names} s ON s.eid = p.s1
        WHERE p.p >= {TEXT_MIN_P} AND {where}
    """
