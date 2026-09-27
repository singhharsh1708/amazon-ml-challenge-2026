"""Name and address classifiers used by the France audit.

Step context: Step 4 input (France audit). Rule and classifier audit of France assignments that produced judge_keep_removals, judge_keep_adds and judge_n2_adds.
"""
import re
from rapidfuzz import fuzz
STOP = {"sa","sas","sasu","sarl","eurl","sci","snc","ei","eirl","gmbh","inc","llc","ltd","pvt","private","limited","llp","lp",
        "corp","co","plc","pllc","pc","incorporated","corporation",
        "de","du","des","la","le","les","et","and","the","of","d","l","a","ms","shri","sri","smt","mr","dr","m","s"}
def toks(n):
    """Return the name tokens that are not stop words."""
    return [t for t in (n or "").split() if t not in STOP and len(t) > 1]
def name_class(qn, sn):
    """Classify how two names relate (same, subset, conflicting)."""
    q, s = toks(qn), toks(sn)
    if not q or not s:
        return ("empty", "", "")
    qs, ss = set(q), set(s)
    if qs == ss:
        return ("eq", "", "")
    if "".join(q) == "".join(s):
        return ("spaceless", "", "")
    if len(q) == 1 and 2 <= len(q[0]) <= 5 and len(s) >= 2:
        ini = "".join(t[0] for t in s)
        ini2 = "".join(t[0] for t in (sn or "").split() if t)
        if q[0] in (ini, ini2):
            return ("acr", "", "")
    add = sorted(qs - ss); rm = sorted(ss - qs)
    typo = 0
    for a in list(add):
        for r in rm:
            if fuzz.ratio(a, r) >= 80:
                add.remove(a); rm.remove(r); typo = 1; break
    if not add and not rm:
        return ("typo", "", "")
    if not (qs & ss) and not typo:
        return ("disjoint", " ".join(add), " ".join(rm))
    if add and not rm:
        return ("add", " ".join(add), "")
    if rm and not add:
        return ("rm", "", " ".join(rm))
    if len(add) == 1 and len(rm) == 1:
        return ("swap1", add[0], rm[0])
    return ("other", " ".join(add), " ".join(rm))
def nums(t):
    """Return the numbers found in a string."""
    return [x for x in t if x.isdigit()]
SKIPW = {"bis","ter","b","t","ndeg"}
def addr_class(qk, sk):
    """Classify how two addresses relate (same, compatible, conflicting)."""
    if not qk:
        return ("missing", None)
    if not sk:
        return ("s_missing", None)
    if qk == sk:
        return ("same", None)
    qt, st = qk.split(), sk.split()
    qn, sn = nums(qt), nums(st)
    qw = sorted(x for x in qt if not x.isdigit() and x not in SKIPW)
    sw = sorted(x for x in st if not x.isdigit() and x not in SKIPW)
    if qw == sw:
        if sorted(qn) == sorted(sn):
            return ("same_ns", None)
        if not qn and sn:
            return ("numdrop", None)
        if len(qn) == 1 and len(sn) == 1:
            return ("numchg1", int(qn[0]) - int(sn[0]))
        return ("numchg", None)
    ws = fuzz.token_sort_ratio(" ".join(qw), " ".join(sw))
    if sorted(qn) == sorted(sn) and qn:
        return ("same_num_wtypo" if ws >= 85 else "street", None)
    if len(qn) == 1 and len(sn) == 1 and ws >= 85:
        return ("numchg1_wtypo", int(qn[0]) - int(sn[0]))
    return ("other", None)
