"""Review helpers for France pairs (name/address/key review and verdict) on top of france_rules.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
FINAL_STEPS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, f"{REPO_DIR}/src"); sys.path.insert(0, f"{FINAL_STEPS}/france_rerun/src")
from france_rules import review_name, review_address, review_key, KEEP_NOISE
KN = set(KEEP_NOISE) | {"compagnie", "et", "cb"}
SAMEA = {"same", "same_ns", "same_num_wtypo"}
def review_verdict(nc, nadd, nrm, ac, delta):
    """Decide whether a review passes from the counts and the address check."""
    aw = [w for w in nadd.split() if w]
    rw = [w for w in nrm.split() if w]
    noise_add = all(w in KN for w in aw)
    if ac in ("numchg1", "numchg1_wtypo", "numchg", "street"):
        if nc in ("eq", "typo", "spaceless", "acr") and ac in ("numchg1", "numchg1_wtypo") and delta is not None and abs(delta) not in (1, 2, 3, 4, 5, 7, 9, 11, 13, 21):
            return "unsure_num"
        return "diff_addr"
    if nc in ("eq", "typo", "spaceless", "acr"):
        return "same" if ac in SAMEA | {"missing", "s_missing", "numdrop"} else "unsure_addr"
    if nc in ("add", "swap1", "other", "rm"):
        if aw and not noise_add:
            return "diff_typeswap" if rw else "unsure_addword"
        if ac in SAMEA | {"missing", "s_missing", "numdrop"}:
            return "same_noise"
        return "unsure_addr"
    return "unsure_" + nc
def verdicts(qn, sn, qa, sa):
    """Return the review verdicts for a name/address pair."""
    return [review_verdict(*(review_name(a, b) + review_address(x, y))) for a, b, x, y in zip(qn, sn, qa, sa)]
