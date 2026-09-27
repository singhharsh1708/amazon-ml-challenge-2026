"""Veto rules for assigned pairs based on name words.

A pair is vetoed when the query name adds a learned decoy word, when it both adds and loses a
real (frequent, non-noise) word relative to the source 1 name (word swap), or when it swaps a
single business-type word (type swap). Library module used by predict_submission.py.
"""

import json
from collections import Counter

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from config import DATA_DIR

DECOY_FILE = DATA_DIR / "decoy_words.json"
SWAP_MAX_P = 0.99
VOCAB_MIN = 50

NOISE = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "lp", "plc", "pllc", "pc", "sarl", "sas", "sasu", "eurl",
    "sa", "sci", "snc", "cie", "fils", "et", "and", "the", "of", "de", "du", "des", "la", "le",
    "les", "associes", "ei", "services", "service", "center", "centre", "shri", "sri", "ms",
    "formerly", "known", "as", "dba", "com", "www",
}


def load_decoy_words():
    """Load the per-country decoy word sets, or an empty dict when the file is missing."""
    if not DECOY_FILE.exists():
        return {}
    return {country: set(words) for country, words in json.loads(DECOY_FILE.read_text()).items()}


def tokens(text):
    """Return alphabetic tokens of length 3 or more."""
    return [t for t in (text or "").split() if len(t) >= 3 and t.isalpha()]


def build_vocab(countries, names):
    """Count per-country document frequencies of name tokens."""
    vocab = {}
    for country, name in zip(countries, names):
        vocab.setdefault(country, Counter()).update(set(tokens(name)))
    return vocab


def _real_new(own, other, other_joined, decoys, vocab):
    """Return tokens of one name that are frequent, not noise or decoy, and absent from the other name."""
    out = []
    for t in own:
        if t in other or t in NOISE or t in decoys:
            continue
        if vocab.get(t, 0) < VOCAB_MIN or t in other_joined:
            continue
        if any(fuzz.ratio(t, o) >= 70 for o in other):
            continue
        out.append(t)
    return out


def veto_flags(countries, q_names, s_names, probs, vocab, decoy_words, swap_countries=None):
    """Return boolean arrays (decoy_hit, swap_hit) for a batch of candidate assignments."""
    decoy_hit = np.zeros(len(probs), dtype=bool)
    swap_hit = np.zeros(len(probs), dtype=bool)
    for i, (country, q_name, s_name, p) in enumerate(zip(countries, q_names, s_names, probs)):
        q_tok, s_tok = tokens(q_name), tokens(s_name)
        q_set, s_set = set(q_tok), set(s_tok)
        decoys = decoy_words.get(country, set())
        if decoys and any(t in decoys and t not in s_set for t in q_tok):
            decoy_hit[i] = True
            continue
        if p >= SWAP_MAX_P or (swap_countries is not None and country not in swap_countries):
            continue
        country_vocab = vocab.get(country, {})
        added = _real_new(q_tok, s_set, "".join(s_tok), decoys, country_vocab)
        if not added:
            continue
        lost = _real_new(s_tok, q_set, "".join(q_tok), decoys, country_vocab)
        if lost:
            swap_hit[i] = True
    return decoy_hit, swap_hit


TRADE_WORDS = {
    "fils", "cie", "compagnie", "associes", "groupe", "developpement", "frs", "freres", "st", "saint",
    "ste", "sainte", "ets", "etablissements", "societe", "partners", "id", "lnc",
}
TYPE_WORD_SHARE = 0.005


def type_vocab(countries, name_cores, addresses):
    """Build per-country sets of frequent business-type words that are not common address words."""
    names, places, totals = {}, {}, Counter()
    for country, core, address in zip(countries, name_cores, addresses):
        totals[country] += 1
        names.setdefault(country, Counter()).update(set((core or "").split()))
        places.setdefault(country, Counter()).update(set((address or "").split()))
    vocab = {}
    for country, counts in names.items():
        floor = TYPE_WORD_SHARE * totals[country]
        place_counts = places.get(country, Counter())
        vocab[country] = {
            t for t, n in counts.items()
            if n >= floor and t.isalpha() and len(t) >= 3 and t not in TRADE_WORDS and place_counts[t] < floor
        }
    return vocab


def type_swap_flags(countries, q_cores, s_cores, vocab, gated_countries=None):
    """Flag pairs whose names differ by one added business-type word replacing at most one other word."""
    hit = np.zeros(len(q_cores), dtype=bool)
    for i, (country, q_core, s_core) in enumerate(zip(countries, q_cores, s_cores)):
        if gated_countries is not None and country not in gated_countries:
            continue
        q_tok, s_tok = set((q_core or "").split()), set((s_core or "").split())
        added, removed = q_tok - s_tok, s_tok - q_tok
        if len(added) != 1 or len(removed) > 1:
            continue
        new = next(iter(added))
        if new not in vocab.get(country, set()):
            continue
        if removed and JaroWinkler.normalized_similarity(new, next(iter(removed))) >= 0.75:
            continue
        hit[i] = True
    return hit
