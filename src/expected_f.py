"""Expected-F0.5 selection of matches per source 1 entity.

Treats each candidate probability as an independent Bernoulli event and, for each source 1
entity, keeps the number of top candidates that maximises the expected per-entity F0.5 (Poisson
binomial distribution of true matches). Library module used by predict_submission.py.
"""

import numpy as np

MAX_CANDIDATES = 12


def poisson_binomial(ps):
    """Return the distribution of the number of successes for independent probabilities ps."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for q in ps:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return dist


def best_count(ps):
    """Return how many of the sorted probabilities to keep to maximise expected F0.5."""
    m = len(ps)
    best, arg = -1.0, 0
    for k in range(m + 1):
        a = poisson_binomial(ps[:k])
        b = poisson_binomial(ps[k:])
        i = np.arange(k + 1)[:, None]
        n = i + np.arange(m - k + 1)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(i == 0, 0.0, 1.25 * i / (1.25 * i + 0.25 * (n - i) + (k - i)))
        if k == 0:
            f = np.where(n == 0, 1.0, 0.0)
        value = float((a[:, None] * b[None, :] * f).sum())
        if value > best:
            best, arg = value, k
    return arg


def select(s1, p, decoy_weight=1.0, rid=None):
    """Return a keep mask choosing the expected-F0.5-optimal top candidates for every source 1 id."""
    order = np.lexsort((-p, s1)) if rid is None else np.lexsort((rid, -p, s1))
    s1_sorted, p_sorted = s1[order], p[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s1_sorted)) + 1]
    ends = np.r_[starts[1:], len(s1_sorted)]
    keep_sorted = np.zeros(len(s1_sorted), dtype=bool)
    for start, end in zip(starts, ends):
        ps = p_sorted[start:end][:MAX_CANDIDATES]
        q = ps / (ps + decoy_weight * (1 - ps))
        keep_sorted[start:start + best_count(q)] = True
    keep = np.zeros(len(s1), dtype=bool)
    keep[order] = keep_sorted
    return keep
