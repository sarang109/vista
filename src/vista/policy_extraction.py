"""Foundation N7: policy-extraction bounds and requirement tokens (Sec 26).

Every payment attempt is a membership query y in {0,1}^n on the receiver's monotone policy.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


class PolicyOracle:
    """Monotone DNF over n atoms (terms = frozensets). Counts queries; optional budget."""

    def __init__(self, n: int, terms, budget: int | None = None):
        self.n = n
        self.terms = [frozenset(t) for t in terms]
        self.queries = 0
        self.budget = budget

    def __call__(self, y: frozenset) -> bool:
        if self.budget is not None and self.queries >= self.budget:
            raise BudgetExceeded()
        self.queries += 1
        return any(t <= y for t in self.terms)

    def truth(self, y) -> bool:
        return any(t <= y for t in self.terms)


class BudgetExceeded(Exception):
    pass


def log2_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def counting_bound_exact(n: int, k: int, m: int) -> float:
    """log2 |F| for m-term k-uniform monotone DNFs: log2 C(C(n,k), m)."""
    N = math.comb(n, k)
    return log2_comb(N, m)


def counting_bound_closed(n: int, k: int, m: int) -> float:
    return m * (k * math.log2(n / k) - math.log2(m))


def min_queries(logF: float, alphabet: int) -> int:
    return math.ceil(logF / math.log2(alphabet))


def extract_conjunction(oracle: PolicyOracle) -> frozenset:
    """Prop 7.2: exactly n queries for single-conjunction policies."""
    y = set(range(oracle.n))
    for i in range(oracle.n):
        if oracle(frozenset(y - {i})):
            y.discard(i)
    return frozenset(y)


def _shrink(oracle, y: set) -> frozenset:
    for i in sorted(y):
        if oracle(frozenset(y - {i})):
            y = y - {i}
    return frozenset(y)


def minimal_transversals(sets, n: int):
    """Minimal hitting sets of a family (brute force by size; fine for n <= 16)."""
    out = []
    for r in range(0, n + 1):
        for T in itertools.combinations(range(n), r):
            Ts = set(T)
            if all(Ts & s for s in sets) and not any(set(o) <= Ts for o in out):
                out.append(frozenset(T))
    return out


def extract_dnf(oracle: PolicyOracle) -> dict:
    """Minterm-shrinking attack (Prop 7.3). Stops when budget is exhausted."""
    found = []
    complete = False
    try:
        if not oracle(frozenset(range(oracle.n))):
            return {"terms": [], "complete": True, "queries": oracle.queries}
        found.append(_shrink(oracle, set(range(oracle.n))))
        while True:
            progressed = False
            for T in minimal_transversals(found, oracle.n):
                y = set(range(oracle.n)) - T  # contains no known minterm
                if oracle(frozenset(y)):
                    found.append(_shrink(oracle, y))
                    progressed = True
                    break
            if not progressed:
                complete = True
                break
    except BudgetExceeded:
        pass
    return {"terms": found, "complete": complete, "queries": oracle.queries}


def agreement(true_terms, hyp_terms, n, rng, samples=4000, p_one=0.5) -> float:
    Y = rng.random((samples, n)) < p_one
    ok = 0
    for row in Y:
        y = frozenset(np.flatnonzero(row).tolist())
        ok += any(t <= y for t in true_terms) == any(t <= y for t in hyp_terms)
    return ok / samples


def random_kdnf(rng, n, k, m):
    terms = set()
    while len(terms) < m:
        terms.add(frozenset(rng.choice(n, size=k, replace=False).tolist()))
    # keep an antichain (drop terms containing another term)
    terms = [t for t in terms if not any(o < t for o in terms)]
    return terms
