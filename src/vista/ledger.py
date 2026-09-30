"""Foundation N3: cumulative disclosure control (Sec 22).

* threshold-probing attack (Prop 3.1)
* simulatable interval LEDGER with resolution rho (Thm 3.2)
* optimal claim-grid design OCGD (Sec 22.3)
The ledger decision depends only on (I, theta), never on x, so a forced FAIL leaks nothing.
"""
from __future__ import annotations

import math

import numpy as np

TRUE, FALSE, FAIL = "TRUE", "FALSE", "FAIL"


class IntervalLedger:
    def __init__(self, lo: int, hi: int, rho: int):
        self.lo, self.hi, self.rho = lo, hi, rho

    @property
    def width(self) -> int:
        return self.hi - self.lo + 1

    def decide(self, theta: int) -> str:
        """Public (simulatable) decision: 'known_true', 'known_false', 'split' or 'refuse'."""
        if theta <= self.lo:
            return "known_true"
        if theta > self.hi:
            return "known_false"
        ge, lt = self.hi - theta + 1, theta - self.lo
        return "split" if (ge >= self.rho and lt >= self.rho) else "refuse"

    def answer(self, x: int, theta: int) -> str:
        d = self.decide(theta)
        if d == "known_true":
            return TRUE
        if d == "known_false":
            return FALSE
        if d == "refuse":
            return FAIL
        if x >= theta:
            self.lo = theta
            return TRUE
        self.hi = theta - 1
        return FALSE


class LedgerStore:
    """Ledgers keyed by (payer, attribute, audience-group). audience=None => collusion-aware (global)."""

    def __init__(self, dims, rho: int, collusion_aware: bool = False):
        self.dims, self.rho, self.global_ = dims, rho, collusion_aware
        self.store: dict = {}

    def get(self, payer, attr: int, audience) -> IntervalLedger:
        key = (payer, attr, None if self.global_ else audience)
        if key not in self.store:
            self.store[key] = IntervalLedger(0, self.dims[attr] - 1, self.rho)
        return self.store[key]


# ---------------------------------------------------------------------------- adversaries
def probe(domain: int, x: int, strategy: str, rho: int | None, rng, max_queries: int = 400) -> dict:
    """Receiver adaptively picks thresholds. rho=None means no ledger (truthful answers)."""
    led = IntervalLedger(0, domain - 1, rho if rho else 1)
    lo, hi = 0, domain - 1  # receiver's own consistent interval
    q = 0
    scan_t = 1
    refused = set()
    while q < max_queries and hi > lo:
        if strategy == "binary":
            th = (lo + hi + 1) // 2
        elif strategy == "random":
            th = int(rng.integers(lo + 1, hi + 1))
        elif strategy == "scan":
            th = max(scan_t, lo + 1)
            scan_t = th + 1
            if th > hi:
                break
        else:
            raise ValueError(strategy)
        if rho:
            if (lo, hi, th) in refused and strategy == "binary":
                break
            ans = led.answer(x, th)
        else:
            ans = TRUE if x >= th else FALSE
        q += 1
        if ans == TRUE:
            lo = max(lo, th)
        elif ans == FALSE:
            hi = min(hi, th - 1)
        else:
            refused.add((lo, hi, th))
            if strategy == "binary":
                break  # a refusal at the midpoint means no further split is possible
            if strategy == "random" and all(led.decide(t) != "split" for t in range(lo + 1, hi + 1)):
                break
    width = hi - lo + 1
    return {"queries": q, "final_width": width, "exact": width == 1,
            "bits_learned": math.log2(domain / width), "consistent": lo <= x <= hi}


# ---------------------------------------------------------------------------- claim grids (OCGD)
def survival(prior_1d: np.ndarray):
    """S(t) = P(x >= t) for a 1-D prior over 0..D-1."""
    s = np.concatenate([np.cumsum(prior_1d[::-1])[::-1], [0.0]])
    return lambda t: float(s[min(max(int(t), 0), len(prior_1d))])


def round_up(theta: int, grid) -> int | None:
    g = [x for x in sorted(grid) if x >= theta]
    return g[0] if g else None


def grid_loss(thresholds, weights, grid, S) -> float:
    tot = 0.0
    for th, mu in zip(thresholds, weights):
        g = round_up(th, grid)
        tot += mu * (S(th) - (S(g) if g is not None else 0.0))
    return tot


def ocgd(thresholds, weights, k: int, S) -> tuple[list, float]:
    """Optimal k-point grid: split sorted thresholds into k consecutive blocks rounded up to each
    block maximum. f[g][b] = min_a f[g-1][a-1] + cost(a,b). O(k m^2)."""
    order = np.argsort(thresholds)
    th = [thresholds[i] for i in order]
    mu = [weights[i] for i in order]
    m = len(th)
    k = min(k, m)

    def cost(a, b):
        return sum(mu[j] * (S(th[j]) - S(th[b])) for j in range(a, b + 1))

    INF = math.inf
    f = [[INF] * m for _ in range(k + 1)]
    arg = [[-1] * m for _ in range(k + 1)]
    for b in range(m):
        f[1][b] = cost(0, b)
        arg[1][b] = 0
    for g in range(2, k + 1):
        for b in range(m):
            for a in range(1, b + 1):
                v = f[g - 1][a - 1] + cost(a, b)
                if v < f[g][b]:
                    f[g][b], arg[g][b] = v, a
    best_g = min(range(1, k + 1), key=lambda g: f[g][m - 1])
    grid, b, g = [], m - 1, best_g
    while g >= 1 and b >= 0:
        a = arg[g][b]
        grid.append(th[b])
        b, g = a - 1, g - 1
    return sorted(grid), f[best_g][m - 1]


def ocgd_bruteforce(thresholds, weights, k, S):
    import itertools
    best, bl = None, math.inf
    ths = sorted(set(thresholds))
    for r in range(1, k + 1):
        for comb in itertools.combinations(ths, r):
            if max(thresholds) not in comb:
                continue
            l = grid_loss(thresholds, weights, comb, S)
            if l < bl - 1e-12:
                best, bl = list(comb), l
    return best, bl
