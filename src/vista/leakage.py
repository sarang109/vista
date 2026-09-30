"""Foundation N4: correlation-aware surprisal leakage and submodularity (Sec 23).

f(S) = -log2 pi(conjunction of claims in S). Claim sets are represented by their profile
(join of the claims). Under a log-supermodular (FKG) prior f is monotone submodular and
subadditive (Thm 4.3); under negative dependence additive weights can underestimate.
"""
from __future__ import annotations

import math

import numpy as np

from .prior import Prior


def f_profile(prior: Prior, prof) -> float:
    return prior.surprisal(prof)


def claim_profile(n, claims):
    v = [0] * n
    for i, t in claims:
        v[i] = max(v[i], t)
    return tuple(v)


def submodularity_check(prior: Prior, rng, trials: int) -> dict:
    n = len(prior.dims)
    viol, worst = 0, 0.0
    subadd_viol = 0
    for _ in range(trials):
        base = []
        for _ in range(int(rng.integers(0, 3))):
            i = int(rng.integers(n)); base.append((i, int(rng.integers(1, prior.dims[i]))))
        i = int(rng.integers(n)); c = (i, int(rng.integers(1, prior.dims[i])))
        j = int(rng.integers(n)); d = (j, int(rng.integers(1, prior.dims[j])))
        fA = f_profile(prior, claim_profile(n, base))
        fAc = f_profile(prior, claim_profile(n, base + [c]))
        fAd = f_profile(prior, claim_profile(n, base + [d]))
        fAcd = f_profile(prior, claim_profile(n, base + [c, d]))
        gap = (fAcd - fAd) - (fAc - fA)  # > 0 violates submodularity
        if gap > 1e-9:
            viol += 1
            worst = max(worst, gap)
        S = base + [c, d]
        if f_profile(prior, claim_profile(n, S)) > sum(f_profile(prior, claim_profile(n, [s])) for s in S) + 1e-9:
            subadd_viol += 1
    return {"trials": trials, "submod_violations": viol, "worst_violation_bits": worst,
            "subadditivity_violations": subadd_viol}


def kl_bound_check(prior: Prior, rng, trials: int) -> int:
    """Prop 4.1: KL(P(Y|E) || P(Y)) <= log2(1/pi(E)) for Y = a coordinate of X, E an up-set."""
    n = len(prior.dims)
    viol = 0
    for _ in range(trials):
        prof = tuple(int(rng.integers(0, d)) for d in prior.dims)
        m = prior.mask_ge(prof)
        pe = prior.p[m].sum()
        y = int(rng.integers(n))
        py = np.bincount(prior.states[:, y], weights=prior.p, minlength=prior.dims[y])
        pye = np.bincount(prior.states[m, y], weights=prior.p[m], minlength=prior.dims[y]) / pe
        nz = pye > 0
        kl = float(np.sum(pye[nz] * np.log2(pye[nz] / py[nz])))
        if kl > -math.log2(pe) + 1e-9:
            viol += 1
    return viol


def additive_weights(prior: Prior):
    """Additive surrogate: w(i,t) = f({(i,t)})."""
    n = len(prior.dims)
    from .cost import AdditiveCost
    tab = []
    for i, d in enumerate(prior.dims):
        row = [0.0]
        for t in range(1, d):
            v = [0] * n; v[i] = t
            row.append(prior.surprisal(v))
        tab.append(row)
    return AdditiveCost(tab)


def sensitive_target_gain(prior: Prior, prof, target_attr: int, target_pred) -> float:
    """G_Y(l) = KL(P(Y | X >= l) || P(Y)) for a binary target Y = target_pred(x_target)."""
    m = prior.mask_ge(prof)
    y = target_pred(prior.states[:, target_attr])
    p1 = prior.p[y].sum()
    q1 = prior.p[m & y].sum() / prior.p[m].sum()
    out = 0.0
    for q, p in ((q1, p1), (1 - q1, 1 - p1)):
        if q > 0:
            out += q * math.log2(q / p)
    return out
