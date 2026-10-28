"""Repeated disclosure to the same receiver: branch-inference attack and sticky branch selection (SBS).

A sender pays the same receiver n times with unchanged evidence. With fresh draws V_1..V_n are i.i.d.
q(.|A(X)) given X, so the count vector N (how often each branch was shown) is a sufficient statistic.

  Lambda_n = I(X; V^n) - E[L_S(J_n)],  J_n = join of the branches shown (the claims the receiver holds)

  Theorem 3 (1) n >= ln(m/delta)/q_min fresh payments reveal every supported branch w.p. >= 1 - delta
            (2) I(X;V^n) <= n I(X;V) and I(X;V^n) -> I(X;A(X)) when q > 0 on A(x)
            (3) sticky selection repeats V_1 while it stays supported, so Lambda_n = Lambda_1

Sticky draws use a keyed pseudorandom function so the choice is reproducible for audit but unpredictable
to the receiver: u = HMAC-SHA256(k, key || counter) / 2^256, then inverse-CDF sampling from q(.|a).
"""
from __future__ import annotations

import hashlib
import hmac
import itertools
import math

import numpy as np

from .pcgs import SignatureSpace
from .util import canonical_json


# ---------------------------------------------------------------------------- count vectors
def compositions(n: int, m: int) -> np.ndarray:
    """All count vectors of n draws over m branches (C(n+m-1, m-1) rows)."""
    out = []
    for bars in itertools.combinations(range(n + m - 1), m - 1):
        prev, row = -1, []
        for b in bars:
            row.append(b - prev - 1)
            prev = b
        row.append(n + m - 1 - prev - 1)
        out.append(row)
    return np.array(out, dtype=np.int64).reshape(-1, m)


def _log_multinomial(N: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """log P(N | a) for every signature a (rows of Q) and count vector N (rows); -inf off support."""
    n = int(N[0].sum())
    lcoef = math.lgamma(n + 1) - np.array([sum(math.lgamma(c + 1) for c in row) for row in N])
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log(Q)
        t = np.where(N[None, :, :] > 0, N[None, :, :] * lq[:, None, :], 0.0)
    return lcoef[None, :] + t.sum(2)


def _log_mix(lp: np.ndarray, mass: np.ndarray) -> np.ndarray:
    mx = lp.max(0)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return mx + np.log((mass[:, None] * np.exp(lp - mx[None, :])).sum(0))


def repeated_metrics(ss: SignatureSpace, Q: np.ndarray, n: int, max_rows: int = 2_000_000,
                     mc_samples: int = 200_000, rng=None) -> dict:
    """Exact Lambda_n, I(X;V^n) and Bayes attacker accuracies after n fresh payments.

    Exact enumeration over count vectors when |signatures| x |compositions| <= max_rows, otherwise a
    Monte Carlo estimate with mc_samples draws (P(N) is still computed exactly for each draw)."""
    m = ss.m
    n_comp = math.comb(n + m - 1, m - 1)
    if n_comp * ss.n_signatures <= max_rows:
        N = compositions(n, m)
        lp = _log_multinomial(N, Q)  # signatures x compositions
        lP = _log_mix(lp, ss.mass)
        P = np.exp(lP)
        with np.errstate(invalid="ignore"):
            I = float(np.nansum(ss.mass[:, None] * np.exp(lp) * (lp - lP[None, :]))) / math.log(2)
        seen = N > 0
        upJ = ss.mass @ (~seen[None, :, :] | ss.sig[:, None, :]).all(2)  # pi_S(X >= J_n) per count vector
        live = P > 0
        EL = float((P[live] * -np.log2(upJ[live])).sum())
        joint = ss.mass[:, None] * np.exp(lp)
        out = {"n": n, "I_bits": I, "E_L_bits": EL, "Lambda": I - EL, "exact": True,
               "signature_recovery_acc": float(joint.max(0).sum())}
        for name, y in ss.extra.items():
            p1 = (y[:, None] * np.exp(lp)).sum(0)
            p0 = ((ss.mass - y)[:, None] * np.exp(lp)).sum(0)
            out[f"{name}_attack_acc"] = float(np.maximum(p1, p0).sum())
            out[f"{name}_attack_base"] = float(max(y.sum(), 1 - y.sum()))
        return out
    rng = rng or np.random.default_rng(0)
    a = rng.choice(ss.n_signatures, size=mc_samples, p=ss.mass)
    N = np.stack([rng.multinomial(n, Q[k]) for k in a])
    lp = _log_multinomial(N, Q)
    lP = _log_mix(lp, ss.mass)
    I = float(np.mean(lp[a, np.arange(mc_samples)] - lP)) / math.log(2)
    seen = N > 0
    upJ = np.array([ss.mass[(~s[None, :] | ss.sig).all(1)].sum() for s in seen])
    EL = float(np.mean(-np.log2(upJ)))
    post = np.exp(lp + np.log(ss.mass)[:, None] - lP[None, :])
    out = {"n": n, "I_bits": I, "E_L_bits": EL, "Lambda": I - EL, "exact": False,
           "signature_recovery_acc": float(np.mean(post.argmax(0) == a))}
    for name, y in ss.extra.items():
        frac = np.divide(y, ss.mass)
        truth = rng.random(mc_samples) < frac[a]
        pred = (frac[:, None] * post).sum(0) > 0.5
        out[f"{name}_attack_acc"] = float(np.mean(pred == truth))
        out[f"{name}_attack_base"] = float(max(y.sum(), 1 - y.sum()))
    return out


def sticky_metrics(ss: SignatureSpace, Q: np.ndarray, n: int) -> dict:
    """Theorem 3(3): with unchanged evidence V^n = (V_1, ..., V_1), so every n-payment quantity equals the
    single-payment one."""
    out = repeated_metrics(ss, Q, 1)
    out["n"] = n
    return out


# ---------------------------------------------------------------------------- Theorem 3 (1)
def coverage_payments(q_min: float, m: int, delta: float) -> int:
    """Payments after which every supported branch has been shown w.p. >= 1 - delta (union bound)."""
    return math.ceil(math.log(m / delta) / q_min)


def prob_all_seen(q: np.ndarray, n: int) -> float:
    """Exact P(every branch with q_v > 0 appears in n i.i.d. draws), by inclusion-exclusion."""
    sup = np.flatnonzero(q > 0)
    tot = 0.0
    for r in range(len(sup) + 1):
        for T in itertools.combinations(sup, r):
            tot += (-1) ** r * max(1.0 - float(q[list(T)].sum()), 0.0) ** n
    return tot


def q_min(ss: SignatureSpace, Q: np.ndarray) -> float:
    vals = Q[ss.sig]
    return float(vals.min()) if vals.size else 0.0


# ---------------------------------------------------------------------------- sticky selection
def prf_uniform(prf_key: bytes, key, counter: int = 0) -> float:
    """Keyed pseudorandom number in [0, 1) for a (sticky key, counter) pair."""
    d = hmac.new(prf_key, canonical_json({"k": list(key) if isinstance(key, tuple) else key, "c": counter}),
                 hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big") / 2.0 ** 64


def draw(probs: np.ndarray, u: float) -> int:
    c = np.cumsum(probs)
    return int(min(np.searchsorted(c, u * c[-1], side="right"), len(probs) - 1))


class StickyStore:
    """store[key] = branch index; key = (sender account, receiver frontier signature, evidence epoch)."""

    def __init__(self, prf_key: bytes):
        self.prf_key = prf_key
        self.store: dict = {}
        self.counters: dict = {}
        self.hits = self.redraws = self.first = 0

    def choose(self, key, supported: np.ndarray, probs: np.ndarray) -> tuple[int, str]:
        """Return (branch, event) where event is 'hit', 'first' or 'redraw' (stored branch unsupported)."""
        v = self.store.get(key)
        if v is not None and supported[v]:
            self.hits += 1
            return v, "hit"
        event = "first" if v is None else "redraw"
        c = self.counters.get(key, 0)
        self.counters[key] = c + 1
        v = draw(np.where(supported, probs, 0.0), prf_uniform(self.prf_key, key, c))
        self.store[key] = v
        if event == "first":
            self.first += 1
        else:
            self.redraws += 1
        return v, event
