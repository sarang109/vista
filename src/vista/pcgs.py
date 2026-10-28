"""Selection-leakage theory on the signature space: no-free-selection test, prior-corrected Gibbs
selection (PCGS) and the exact leakage-cost frontier.

Notation (see docs/theory.md):
  pi_S        evidence prior conditioned on Phi = 1 (satisfiable states only)
  A(x)        signature: the frontier branches supported by x (v in A(x) iff x >= v)
  q(v|a)      selection kernel, supported on the signature a
  up(v)       pi_S(X >= v) = sum of pi_S(a) over signatures a containing v
  L_S(v)      -log2 up(v)

  Lemma 1     Lambda(q) = I(X;V) - E[L_S(V)]
  Lemma 2     an optimal kernel depends on x only through A(x) (data processing); the problem therefore
              lives on at most 2^m signatures instead of the full state space
  Theorem 1   Lambda = 0 is attainable iff  sum_{v in a} w_v = 1 for every signature a, w >= 0, is feasible
  Theorem 2   min Lambda s.t. E[C] <= B is convex; its Lagrangian minimiser has the Gibbs form
              q(v|a) ~ r(v) exp(-mu C(v)) / up(v) and is computed by Blahut-Arimoto alternation

Information quantities are in bits unless a name ends in _nats. For kernels that depend on x only through
A(x) (DET, the exponential mechanism, PCGS) every quantity here equals the state-level value computed by
selection.SelectionSpace.metrics.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

LN2 = math.log(2.0)


@dataclass
class Compiled:
    """PCGS-Compile output for one policy and prior (Algorithm 1)."""
    Q: np.ndarray  # signature x branch kernel table
    mu: float
    budget: float
    lam_bits: float
    cost: float
    gap_bits: float  # certified Lambda(Q) - Lambda*(B) upper bound
    iterations: int
    seconds: float
    r: np.ndarray = field(repr=False, default=None)


class SignatureSpace:
    def __init__(self, sig: np.ndarray, mass: np.ndarray, C: np.ndarray, extra: dict | None = None):
        sig = np.asarray(sig, dtype=bool)
        mass = np.asarray(mass, dtype=float)
        keep = (mass > 0) & sig.any(1)
        self.sig, self.mass = sig[keep], mass[keep] / mass[keep].sum()
        self.C = np.asarray(C, dtype=float)
        self.m = self.sig.shape[1]
        self.up = self.mass @ self.sig  # pi_S(X >= v)
        self.live = self.up > 0
        with np.errstate(divide="ignore"):
            self.L = np.where(self.live, -np.log2(np.where(self.live, self.up, 1.0)), np.inf)
        # optional per-signature masses of a binary target Y (for Bayes attackers): extra[name] = P(A=a, Y=1)
        self.extra = {k: np.asarray(v, dtype=float)[keep] / mass[keep].sum() for k, v in (extra or {}).items()}
        self.codes = self.sig.astype(np.int64) @ (1 << np.arange(self.m, dtype=np.int64))

    # ------------------------------------------------------------------ construction (Lemma 2)
    @classmethod
    def from_states(cls, avail: np.ndarray, p: np.ndarray, C: np.ndarray, targets: dict | None = None):
        """Aggregate a state-level space (avail: N x m bool, p: prior over N states) by signature."""
        acc = avail.any(1) & (p > 0)
        codes = avail[acc].astype(np.int64) @ (1 << np.arange(avail.shape[1], dtype=np.int64))
        uniq, inv = np.unique(codes, return_inverse=True)
        mass = np.bincount(inv, weights=p[acc], minlength=len(uniq))
        sig = ((uniq[:, None] >> np.arange(avail.shape[1])) & 1).astype(bool)
        extra = {k: np.bincount(inv, weights=(p * t)[acc], minlength=len(uniq)) for k, t in (targets or {}).items()}
        return cls(sig, mass, C, extra)

    def index_of(self, sig_row) -> int:
        code = int(np.asarray(sig_row, dtype=np.int64) @ (1 << np.arange(self.m, dtype=np.int64)))
        hit = np.flatnonzero(self.codes == code)
        return int(hit[0]) if len(hit) else -1

    @property
    def n_signatures(self) -> int:
        return len(self.mass)

    # ------------------------------------------------------------------ kernels
    def _normalise(self, logits: np.ndarray) -> np.ndarray:
        logits = np.where(self.sig, logits, -np.inf)
        mx = logits.max(1, keepdims=True)
        z = np.exp(logits - mx)
        return z / z.sum(1, keepdims=True)

    def kernel_det(self) -> np.ndarray:
        Q = np.zeros(self.sig.shape)
        Q[np.arange(len(Q)), np.argmin(np.where(self.sig, self.C[None, :], np.inf), axis=1)] = 1.0
        return Q

    def kernel_exp(self, lam: float) -> np.ndarray:
        return self._normalise(np.broadcast_to(-lam * (self.C - self.C.min()), self.sig.shape))

    def kernel_uniform(self) -> np.ndarray:
        return self._normalise(np.zeros(self.sig.shape))

    def gibbs(self, r: np.ndarray, mu: float) -> np.ndarray:
        with np.errstate(divide="ignore"):
            lr = np.log(np.maximum(r, 0.0))
            lu = np.log(np.where(self.live, self.up, 1.0))
        return self._normalise(np.broadcast_to(lr - mu * (self.C - self.C.min()) - lu, self.sig.shape))

    # ------------------------------------------------------------------ metrics (Lemma 1)
    def metrics(self, Q: np.ndarray) -> dict:
        P = self.mass @ Q
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(Q > 0, Q / np.where(P > 0, P, 1.0)[None, :], 1.0)
            I = float((self.mass[:, None] * Q * np.log2(ratio)).sum())
        used = P > 0
        EL = float((P[used] * self.L[used]).sum())
        return {"I_bits": I, "E_L_bits": EL, "Lambda": I - EL, "cost": float(P @ self.C), "P": P}

    def lam(self, Q) -> float:
        return self.metrics(Q)["Lambda"]

    def cost(self, Q) -> float:
        return float(self.mass @ Q @ self.C)

    # ------------------------------------------------------------------ Theorem 1
    def zero_leakage_system(self) -> dict:
        """Feasibility of sum_{v in a} w_v = 1 (all signatures a), w >= 0, solved as an L1 residual LP:
        min sum(s+ + s-) s.t. S w + s+ - s- = 1. Residual 0 <=> some kernel has Lambda = 0."""
        k, m = self.sig.shape
        A = np.hstack([self.sig.astype(float), np.eye(k), -np.eye(k)])
        c = np.concatenate([np.zeros(m), np.ones(2 * k)])
        res = linprog(c, A_eq=A, b_eq=np.ones(k), bounds=(0, None), method="highs")
        resid = float(res.fun) if res.status == 0 else math.inf
        w = res.x[:m] if res.status == 0 else None
        return {"feasible": resid < 1e-9, "residual": resid, "w": w}

    def zero_leakage_kernel(self, w: np.ndarray) -> np.ndarray:
        """The Lambda = 0 kernel q(v|a) = w_v from a solution of the Theorem 1 system."""
        Q = np.where(self.sig, np.maximum(w, 0.0)[None, :], 0.0)
        return Q / Q.sum(1, keepdims=True)

    def two_branch_obstruction(self) -> tuple | None:
        """Sufficient condition of Theorem 1: branches a != b with singleton signatures {a}, {b} of positive
        mass and some positive-mass signature containing both. Returns (a, b) or None."""
        single = [int(np.flatnonzero(s)[0]) for s in self.sig if s.sum() == 1]
        for i, a in enumerate(single):
            for b in single[i + 1:]:
                if (self.sig[:, a] & self.sig[:, b]).any():
                    return a, b
        return None

    # ------------------------------------------------------------------ Theorem 2: Blahut-Arimoto
    def _bounds(self, r: np.ndarray, mu: float):
        """Upper and lower bounds on F*(mu) = min_q Lambda_nats(q) + mu E[C - C_min] given r."""
        # log domain: exp(-mu C) underflows for large mu
        with np.errstate(divide="ignore"):
            la = np.where(self.live, -mu * (self.C - self.C.min()) - np.log(np.where(self.live, self.up, 1.0)), -np.inf)
            lr = np.log(np.maximum(r, 0.0))
        t = np.where(self.sig, (lr + la)[None, :], -np.inf)
        mx = t.max(1, keepdims=True)
        logZ = (mx + np.log(np.exp(t - mx).sum(1, keepdims=True)))[:, 0]
        upper = -float(self.mass @ logZ)
        # log c(v) = la(v) + log sum_{a containing v} mass_a / Z_a
        w = np.where(self.sig, (np.log(self.mass) - logZ)[:, None], -np.inf)
        wm = w.max(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lc = la + np.where(np.isfinite(wm), wm + np.log(np.exp(w - np.where(np.isfinite(wm), wm, 0.0)).sum(0)), -np.inf)
        return upper, upper - max(float(np.max(lc[np.isfinite(lc)])) if np.isfinite(lc).any() else 0.0, 0.0)

    def blahut_arimoto(self, mu: float, r0: np.ndarray | None = None, tol: float = 1e-10,
                       max_iter: int = 20_000) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Minimise Lambda_nats + mu E[C]. Returns (Q, r, gap_nats, iterations).

        Q = gibbs(r_t) satisfies F(Q) <= G(Q, r_t) = upper(r_t), and F* >= lower(r_{t+1}) for any r_{t+1},
        so gap = upper(r_t) - lower(r_{t+1}) certifies F(Q) - F* <= gap."""
        r = np.where(self.live, 1.0, 0.0) if r0 is None else np.where(self.live, np.maximum(r0, 1e-300), 0.0)
        r = r / r.sum()
        gap, it = math.inf, 0
        while it < max_iter:
            it += 1
            check = it % 10 == 0 or it == 1 or it == max_iter
            if check:
                up_old, _ = self._bounds(r, mu)
            Q = self.gibbs(r, mu)
            r = self.mass @ Q
            if check:
                gap = up_old - self._bounds(r, mu)[1]
                if gap < tol:
                    break
        return Q, r, max(gap, 0.0), it

    def lambda_floor(self, tol: float = 1e-10) -> dict:
        """Smallest Lambda over all kernels (mu = 0), with a certified lower bound."""
        Q, r, gap, it = self.blahut_arimoto(0.0, tol=tol)
        _, lo = self._bounds(r, 0.0)
        return {"Q": Q, "Lambda": self.lam(Q), "lower_bound": max(lo, 0.0) / LN2, "gap_bits": gap / LN2,
                "iterations": it}

    def compile(self, overhead: float, tol: float = 1e-10, bisect: int = 40) -> Compiled:
        """Algorithm 1 (PCGS-Compile): minimum-Lambda kernel with E[C] <= (1 + overhead) * E[C] under DET."""
        t0 = time.perf_counter()
        c_det = self.cost(self.kernel_det())
        B = c_det + overhead * abs(c_det) if c_det != 0 else overhead
        Q0, r0, g0, it = self.blahut_arimoto(0.0, tol=tol)
        total_it = it
        if self.cost(Q0) <= B + 1e-12:
            Q, r, mu, gap = Q0, r0, 0.0, g0
        else:
            lo, hi = 0.0, 1.0
            Qh, rh, gh, it = self.blahut_arimoto(hi, r0, tol)
            total_it += it
            while self.cost(Qh) > B and hi < 1e8:
                lo, hi = hi, hi * 4
                Qh, rh, gh, it = self.blahut_arimoto(hi, rh, tol)
                total_it += it
            for _ in range(bisect):
                mid = 0.5 * (lo + hi)
                Qm, rm, gm, it = self.blahut_arimoto(mid, rh, tol)
                total_it += it
                if self.cost(Qm) <= B:
                    hi, Qh, rh, gh = mid, Qm, rm, gm
                else:
                    lo = mid
                if hi - lo < 1e-9 * max(1.0, hi):
                    break
            Q, r, mu, gap = Qh, rh, hi, gh
        cost = self.cost(Q)
        lam = self.lam(Q)
        # weak duality: Lambda*(B) >= F*(mu) - mu (B - C_min)  (all in nats)
        _, lo_F = self._bounds(r, mu)
        lower = (lo_F - mu * (B - self.C.min())) / LN2
        return Compiled(Q, mu, B, lam, cost, max(lam - lower, 0.0), total_it, time.perf_counter() - t0, r)

    def frontier(self, mus) -> list[dict]:
        """Exact leakage-cost frontier: Lagrangian minimisers for a sweep of mu (warm-started)."""
        out, r = [], None
        for mu in sorted(mus):
            Q, r, gap, it = self.blahut_arimoto(float(mu), r)
            m = self.metrics(Q)
            out.append({"mu": float(mu), "Lambda": m["Lambda"], "cost": m["cost"], "gap_bits": gap / LN2,
                        "iterations": it})
        return out

    # ------------------------------------------------------------------ equal-cost baselines
    def exp_at_budget(self, B: float, iters: int = 60) -> tuple[np.ndarray, float]:
        """Exponential mechanism with the smallest lambda whose expected cost is <= B (cost falls in lambda)."""
        if self.cost(self.kernel_exp(0.0)) <= B:
            return self.kernel_exp(0.0), 0.0
        lo, hi = 0.0, 1.0
        while self.cost(self.kernel_exp(hi)) > B and hi < 1e8:
            lo, hi = hi, hi * 4
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.cost(self.kernel_exp(mid)) <= B:
                hi = mid
            else:
                lo = mid
        return self.kernel_exp(hi), hi


def lp_at_budget(space, B: float, iters: int = 14, eps_hi: float = 12.0):
    """State-level LP eps-private kernel (selection.SelectionSpace.kernel_lp) with the smallest eps whose
    expected cost is <= B. Returns (K, eps) or (None, inf) when no eps <= eps_hi fits the budget."""
    K = space.kernel_lp(eps_hi)
    if K is None or space.metrics(K)["exp_cost"] > B + 1e-9:
        return None, math.inf
    lo, hi, best = 0.0, eps_hi, K
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Km = space.kernel_lp(mid)
        if Km is not None and space.metrics(Km)["exp_cost"] <= B + 1e-9:
            hi, best = mid, Km
        else:
            lo = mid
    return best, hi
