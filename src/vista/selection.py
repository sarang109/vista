"""Foundation N2: selection-channel leakage and the ideal-disclosure floor (Sec 21).

A selection rule is a row-stochastic kernel K[x, v] over frontier elements v in Avail(x).
All quantities are computed exactly over a finite evidence space with prior pi.

  L(v)      = -log2 pi(X >= v)                          atom-only surprisal
  Total     = E_V KL(P(X|V) || pi)                     leakage relative to the prior
  Lambda    = E_V KL(P(X|V) || pi(. | X >= v))         selection leakage  (Thm 2.1: Total = E[L] + Lambda)
  eps_sel   = max_v max_{x',x'' >= v} ln K[x',v]/K[x'',v]
  L_min     = -log2 pi(Phi = 1)                        information floor (BOP attains it, Prop 2.6)
  PoD       = E[L(V)] - L_min                          price of atomic disclosure
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .prior import Prior


@dataclass
class SelectionSpace:
    prior: Prior
    front: np.ndarray  # |M| x n
    w: np.ndarray  # cost per frontier element

    def __post_init__(self):
        S = self.prior.states
        self.avail = (S[:, None, :] >= self.front[None, :, :]).all(2)  # N x |M|
        self.accept = self.avail.any(1)
        self.pA = float(self.prior.p[self.accept].sum())
        self.L = np.array([self.prior.surprisal(u) for u in self.front])

    @property
    def L_min(self) -> float:
        return -math.log2(self.pA)

    # ------------------------------------------------------------------ kernels
    def kernel_det(self) -> np.ndarray:
        K = np.zeros(self.avail.shape)
        big = np.where(self.avail, self.w[None, :], np.inf)
        rows = np.flatnonzero(self.accept)
        K[rows, np.argmin(big[rows], axis=1)] = 1.0
        return K

    def kernel_exp(self, lam: float) -> np.ndarray:
        logits = np.where(self.avail, -lam * (self.w - self.w.min())[None, :], -np.inf)
        K = np.zeros(self.avail.shape)
        rows = self.accept
        z = np.exp(logits[rows] - logits[rows].max(1, keepdims=True))
        K[rows] = z / z.sum(1, keepdims=True)
        return K

    def kernel_lp(self, eps: float) -> np.ndarray | None:
        """Prop 2.5: cost-optimal eps-selection-private kernel as a linear program.
        Uses m_v <= P(v|x) <= M_v, M_v <= e^eps m_v for all x >= v (equivalent to the pairwise ratio)."""
        acc = np.flatnonzero(self.accept)
        pairs = [(x, v) for x in acc for v in np.flatnonzero(self.avail[x])]
        nP = len(pairs)
        nM = len(self.front)
        nv = nP + 2 * nM  # P vars, m_v, M_v
        pa = self.prior.p / self.pA
        c = np.zeros(nv)
        for k, (x, v) in enumerate(pairs):
            c[k] = pa[x] * self.w[v]
        # equality: sum_v P(v|x) = 1
        er, ec, ev = [], [], []
        row_of = {x: r for r, x in enumerate(acc)}
        for k, (x, v) in enumerate(pairs):
            er.append(row_of[x]); ec.append(k); ev.append(1.0)
        Aeq = coo_matrix((ev, (er, ec)), shape=(len(acc), nv)).tocsr()
        beq = np.ones(len(acc))
        ur, uc, uv = [], [], []
        r = 0
        for k, (x, v) in enumerate(pairs):
            # m_v - P <= 0
            ur += [r, r]; uc += [nP + v, k]; uv += [1.0, -1.0]; r += 1
            # P - M_v <= 0
            ur += [r, r]; uc += [k, nP + nM + v]; uv += [1.0, -1.0]; r += 1
        for v in range(nM):
            ur += [r, r]; uc += [nP + nM + v, nP + v]; uv += [1.0, -math.exp(eps)]; r += 1
        Aub = coo_matrix((uv, (ur, uc)), shape=(r, nv)).tocsr()
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(r), A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
        if res.status != 0:
            return None
        K = np.zeros(self.avail.shape)
        for k, (x, v) in enumerate(pairs):
            K[x, v] = max(res.x[k], 0.0)
        K[acc] /= K[acc].sum(1, keepdims=True)
        return K

    def min_feasible_eps(self, lo=0.0, hi=12.0, iters=30) -> float:
        if self.kernel_lp(hi) is None:
            return math.inf
        for _ in range(iters):
            mid = (lo + hi) / 2
            if self.kernel_lp(mid) is None:
                lo = mid
            else:
                hi = mid
        return hi

    # ------------------------------------------------------------------ metrics
    def metrics(self, K: np.ndarray, target_mask: np.ndarray | None = None) -> dict:
        p = self.prior.p
        pa = np.where(self.accept, p / self.pA, 0.0)
        joint = pa[:, None] * K  # P(x, v | accept)
        Pv = joint.sum(0)
        total = lam = 0.0
        for v in np.flatnonzero(Pv > 0):
            post = joint[:, v] / Pv[v]
            nz = post > 0
            total += Pv[v] * float(np.sum(post[nz] * np.log2(post[nz] / p[nz])))
            up = self.avail[:, v]
            cond = np.where(up, p / p[up].sum(), 0.0)
            lam += Pv[v] * float(np.sum(post[nz] * np.log2(post[nz] / cond[nz])))
        EL = float((Pv * self.L).sum())
        cost = float((joint * self.w[None, :]).sum())
        out = {
            "E_L": EL, "Lambda": lam, "total_leakage": total, "identity_gap": total - (EL + lam),
            "L_min": self.L_min, "PoD": EL - self.L_min, "mutual_info_given_accept": total - self.L_min,
            "eps_sel": self.eps_sel(K), "worst_adv": self.worst_advantage(K), "exp_cost": cost,
            "n_views": int((Pv > 1e-15).sum()),
        }
        # branch-inference attack: predict "the globally cheapest branch was unavailable"
        cheapest = int(np.argmin(self.w))
        y = ~self.avail[:, cheapest]
        out.update(self._bayes_attack(joint, y, "branch_attack"))
        if target_mask is not None:
            out.update(self._bayes_attack(joint, target_mask, "target_attack"))
        return out

    @staticmethod
    def _bayes_attack(joint, y, name):
        p1 = joint[y].sum(0)
        p0 = joint[~y].sum(0)
        acc = float(np.maximum(p1, p0).sum())
        base = float(max(p1.sum(), p0.sum()))
        return {f"{name}_acc": acc, f"{name}_base": base, f"{name}_lift": acc - base}

    def eps_sel(self, K: np.ndarray) -> float:
        worst = 0.0
        for v in range(len(self.front)):
            col = K[self.avail[:, v], v]
            if col.size == 0:
                continue
            mx, mn = col.max(), col.min()
            if mx <= 0:
                continue
            if mn <= 0:
                return math.inf
            worst = max(worst, math.log(mx / mn))
        return worst

    def worst_advantage(self, K: np.ndarray) -> float:
        """G-DIS: max TV distance between the view distributions of two evidence vectors
        that both satisfy Phi and share a common supportable profile."""
        acc = np.flatnonzero(self.accept)
        keys = np.concatenate([np.round(K[acc], 12), self.avail[acc].astype(float)], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        rows = acc[first]
        Kr, Ar = K[rows], self.avail[rows]
        best = 0.0
        for a in range(len(rows)):
            share = (Ar[a][None, :] & Ar).any(1)
            if not share.any():
                continue
            tv = 0.5 * np.abs(Kr[a][None, :] - Kr[share]).sum(1)
            best = max(best, float(tv.max()))
        return best

    def bop_metrics(self) -> dict:
        """Branch-oblivious proof: single view symbol; leakage equals the floor; proves every branch."""
        return {"E_L": self.L_min, "Lambda": 0.0, "total_leakage": self.L_min, "identity_gap": 0.0,
                "L_min": self.L_min, "PoD": 0.0, "mutual_info_given_accept": 0.0, "eps_sel": 0.0,
                "worst_adv": 0.0, "exp_cost": float(self.w.sum()), "n_views": 1}

    # ------------------------------------------------------------------ signature space (pcgs.py)
    def signature_space(self, target_mask: np.ndarray | None = None):
        """Aggregate states by signature A(x) (Lemma 2). Targets carried along for Bayes attackers:
        'branch' = the globally cheapest branch is unavailable (the attack used in metrics())."""
        from .pcgs import SignatureSpace
        targets = {"branch": (~self.avail[:, int(np.argmin(self.w))]).astype(float)}
        if target_mask is not None:
            targets["target"] = np.asarray(target_mask, dtype=float)
        return SignatureSpace.from_states(self.avail, np.where(self.accept, self.prior.p, 0.0), self.w, targets)

    def expand(self, ss, Q: np.ndarray) -> np.ndarray:
        """State-level kernel K[x, v] = Q[A(x), v] for a signature-level kernel table."""
        K = np.zeros(self.avail.shape)
        rows = np.flatnonzero(self.accept & (self.prior.p > 0))
        codes = self.avail[rows].astype(np.int64) @ (1 << np.arange(self.avail.shape[1], dtype=np.int64))
        order = np.argsort(ss.codes)
        K[rows] = Q[order[np.searchsorted(ss.codes[order], codes)]]
        return K

    def kernel_pcgs(self, overhead: float) -> np.ndarray:
        """Prior-corrected Gibbs selection: minimum-Lambda kernel within (1 + overhead) x the DET cost."""
        ss = self.signature_space()
        return self.expand(ss, ss.compile(overhead).Q)

    def zero_leakage_feasible(self) -> bool:
        """Theorem 1: some branch-revealing kernel has Lambda = 0 iff the exact-cover system is feasible."""
        return self.signature_space().zero_leakage_system()["feasible"]

    def a_max(self) -> int:
        return int(self.avail.sum(1).max())

    def solitary_reachable(self) -> bool:
        cnt = self.avail.sum(1)
        return all(((cnt == 1) & self.avail[:, v]).any() for v in range(len(self.front)))


def blueprint_toy_space(prior: Prior | None = None):
    """A 4-attribute, 4-level toy in the spirit of Sec 21.3 (three-branch policy)."""
    from .claims import synthetic_schema
    from .policy import frontier, parse

    sch = synthetic_schema(4, 4, "toy-4x4")
    pol = parse("(a0 >= 3 AND a1 >= 2) OR (a0 >= 2 AND a2 >= 3 AND a3 >= 1) OR (a1 >= 3 AND a3 >= 2)", sch)
    fr = frontier(pol, sch.n)
    pr = prior or Prior.uniform(sch.dims)
    w = np.array([1.7 * u[0] + 1.1 * u[1] + 0.9 * u[2] + 1.0 * u[3] for u in fr], dtype=float)
    return sch, pol, SelectionSpace(pr, fr, w)
