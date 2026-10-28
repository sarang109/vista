"""Foundation N5: group-cost (fixed-charge) MDPS (Sec 24).

cost(S) = sum_{groups g touched} F_g + sum_{c in S} m(c)
Instances are CNF policies: each clause must be covered by at least one claim.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class GroupInstance:
    n_clauses: int
    F: list  # fixed charge per group
    claims: list = field(default_factory=list)  # (group, marginal, frozenset(clauses covered))

    def cost(self, chosen) -> float:
        groups = {self.claims[c][0] for c in chosen}
        return sum(self.F[g] for g in groups) + sum(self.claims[c][1] for c in chosen)

    def covers(self, chosen) -> bool:
        cov = set()
        for c in chosen:
            cov |= self.claims[c][2]
        return len(cov) == self.n_clauses


def exact_ilp(inst: GroupInstance):
    G, C = len(inst.F), len(inst.claims)
    nv = G + C
    obj = np.concatenate([np.asarray(inst.F, float), [c[1] for c in inst.claims]])
    rows = []
    lb, ub = [], []
    for k, (g, _, _) in enumerate(inst.claims):  # s_c - open_g <= 0
        r = np.zeros(nv); r[G + k] = 1; r[g] = -1; rows.append(r); lb.append(-np.inf); ub.append(0)
    for j in range(inst.n_clauses):
        r = np.zeros(nv)
        for k, (_, _, cov) in enumerate(inst.claims):
            if j in cov:
                r[G + k] = 1
        rows.append(r); lb.append(1); ub.append(np.inf)
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1))
    if res.x is None:
        return None, math.inf
    chosen = [k for k in range(C) if res.x[G + k] > 0.5]
    return chosen, inst.cost(chosen)


def exact_enum(inst: GroupInstance):
    """Enumerate opened group subsets; cheapest claim per clause (exact when each claim covers one clause)."""
    best, bc = None, math.inf
    G = len(inst.F)
    for r in range(1, G + 1):
        for opened in itertools.combinations(range(G), r):
            op = set(opened)
            chosen = []
            ok = True
            for j in range(inst.n_clauses):
                cands = [k for k, (g, m, cov) in enumerate(inst.claims) if g in op and j in cov]
                if not cands:
                    ok = False
                    break
                chosen.append(min(cands, key=lambda k: inst.claims[k][1]))
            if ok:
                c = inst.cost(set(chosen))
                if c < bc:
                    best, bc = sorted(set(chosen)), c
    return best, bc


def myopic_greedy(inst: GroupInstance):
    opened, chosen, unc = set(), [], set(range(inst.n_clauses))
    while unc:
        best = None
        for k, (g, m, cov) in enumerate(inst.claims):
            new = len(cov & unc)
            if k in chosen or new == 0:
                continue
            marg = m + (0 if g in opened else inst.F[g])
            key = (marg, -new, k)
            if best is None or key < best:
                best = key
        if best is None:
            return None, math.inf
        k = best[2]
        chosen.append(k); opened.add(inst.claims[k][0]); unc -= inst.claims[k][2]
    return chosen, inst.cost(chosen)


def density_greedy(inst: GroupInstance):
    """Algorithm DG: repeatedly add the star (group g, claim set T) minimising
    (F_g [g unopened] + sum m(c)) / newly covered clauses. Best star per group by prefix scan."""
    opened, chosen, unc = set(), set(), set(range(inst.n_clauses))
    by_group: dict = {}
    for k, (g, m, cov) in enumerate(inst.claims):
        by_group.setdefault(g, []).append(k)
    while unc:
        best = None
        for g, ks in by_group.items():
            cands = [(inst.claims[k][1] / max(len(inst.claims[k][2] & unc), 1), k) for k in ks
                     if k not in chosen and inst.claims[k][2] & unc]
            cands.sort()
            fixed = 0.0 if g in opened else inst.F[g]
            tot, cov = fixed, set()
            T = []
            for _, k in cands:
                tot += inst.claims[k][1]
                cov |= inst.claims[k][2] & unc
                T.append(k)
                ratio = tot / len(cov)
                if best is None or ratio < best[0] - 1e-12:
                    best = (ratio, g, list(T))
        if best is None:
            return None, math.inf
        _, g, T = best
        opened.add(g)
        for k in T:
            chosen.add(k)
            unc -= inst.claims[k][2]
    return sorted(chosen), inst.cost(chosen)


def adversarial_instance(k: int) -> GroupInstance:
    """Prop 5.1 family: Phi = AND_i (x_i OR y_i); each x_i alone (F=1), all y_i share H (F=2)."""
    F = [1.0] * k + [2.0]
    claims = [(i, 0.0, frozenset([i])) for i in range(k)] + [(k, 0.0, frozenset([i])) for i in range(k)]
    return GroupInstance(k, F, claims)


def random_instance(rng, m_clauses: int, n_groups: int, density: float = 0.5,
                    F_range=(0.5, 4.0), m_range=(0.05, 1.0)) -> GroupInstance:
    """One claim per clause per group at most (the assumption of Prop 5.2)."""
    F = list(rng.uniform(*F_range, size=n_groups))
    claims = []
    for j in range(m_clauses):
        gs = [g for g in range(n_groups) if rng.random() < density]
        if not gs:
            gs = [int(rng.integers(n_groups))]
        for g in gs:
            claims.append((g, float(rng.uniform(*m_range)), frozenset([j])))
    return GroupInstance(m_clauses, F, claims)


def harmonic(m: int) -> float:
    return sum(1.0 / i for i in range(1, m + 1))


def layout_cost(prof, names, layout: dict, F: dict, per_claim: float) -> float:
    """Fixed-charge proof cost of disclosing a profile under a sender's credential layout:
    F_g for every credential group touched plus per_claim for every disclosed claim (N5 cost folded
    into the disclosure governor). names[i] is the attribute name of coordinate i; layout maps a name to
    its credential group."""
    groups = {layout[names[i]] for i, t in enumerate(prof) if t > 0}
    return float(sum(F.get(g, 1.0) for g in groups) + per_claim * sum(1 for t in prof if t > 0))
