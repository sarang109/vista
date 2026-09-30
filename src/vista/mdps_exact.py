"""Exact Minimum-Disclosure Proof Selection (Sec 5.2, N1 Sec 20).

    S* = argmin_{l <= x, Phi(l)=1} cost(l)

Solvers
  solve_frontier   - min over compiled frontier M(Phi) (Lemma 1.1; exact for isotone costs)
  solve_readonce   - linear DP for read-once AND/OR trees (AND = sum, OR = min), additive cost
  solve_ilp        - 0-1 ILP (HiGHS via scipy.milp) for any monotone formula, additive cost
  solve_bruteforce - enumerate every profile l <= x (ground truth on small instances)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .cost import AdditiveCost
from .policy import And, Atom, Node, Or, evaluate


@dataclass
class MDPSResult:
    profile: tuple | None
    cost: float
    method: str

    @property
    def feasible(self) -> bool:
        return self.profile is not None


INFEASIBLE = lambda m: MDPSResult(None, math.inf, m)  # noqa: E731


def solve_frontier(front: np.ndarray, x, cost_fn, costs: np.ndarray | None = None) -> MDPSResult:
    """argmin cost over u in M(Phi) with u <= x. Ties broken by frontier order (public, deterministic)."""
    avail = (front <= np.asarray(x)).all(1)
    if not avail.any():
        return INFEASIBLE("frontier")
    idx = np.flatnonzero(avail)
    c = costs[idx] if costs is not None else np.array([cost_fn(front[k]) for k in idx])
    j = idx[int(np.argmin(c))]
    return MDPSResult(tuple(int(v) for v in front[j]), float(c.min()), "frontier")


def solve_readonce(node: Node, x, cost: AdditiveCost) -> MDPSResult:
    def rec(nd):
        if isinstance(nd, Atom):
            if x[nd.i] < nd.theta:
                return math.inf, {}
            # cheapest supported threshold >= theta (equals theta when costs are isotone)
            best_t = min(range(nd.theta, x[nd.i] + 1), key=lambda t: cost.claim(nd.i, t))
            return cost.claim(nd.i, best_t), {nd.i: best_t}
        subs = [rec(c) for c in nd.children]
        if isinstance(nd, And):
            if any(math.isinf(s[0]) for s in subs):
                return math.inf, {}
            d = {}
            for _, dd in subs:
                d.update(dd)
            return sum(s[0] for s in subs), d
        return min(subs, key=lambda s: s[0])

    c, d = rec(node)
    if math.isinf(c):
        return INFEASIBLE("readonce_dp")
    prof = [0] * len(x)
    for i, t in d.items():
        prof[i] = t
    return MDPSResult(tuple(prof), float(cost(prof)), "readonce_dp")


def solve_ilp(node: Node, x, cost: AdditiveCost, time_limit: float = 30.0) -> MDPSResult:
    n = len(x)
    if not evaluate(node, tuple(x)):
        return INFEASIBLE("ilp")
    claim_vars = [(i, t) for i in range(n) for t in range(1, int(x[i]) + 1)]
    cidx = {c: k for k, c in enumerate(claim_vars)}
    nodes: list[Node] = []

    def collect(nd):
        nodes.append(nd)
        if not isinstance(nd, Atom):
            for c in nd.children:
                collect(c)

    collect(node)
    nidx = {id(nd): len(claim_vars) + k for k, nd in enumerate(nodes)}
    nv = len(claim_vars) + len(nodes)
    rows, cols, vals, ub = [], [], [], []
    r = 0

    def add(entries, u):
        nonlocal r
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        ub.append(u)
        r += 1

    for nd in nodes:
        y = nidx[id(nd)]
        if isinstance(nd, Atom):
            sup = [cidx[(nd.i, t)] for t in range(nd.theta, int(x[nd.i]) + 1)]
            add([(y, 1.0)] + [(s, -1.0) for s in sup], 0.0)
        elif isinstance(nd, And):
            for c in nd.children:
                add([(y, 1.0), (nidx[id(c)], -1.0)], 0.0)
        else:
            add([(y, 1.0)] + [(nidx[id(c)], -1.0) for c in nd.children], 0.0)
    add([(nidx[id(node)], -1.0)], -1.0)  # root must be satisfied
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    obj = np.zeros(nv)
    for (i, t), k in cidx.items():
        obj[k] = cost.claim(i, t)
    res = milp(obj, constraints=LinearConstraint(A, -np.inf, np.array(ub)),
               integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return INFEASIBLE("ilp")
    prof = [0] * n
    for (i, t), k in cidx.items():
        if res.x[k] > 0.5:
            prof[i] = max(prof[i], t)
    return MDPSResult(tuple(prof), float(cost(prof)), "ilp")


def solve_bruteforce(node: Node, x, cost_fn) -> MDPSResult:
    best, bc = None, math.inf
    for prof in itertools.product(*[range(int(v) + 1) for v in x]):
        if evaluate(node, prof):
            c = cost_fn(prof)
            if c < bc - 1e-12:
                best, bc = prof, c
    if best is None:
        return INFEASIBLE("bruteforce")
    return MDPSResult(tuple(best), float(bc), "bruteforce")
