"""Greedy MDPS heuristics (Sec 5.2, B5).

greedy_cnf is the classical weighted set-cover greedy on the CNF form of the policy
(price = marginal cost / newly satisfied clauses), which carries the H_m guarantee for
additive costs (Prop 1.2). Subsumption is respected: raising attribute i from l_i to t
costs w(i,t) - w(i,l_i).
"""
from __future__ import annotations

import math

from .cost import AdditiveCost
from .mdps_exact import INFEASIBLE, MDPSResult
from .policy import Node, to_cnf


def greedy_cnf(clauses, x, cost: AdditiveCost) -> MDPSResult:
    n = len(x)
    prof = [0] * n
    unc = set(range(len(clauses)))
    # clause j satisfied by (i, t) iff some atom (i, th) in clause has th <= t
    need = [dict() for _ in clauses]
    for j, cl in enumerate(clauses):
        for a in cl:
            need[j][a.i] = min(need[j].get(a.i, math.inf), a.theta)
    unc = {j for j in unc if not any(prof[i] >= th for i, th in need[j].items())}
    while unc:
        best = None
        for i in range(n):
            base = cost.claim(i, prof[i]) if prof[i] > 0 else 0.0
            for t in range(prof[i] + 1, int(x[i]) + 1):
                cov = sum(1 for j in unc if need[j].get(i, math.inf) <= t)
                if cov == 0:
                    continue
                marg = max(cost.claim(i, t) - base, 0.0)
                key = (marg / cov, -cov, i, t)
                if best is None or key < best[0]:
                    best = (key, i, t)
        if best is None:
            return INFEASIBLE("greedy")
        _, i, t = best
        prof[i] = t
        unc = {j for j in unc if not need[j].get(i, math.inf) <= t}
    return MDPSResult(tuple(prof), float(cost(prof)), "greedy")


def greedy(node: Node, x, cost: AdditiveCost, max_clauses: int = 5000) -> MDPSResult:
    return greedy_cnf(to_cnf(node, max_clauses), x, cost)
