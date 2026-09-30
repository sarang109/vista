"""Minimum Assurance Negotiation (Sec 5.3) and stochastic MAN-S (Foundation N6, Sec 25).

Deterministic MAN: least-cost set of step-up actions Q with Phi(C u Delta(Q)) = 1 (a covering
problem, R4) - solved exactly by enumeration / ILP.

Stochastic MAN-S: each action a has friction k(a) and completion probability p(a); the policy
becomes a monotone Boolean function of action outcomes (stochastic Boolean function evaluation).
  * exact DP over three-valued states (3^|A|)
  * ratio rules for read-once formulas (AND: k/(1-p), OR: k/p), applied recursively (depth-first)
  * baselines: fixed listed order, blanket strongest-branch
Assurance deficit d(C, Phi) = optimal expected friction; P* = Pr[policy satisfiable by step-up].
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .policy import And, Atom, Node, Or

UNTESTED, OK, FAILED = 0, 1, 2


@dataclass(frozen=True)
class Action:
    name: str
    k: float  # friction cost
    p: float  # completion probability
    delta: tuple = ()  # claims (i, theta) produced on completion


# ------------------------------------------------------------------ action formulas
@dataclass(frozen=True)
class ALeaf:
    a: int


@dataclass(frozen=True)
class AAnd:
    children: tuple


@dataclass(frozen=True)
class AOr:
    children: tuple


@dataclass(frozen=True)
class AConst:
    v: bool


AN = ALeaf | AAnd | AOr | AConst


def simplify(nd: AN) -> AN:
    if isinstance(nd, (ALeaf, AConst)):
        return nd
    kids = [simplify(c) for c in nd.children]
    if isinstance(nd, AAnd):
        if any(isinstance(c, AConst) and not c.v for c in kids):
            return AConst(False)
        kids = [c for c in kids if not isinstance(c, AConst)]
        if not kids:
            return AConst(True)
        return kids[0] if len(kids) == 1 else AAnd(tuple(kids))
    if any(isinstance(c, AConst) and c.v for c in kids):
        return AConst(True)
    kids = [c for c in kids if not isinstance(c, AConst)]
    if not kids:
        return AConst(False)
    return kids[0] if len(kids) == 1 else AOr(tuple(kids))


def action_formula(policy: Node, current, actions: list[Action]) -> AN:
    """Residual policy over action outcomes given currently proven profile `current`."""

    def conv(nd):
        if isinstance(nd, Atom):
            if current[nd.i] >= nd.theta:
                return AConst(True)
            prov = [ALeaf(j) for j, a in enumerate(actions)
                    if any(i == nd.i and t >= nd.theta for i, t in a.delta)]
            if not prov:
                return AConst(False)
            return prov[0] if len(prov) == 1 else AOr(tuple(prov))
        kids = tuple(conv(c) for c in nd.children)
        return AAnd(kids) if isinstance(nd, And) else AOr(kids)

    return simplify(conv(policy))


def leaves(nd: AN) -> list[int]:
    if isinstance(nd, ALeaf):
        return [nd.a]
    if isinstance(nd, AConst):
        return []
    out = []
    for c in nd.children:
        out.extend(leaves(c))
    return out


def is_read_once(nd: AN) -> bool:
    lv = leaves(nd)
    return len(lv) == len(set(lv))


def eval3(nd: AN, s) -> bool | None:
    """Kleene evaluation; exact for monotone formulas (decided iff f(unknown=0)=1 or f(unknown=1)=0)."""
    if isinstance(nd, AConst):
        return nd.v
    if isinstance(nd, ALeaf):
        return None if s[nd.a] == UNTESTED else s[nd.a] == OK
    vals = [eval3(c, s) for c in nd.children]
    if isinstance(nd, AAnd):
        if any(v is False for v in vals):
            return False
        return True if all(v is True for v in vals) else None
    if any(v is True for v in vals):
        return True
    return False if all(v is False for v in vals) else None


def relevant(nd: AN, s) -> list[int]:
    if eval3(nd, s) is not None:
        return []
    if isinstance(nd, ALeaf):
        return [nd.a]
    out = []
    for c in nd.children:
        out.extend(relevant(c, s))
    return out


# ------------------------------------------------------------------ strategies
def evaluate_strategy(nd: AN, actions: list[Action], choose, n: int | None = None) -> dict:
    """Exact expected friction and success probability of an adaptive strategy."""
    n = n if n is not None else len(actions)

    @lru_cache(maxsize=None)
    def V(s):
        v = eval3(nd, s)
        a = choose(s)
        if a is None:
            return 0.0, 1.0 if v is True else 0.0, 0.0
        act = actions[a]
        s1 = s[:a] + (OK,) + s[a + 1:]
        s0 = s[:a] + (FAILED,) + s[a + 1:]
        c1, p1, r1 = V(s1)
        c0, p0, r0 = V(s0)
        return (act.k + act.p * c1 + (1 - act.p) * c0,
                act.p * p1 + (1 - act.p) * p0,
                1 + act.p * r1 + (1 - act.p) * r0)

    c, p, r = V(tuple([UNTESTED] * n))
    return {"exp_friction": c, "success_prob": p, "exp_rounds": r,
            "friction_per_success": c / p if p > 0 else math.inf}


def optimal_dp(nd: AN, actions: list[Action], max_actions: int = 14):
    """Exact optimal adaptive strategy (DP over 3^|A| states)."""
    rel = sorted(set(leaves(nd)))
    if len(rel) > max_actions:
        raise OverflowError("too many actions for exact DP")
    n = len(actions)

    @lru_cache(maxsize=None)
    def best(s):
        if eval3(nd, s) is not None:
            return 0.0, None
        bc, ba = math.inf, None
        for a in relevant(nd, s):
            act = actions[a]
            c1, _ = best(s[:a] + (OK,) + s[a + 1:])
            c0, _ = best(s[:a] + (FAILED,) + s[a + 1:])
            c = act.k + act.p * c1 + (1 - act.p) * c0
            if c < bc - 1e-12 or (abs(c - bc) <= 1e-12 and (ba is None or a < ba)):
                bc, ba = c, a
        return bc, ba

    return lambda s: best(s)[1]


def ratio_strategy(nd: AN, actions: list[Action]):
    """Depth-first recursive ratio rule (Prop 6.1/6.2). Exact for read-once depth <= 2."""
    order = {}

    def summarize(x):
        if isinstance(x, ALeaf):
            a = actions[x.a]
            return a.k, a.p
        if isinstance(x, AConst):
            return 0.0, 1.0 if x.v else 0.0
        stats = [(summarize(c), c) for c in x.children]
        if isinstance(x, AAnd):
            stats.sort(key=lambda t: t[0][0] / max(1 - t[0][1], 1e-12))
            c, run = 0.0, 1.0
            for (ci, pi), _ in stats:
                c += run * ci
                run *= pi
            p = run
        else:
            stats.sort(key=lambda t: t[0][0] / max(t[0][1], 1e-12))
            c, run = 0.0, 1.0
            for (ci, pi), _ in stats:
                c += run * ci
                run *= 1 - pi
            p = 1 - run
        order[id(x)] = [ch for _, ch in stats]
        return c, p

    summarize(nd)

    def dfs(x, s):
        if eval3(x, s) is not None:
            return None
        if isinstance(x, ALeaf):
            return x.a if s[x.a] == UNTESTED else None
        for ch in order[id(x)]:
            r = dfs(ch, s)
            if r is not None:
                return r
        return None

    return lambda s: dfs(nd, s)


def fixed_order_strategy(nd: AN, listed: list[int]):
    def choose(s):
        rel = set(relevant(nd, s))
        for a in listed:
            if a in rel and s[a] == UNTESTED:
                return a
        return None

    return choose


def greedy_strategy(nd: AN, actions: list[Action]):
    """Heuristic for non-read-once residual policies: cheapest k/p among relevant actions."""
    def choose(s):
        rel = relevant(nd, s)
        if not rel:
            return None
        return min(rel, key=lambda a: (actions[a].k / actions[a].p, a))

    return choose


def blanket_branch(nd: AN, actions: list[Action], strength) -> AN:
    """Blanket baseline: only the top-level OR branch with the strongest controls."""
    if not isinstance(nd, AOr):
        return nd
    return max(nd.children, key=lambda c: (sum(strength(actions[a]) for a in leaves(c)), -len(leaves(c))))


def simulate(nd: AN, actions: list[Action], choose, rng, trials: int) -> dict:
    n = len(actions)
    costs, succ = np.zeros(trials), np.zeros(trials)
    for t in range(trials):
        s = [UNTESTED] * n
        c = 0.0
        while True:
            a = choose(tuple(s))
            if a is None:
                break
            c += actions[a].k
            s[a] = OK if rng.random() < actions[a].p else FAILED
        costs[t] = c
        succ[t] = eval3(nd, tuple(s)) is True
    return {"mc_exp_friction": float(costs.mean()), "mc_success_prob": float(succ.mean())}


# ------------------------------------------------------------------ deterministic MAN (v1.0)
def deterministic_man(policy: Node, current, actions: list[Action], budget: float = math.inf):
    """Least-cost subset of actions (all assumed to succeed). Exact by enumeration in cost order."""
    from .policy import evaluate
    if evaluate(policy, tuple(current)):
        return [], 0.0
    best, bc = None, math.inf
    n = len(actions)
    for r in range(1, n + 1):
        for Q in itertools.combinations(range(n), r):
            c = sum(actions[a].k for a in Q)
            if c >= bc or c > budget:
                continue
            prof = list(current)
            for a in Q:
                for i, t in actions[a].delta:
                    prof[i] = max(prof[i], t)
            if evaluate(policy, tuple(prof)):
                best, bc = list(Q), c
    return best, bc


# ------------------------------------------------------------------ decision rule
def plan(policy: Node, current, actions: list[Action], budget: float, p_min: float) -> dict:
    """MAN-S (Sec 25.2): residual policy -> strategy -> deficit d and P* -> STEP_UP / HOLD / REJECT."""
    nd = action_formula(policy, current, actions)
    if isinstance(nd, AConst):
        return {"decision": "ACCEPT" if nd.v else "REJECT", "deficit": 0.0 if nd.v else math.inf,
                "p_star": 1.0 if nd.v else 0.0, "first_action": None, "method": "const"}
    rel = set(leaves(nd))
    if is_read_once(nd):
        choose, method = ratio_strategy(nd, actions), "ratio"
        if len(rel) <= 12:
            choose, method = optimal_dp(nd, actions), "dp"
    elif len(rel) <= 12:
        choose, method = optimal_dp(nd, actions), "dp"
    else:
        choose, method = greedy_strategy(nd, actions), "greedy"
    ev = evaluate_strategy(nd, actions, choose)
    d, ps = ev["exp_friction"], ev["success_prob"]
    if d <= budget and ps >= p_min:
        dec = "STEP_UP"
    elif d <= budget and ps > 0:
        dec = "HOLD"
    else:
        dec = "REJECT"
    first = choose(tuple([UNTESTED] * len(actions)))
    return {"decision": dec, "deficit": d, "p_star": ps, "first_action": first, "method": method,
            "exp_rounds": ev["exp_rounds"]}


def random_readonce_actions(rng, n_actions: int, depth: int, k_range=(0.5, 6.0), p_range=(0.4, 0.98)):
    actions = [Action(f"act{j}", float(rng.uniform(*k_range)), float(rng.uniform(*p_range)))
               for j in range(n_actions)]
    pool = list(rng.permutation(n_actions))

    def build(d, kind):
        if d == 0 or len(pool) <= 1:
            return ALeaf(int(pool.pop())) if pool else AConst(False)
        b = int(rng.integers(2, 4))
        kids = []
        for _ in range(b):
            if not pool:
                break
            kids.append(build(d - 1, "and" if kind == "or" else "or"))
        if len(kids) == 1:
            return kids[0]
        return AOr(tuple(kids)) if kind == "or" else AAnd(tuple(kids))

    return actions, simplify(build(depth, "or"))


def formula_depth(nd: AN) -> int:
    if isinstance(nd, (ALeaf, AConst)):
        return 0
    return 1 + max(formula_depth(c) for c in nd.children)
