"""E1 MDPS optimality / heuristic gap and E2 MDPS latency with cached policy compilation."""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .. import policy as pol
from ..claims import synthetic_schema
from ..cost import AdditiveCost
from ..mdps_exact import solve_frontier, solve_ilp, solve_readonce
from ..mdps_greedy import greedy
from ..util import percentiles, write_csv


def _instance(family, sch, rng):
    if family == "dnf":
        return pol.random_dnf(sch, int(rng.integers(3, 11)), int(rng.integers(2, 6)), rng)
    if family == "cnf":
        return pol.random_cnf(sch, int(rng.integers(3, 21)), int(rng.integers(2, 5)), rng)
    if family == "readonce":
        return pol.random_tree(sch, int(rng.integers(2, 5)), 3, rng, read_once=True)
    return pol.random_tree(sch, int(rng.integers(2, 4)), 3, rng)


def run_e1(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 11)
    rows = []
    reps = cfg.get("e1_reps", 20)
    for n_attr in cfg.get("e1_attrs", [5, 10, 20, 40, 50]):
        sch = synthetic_schema(n_attr, 5)
        for family in ("dnf", "cnf", "readonce", "tree"):
            for r in range(reps):
                try:
                    node = _instance(family, sch, rng)
                except ValueError:
                    continue
                x = tuple(int(v) for v in rng.integers(1, 5, n_attr))
                if not pol.evaluate(node, x):
                    x = tuple(4 for _ in range(n_attr))
                w = AdditiveCost.random_isotone(sch.dims, rng)
                base = {"n_attrs": n_attr, "n_claims": n_attr * 4, "family": family, "rep": r,
                        "policy_atoms": len(pol.atoms(node)), "policy_depth": pol.depth(node)}
                t = time.perf_counter()
                ex = solve_ilp(node, x, w)
                t_ilp = (time.perf_counter() - t) * 1e3
                rows.append(dict(base, solver="ilp", cost=ex.cost, ms=t_ilp, gap=0.0, frontier_size=""))
                try:
                    t = time.perf_counter()
                    fr = pol.frontier(node, n_attr, max_size=5_000)
                    t_c = (time.perf_counter() - t) * 1e3
                    t = time.perf_counter()
                    f = solve_frontier(fr, x, w)
                    t_q = (time.perf_counter() - t) * 1e3
                    rows.append(dict(base, solver="frontier_cached", cost=f.cost, ms=t_q, gap=f.cost / ex.cost - 1,
                                     frontier_size=len(fr), compile_ms=t_c))
                except OverflowError:
                    rows.append(dict(base, solver="frontier_cached", cost=math.nan, ms=math.nan, gap=math.nan,
                                     frontier_size=">5000"))
                if pol.is_read_once(node):
                    t = time.perf_counter()
                    d = solve_readonce(node, x, w)
                    rows.append(dict(base, solver="readonce_dp", cost=d.cost, ms=(time.perf_counter() - t) * 1e3,
                                     gap=d.cost / ex.cost - 1, frontier_size=""))
                try:
                    t = time.perf_counter()
                    g = greedy(node, x, w, max_clauses=5000)
                    rows.append(dict(base, solver="greedy", cost=g.cost, ms=(time.perf_counter() - t) * 1e3,
                                     gap=g.cost / ex.cost - 1, frontier_size=""))
                except OverflowError:
                    pass
    p = write_csv(os.path.join(out, "e1_mdps_optimality.csv"), rows)
    summ = []
    import itertools
    keyf = lambda r: (r["n_attrs"], r["family"], r["solver"])  # noqa: E731
    for k, grp in itertools.groupby(sorted(rows, key=keyf), key=keyf):
        grp = list(grp)
        gaps = [g["gap"] for g in grp if isinstance(g["gap"], float) and not math.isnan(g["gap"])]
        ms = [g["ms"] for g in grp if isinstance(g["ms"], float) and not math.isnan(g["ms"])]
        pc = percentiles(ms)
        summ.append({"n_claims": k[0] * 4, "family": k[1], "solver": k[2], "instances": len(grp),
                     "optimal_rate": float(np.mean([g <= 1e-9 for g in gaps])) if gaps else math.nan,
                     "mean_gap": float(np.mean(gaps)) if gaps else math.nan,
                     "max_gap": float(np.max(gaps)) if gaps else math.nan,
                     "p50_ms": pc["p50"], "p99_ms": pc["p99"]})
    return [p, write_csv(os.path.join(out, "e1_mdps_summary.csv"), summ)]


def run_e2(cfg, out, world):
    """Per-payment MDPS latency on the interbank policy library with policy cache on/off."""
    from ..sen import normalize
    from ..simulator import workload as wl
    rng = np.random.default_rng(cfg.get("seed", 1) + 22)
    n = cfg.get("e2_payments", 5000)
    pay = wl.generate_payments(world.pop, world.banks, n, 7, rng)
    rows = []
    for cache in (True, False):
        snd_cache = {}
        lat = []
        for j in range(n):
            s, r = int(pay["sender"][j]), int(pay["receiver"][j])
            p = world.payment(pay, j)
            node = world.policy_for(r, p.amount_minor)
            snd = world.sender(s, "DET", cache=cache)
            b = world.banks[s]
            ev = world.evidence(pay, j, "pid", rng)
            x = normalize(ev, "pid", int(pay["ts"][j]) + 1, world.arche["archetypes"][b.archetype]["profile"]).profile
            t = time.perf_counter()
            fr, w, _ = snd.cache.get(node)
            snd.select(x, fr, w)
            lat.append((time.perf_counter() - t) * 1e3)
        pc = percentiles(lat)
        rows.append(dict({"cache": cache, "payments": n}, **{f"{k}_ms": v for k, v in pc.items() if k != "n"}))
    return [write_csv(os.path.join(out, "e2_mdps_latency.csv"), rows)]
