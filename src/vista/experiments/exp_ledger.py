"""E14 probing attack, E15 ledger utility with naive vs OCGD grids, E16 cumulative leakage (N3)."""
from __future__ import annotations

import math
import os

import numpy as np

from .. import ledger as led
from .. import policy as pol
from ..claims import synthetic_schema
from ..policy import frontier
from ..prior import Prior
from ..util import write_csv


def run_e14(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 14)
    rows = []
    D = 128
    for rho in (None, 1, 4, 8, 16, 32):
        for strat in ("binary", "random", "scan"):
            res = [led.probe(D, int(x), strat, rho, rng) for x in rng.integers(0, D, cfg.get("e14_payers", 500))]
            rows.append({"rho": rho or "no_ledger", "strategy": strat, "payers": len(res),
                         "exact_recovery_rate": float(np.mean([r["exact"] for r in res])),
                         "mean_bits_learned": float(np.mean([r["bits_learned"] for r in res])),
                         "max_bits_learned": float(np.max([r["bits_learned"] for r in res])),
                         "bound_bits": math.log2(D / (rho or 1)),
                         "min_final_width": int(np.min([r["final_width"] for r in res])),
                         "mean_queries": float(np.mean([r["queries"] for r in res])),
                         "all_consistent": all(r["consistent"] for r in res)})
    return [write_csv(os.path.join(out, "e14_probing_attack.csv"), rows)]


def run_e15(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 15)
    D = 128
    prior = np.full(D, 1 / D)
    S = led.survival(prior)
    rows = []
    scenarios = {"spread": [32, 64, 96], "crowded": [60, 64, 68, 72], "realistic_mix": [7, 30, 60, 64, 90, 120]}
    for name, ths in scenarios.items():
        mu = [1.0] * len(ths)
        grids = {"none": sorted(ths), "naive_3pt": [32, 64, 96]}
        for k in (1, 2, 3):
            grids[f"ocgd_k{k}"] = led.ocgd(ths, mu, k, S)[0]
        for gname, grid in grids.items():
            for rho in (1, 8, 16):
                fr_round = fr_ledger = tot = 0
                for _ in range(cfg.get("e15_payers", 1500)):
                    x = int(rng.integers(0, D))
                    L = led.IntervalLedger(0, D - 1, rho)
                    for _ in range(12):
                        th = int(rng.choice(ths))
                        g = th if gname == "none" else led.round_up(th, grid)
                        if x < th:
                            continue
                        tot += 1
                        if g is None or x < g:
                            fr_round += 1  # upward rounding refused a truthful payer
                            continue
                        if L.answer(x, g) != led.TRUE:
                            fr_ledger += 1
                rows.append({"thresholds": name, "grid": gname, "grid_points": " ".join(map(str, grid)), "rho": rho,
                             "false_reject_rate": (fr_round + fr_ledger) / max(tot, 1),
                             "fr_from_rounding": fr_round / max(tot, 1), "fr_from_ledger": fr_ledger / max(tot, 1),
                             "grid_loss_mass": led.grid_loss(ths, mu, grid, S) / len(ths) if gname != "none" else 0.0,
                             "max_bits_per_attr": math.log2(len(grid) + 1)})
    return [write_csv(os.path.join(out, "e15_ledger_utility.csv"), rows)]


def run_e16(cfg, out):
    """Cumulative surprisal over T repeated payments: history-blind vs sticky (history-aware) MDPS."""
    rng = np.random.default_rng(cfg.get("seed", 1) + 16)
    sch = synthetic_schema(4, 5)
    pr = Prior.random_ising(sch.dims, rng, "ferro", 1.5)
    policies = [pol.random_dnf(sch, int(rng.integers(2, 5)), 2, rng) for _ in range(6)]
    fronts = [frontier(p, sch.n) for p in policies]
    rows = []
    T = cfg.get("e16_T", 40)
    payers = cfg.get("e16_payers", 300)
    xs = pr.sample(rng, payers)
    seqs = [np.random.default_rng(10_000 + k).integers(len(fronts), size=T) for k in range(payers)]
    for strategy in ("disclose_all", "history_blind", "sticky"):
        cum = np.zeros(T)
        cnt = np.zeros(T)
        for pi, x in enumerate(xs):
            K = np.zeros(sch.n, dtype=int)
            for t in range(T):
                fr = fronts[int(seqs[pi][t])]  # identical policy sequence for every strategy
                avail = np.flatnonzero((fr <= x).all(1))
                if strategy == "disclose_all":
                    K = np.array(x)
                elif len(avail):
                    if strategy == "history_blind":
                        costs = [pr.surprisal(fr[k]) for k in avail]
                    else:
                        base = pr.surprisal(K)
                        costs = [pr.surprisal(np.maximum(K, fr[k])) - base + 1e-3 * pr.surprisal(fr[k]) for k in avail]
                    K = np.maximum(K, fr[avail[int(np.argmin(costs))]])
                cum[t] += pr.surprisal(K)
                cnt[t] += 1
        for t in range(T):
            rows.append({"strategy": strategy, "payment_index": t + 1, "mean_cumulative_bits": cum[t] / cnt[t]})
    return [write_csv(os.path.join(out, "e16_cumulative_leakage.csv"), rows)]
