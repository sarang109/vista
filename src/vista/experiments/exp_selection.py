"""E11 selection leakage vs policy shape and E12 leakage-cost Pareto curve (Foundation N2)."""
from __future__ import annotations

import math
import os

import numpy as np

from .. import policy as pol
from ..claims import synthetic_schema
from ..prior import Prior
from ..selection import SelectionSpace, blueprint_toy_space
from ..util import write_csv


def run_e11(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 111)
    rows = []
    sch = synthetic_schema(4, 4)
    for prior_kind in ("uniform", "ferro", "anti"):
        for m_terms in (2, 3, 4, 5, 6):
            for spread in (0.5, 1.0, 2.0):
                for rep in range(cfg.get("e11_reps", 8)):
                    node = pol.random_dnf(sch, m_terms, int(rng.integers(2, 4)), rng)
                    fr = pol.frontier(node, sch.n)
                    if len(fr) < 2:
                        continue
                    pr = Prior.uniform(sch.dims) if prior_kind == "uniform" else Prior.random_ising(sch.dims, rng, prior_kind)
                    w = fr.sum(1).astype(float) + rng.normal(0, spread, len(fr))
                    w = w - w.min() + 1.0
                    sp = SelectionSpace(pr, fr, w)
                    if sp.pA <= 0:
                        continue
                    target = sp.prior.states[:, 1] < 2  # a designated undisclosed sensitive predicate
                    md = sp.metrics(sp.kernel_det(), target)
                    rows.append({"prior": prior_kind, "frontier_size": len(fr), "cost_spread": spread, "rep": rep,
                                 "Lambda_bits": md["Lambda"], "E_L_bits": md["E_L"], "L_min_bits": md["L_min"],
                                 "PoD_bits": md["PoD"], "eps_sel": md["eps_sel"], "worst_adv": md["worst_adv"],
                                 "branch_attack_acc": md["branch_attack_acc"],
                                 "branch_attack_base": md["branch_attack_base"],
                                 "target_attack_lift": md["target_attack_lift"], "a_max": sp.a_max()})
    return [write_csv(os.path.join(out, "e11_selection_leakage.csv"), rows)]


def run_e12(cfg, out):
    rows = []
    for prior_name in ("uniform", "ferro"):
        rng = np.random.default_rng(cfg.get("seed", 1) + 12)
        pr = None if prior_name == "uniform" else Prior.random_ising((4, 4, 4, 4), rng, "ferro", 1.5)
        _, _, sp = blueprint_toy_space(pr)
        base_cost = sp.metrics(sp.kernel_det())["exp_cost"]

        def rec(mode, param, K=None, m=None):
            m = m or sp.metrics(K)
            rows.append({"prior": prior_name, "mode": mode, "param": param, "Lambda_bits": m["Lambda"],
                         "total_leakage_bits": m["total_leakage"], "E_L_bits": m["E_L"], "L_min_bits": m["L_min"],
                         "PoD_bits": m["PoD"], "eps_sel": m["eps_sel"], "worst_adv": m["worst_adv"],
                         "exp_cost": m["exp_cost"], "cost_overhead": m["exp_cost"] / base_cost - 1,
                         "mutual_info_bits": m["mutual_info_given_accept"]})

        rec("DET", "", sp.kernel_det())
        for lam in (0, 0.25, 0.5, 1, 2, 4, 8):
            rec("EXP", lam, sp.kernel_exp(lam))
        eps_min = sp.min_feasible_eps(iters=25)
        for eps in sorted({eps_min + 1e-6, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0}):
            if eps < eps_min:
                continue
            K = sp.kernel_lp(eps)
            if K is not None:
                rec("LP", round(eps, 4), K)
        rec("BOP", "", m=sp.bop_metrics())
    return [write_csv(os.path.join(out, "e12_leakage_cost_pareto.csv"), rows)]
