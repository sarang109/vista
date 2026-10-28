"""E23 (Theorem 1: is zero selection leakage ever possible?) and E24 (Theorem 2: how far is randomised
selection from the optimal kernel at equal cost?), with go/no-go gate 1.

Instances: the v1 E11 policies (uniform, ferromagnetic and anti-ferromagnetic priors), sparse-support priors
(the only setting where Theorem 1 allows Lambda = 0), skewed-marginal priors, and the receiver policy library
under the world's population prior. The real-signal prior instances are in exp_realprior (E26).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .. import policy as pol
from ..claims import synthetic_schema
from ..pcgs import lp_at_budget
from ..prior import Prior
from ..selection import SelectionSpace, blueprint_toy_space
from ..util import write_csv
from .exp_selection import e11_instances

OVERHEADS = (0.01, 0.0235, 0.05, 0.10)


def floor_row(sp: SelectionSpace, ss=None) -> dict:
    """Theorem 1 feasibility test against the convex solver's minimum Lambda for one instance."""
    ss = ss or sp.signature_space()
    sysr = ss.zero_leakage_system()
    fl = ss.lambda_floor()
    det = ss.lam(ss.kernel_det())
    row = {"m": ss.m, "signatures": ss.n_signatures, "zero_leakage_feasible": sysr["feasible"],
           "system_residual": sysr["residual"], "two_branch_obstruction": ss.two_branch_obstruction() is not None,
           "Lambda_DET_bits": det, "Lambda_floor_bits": fl["Lambda"], "floor_lower_bound_bits": fl["lower_bound"],
           "floor_gap_bits": fl["gap_bits"], "floor_over_DET": fl["Lambda"] / det if det > 1e-12 else math.nan}
    # agreement: feasible <=> the convex optimum is 0 (certified > 0 when infeasible)
    row["solver_agrees"] = (fl["Lambda"] < 1e-6) if sysr["feasible"] else (fl["lower_bound"] > 1e-9)
    if sysr["feasible"]:
        row["zero_kernel_Lambda_bits"] = ss.lam(ss.zero_leakage_kernel(sysr["w"]))
    return row


def skewed_instances(cfg, field_sd: float, count: int, seed_off: int):
    rng = np.random.default_rng(cfg.get("seed", 1) + seed_off)
    sch = synthetic_schema(4, 4)
    made = 0
    while made < count:
        node = pol.random_dnf(sch, int(rng.integers(2, 7)), int(rng.integers(2, 4)), rng)
        fr = pol.frontier(node, sch.n)
        if len(fr) < 2:
            continue
        pr = Prior.random_ising(sch.dims, rng, "ferro", 2.0, field_sd)
        w = fr.sum(1).astype(float) + rng.normal(0, 1.0, len(fr))
        sp = SelectionSpace(pr, fr, w - w.min() + 1.0)
        if sp.pA <= 0:
            continue
        made += 1
        yield {"family": f"skewed_sd{field_sd}", "frontier_size": len(fr)}, sp


def sparse_instances(cfg, count: int):
    """Priors whose support misses a random share of the states (full-support priors never allow Lambda = 0)."""
    rng = np.random.default_rng(cfg.get("seed", 1) + 230)
    sch = synthetic_schema(3, 4)
    made = 0
    while made < count:
        node = pol.random_dnf(sch, int(rng.integers(2, 5)), int(rng.integers(1, 3)), rng)
        fr = pol.frontier(node, sch.n)
        if len(fr) < 2:
            continue
        pr = Prior.random_ising(sch.dims, rng, "mixed")
        drop = rng.random(len(pr.p)) < rng.choice([0.5, 0.8, 0.95])
        p = np.where(drop, 0.0, pr.p)
        if p.sum() <= 0:
            continue
        pr.p = p / p.sum()
        w = fr.sum(1).astype(float) + rng.normal(0, 1.0, len(fr))
        sp = SelectionSpace(pr, fr, w - w.min() + 1.0)
        if sp.pA <= 0:
            continue
        made += 1
        yield {"family": "sparse_support", "frontier_size": len(fr)}, sp


def library_instances():
    """The receiver policy library (stance x tier) under the world population prior (v1 cost model)."""
    from ..simulator.world import World
    w = World(20, 10_000, 7, 20_000)
    for stance, tiers in w.policies.items():
        for tier, node in tiers.items():
            fr = pol.frontier(node, len(w.prior.dims))
            if len(fr) < 2:
                continue
            yield {"family": "policy_library", "policy": f"{stance}/{tier}", "frontier_size": len(fr)}, \
                SelectionSpace(w.prior, fr, np.array([w.cost(u) for u in fr]))


# ------------------------------------------------------------------------------------------ E23
def run_e23(cfg, out):
    rows = []
    for meta, sp, _ in e11_instances(cfg):
        rows.append(dict({"family": f"v1_e11_{meta['prior']}", "frontier_size": meta["frontier_size"]}, **floor_row(sp)))
    for meta, sp in sparse_instances(cfg, cfg.get("e23_sparse", 103)):
        rows.append(dict(meta, **floor_row(sp)))
    for meta, sp in library_instances():
        rows.append(dict(meta, **floor_row(sp)))
    summ = []
    for fam in sorted({r["family"] for r in rows}):
        rr = [r for r in rows if r["family"] == fam]
        det = np.array([r["Lambda_DET_bits"] for r in rr])
        flo = np.array([r["Lambda_floor_bits"] for r in rr])
        summ.append({"family": fam, "instances": len(rr), "feasible": sum(r["zero_leakage_feasible"] for r in rr),
                     "obstruction_found": sum(r["two_branch_obstruction"] for r in rr),
                     "solver_agrees": sum(r["solver_agrees"] for r in rr),
                     "positive_Lambda_DET": int((det > 1e-12).sum()), "mean_Lambda_DET_bits": float(det.mean()),
                     "mean_Lambda_floor_bits": float(flo.mean()),
                     "mean_floor_over_DET": float(flo.sum() / det.sum()) if det.sum() > 0 else math.nan,  # ratio of means
                     "max_zero_kernel_Lambda_bits": max([r.get("zero_kernel_Lambda_bits", 0.0) for r in rr])})
    return [write_csv(os.path.join(out, "e23_zero_leakage.csv"), rows),
            write_csv(os.path.join(out, "e23_summary.csv"), summ)]


# ------------------------------------------------------------------------------------------ E24
def compare_row(sp: SelectionSpace, ss, eps: float, lp: bool) -> dict:
    t0 = time.perf_counter()
    c = ss.compile(eps)
    t_compile = time.perf_counter() - t0
    Qr, lam = ss.exp_at_budget(c.budget)
    l_rand = ss.lam(Qr)
    row = {"overhead": eps, "budget": c.budget, "Lambda_DET_bits": ss.lam(ss.kernel_det()), "Lambda_RAND_bits": l_rand,
           "RAND_lambda": lam, "Lambda_PCGS_bits": c.lam_bits, "PCGS_cost": c.cost, "RAND_cost": ss.cost(Qr),
           "PCGS_gap_bits": c.gap_bits, "PCGS_mu": c.mu, "compile_ms": 1e3 * t_compile, "ba_iterations": c.iterations,
           "PCGS_reduction_vs_RAND": 1 - c.lam_bits / l_rand if l_rand > 0 else 0.0,
           "PCGS_not_worse": c.lam_bits <= l_rand + c.gap_bits + 1e-9}
    if lp:
        K, e = lp_at_budget(sp, c.budget)
        row["Lambda_LP_bits"] = sp.metrics(K)["Lambda"] if K is not None else math.nan
        row["LP_eps"] = e
        row["PCGS_not_worse_than_LP"] = K is None or c.lam_bits <= row["Lambda_LP_bits"] + c.gap_bits + 1e-7
    return row


def mismatch_row(sp: SelectionSpace, ss, eps: float, samples: int, rng) -> dict:
    """Compile PCGS with a prior estimated from `samples` draws (Laplace-smoothed); score under the true prior."""
    est = Prior(sp.prior.dims)
    counts = np.bincount(rng.choice(len(sp.prior.p), size=samples, p=sp.prior.p), minlength=len(est.p)) + 0.5
    est.p = counts / counts.sum()
    sp_e = SelectionSpace(est, sp.front, sp.w)
    ss_e = sp_e.signature_space()
    c_e = ss_e.compile(eps)
    rows_true = np.array([ss_e.index_of(s) for s in ss.sig])
    Q = c_e.Q[rows_true]  # every true signature has positive estimated mass after smoothing
    B = ss.compile(eps).budget
    Qr_e, _ = ss_e.exp_at_budget(c_e.budget)
    Qr = Qr_e[rows_true]
    return {"samples": samples, "overhead": eps, "Lambda_PCGS_est_bits": ss.lam(Q), "cost_PCGS_est": ss.cost(Q),
            "Lambda_PCGS_true_bits": ss.compile(eps).lam_bits, "Lambda_RAND_est_bits": ss.lam(Qr),
            "cost_RAND_est": ss.cost(Qr), "budget_true": B}


def run_e24(cfg, out):
    rows, mism = [], []
    lp_left = cfg.get("e24_lp_instances", 60)
    rng = np.random.default_rng(cfg.get("seed", 1) + 240)
    fams = [((dict({"family": f"v1_e11_{m['prior']}"}, frontier_size=m["frontier_size"]), sp)
             for m, sp, _ in e11_instances(cfg))]
    n_sk = cfg.get("e24_skewed", 80)
    fams += [skewed_instances(cfg, 1.2, n_sk, 241), skewed_instances(cfg, 1.6, n_sk, 242), library_instances()]
    inst = 0
    for gen in fams:
        for meta, sp in gen:
            ss = sp.signature_space()
            use_lp = lp_left > 0 and meta["family"] != "policy_library"
            for eps in OVERHEADS:
                rows.append(dict(meta, instance=inst, **compare_row(sp, ss, eps, use_lp)))
            if use_lp:
                lp_left -= 1
            if meta["family"] != "policy_library" and inst % max(1, cfg.get("e24_mismatch_every", 4)) == 0:
                for k in cfg.get("e24_mismatch_samples", [1000, 10000]):
                    mism.append(dict(meta, instance=inst, **mismatch_row(sp, ss, 0.0235, k, rng)))
            inst += 1
    summ = []
    for fam in sorted({r["family"] for r in rows}):
        for eps in OVERHEADS:
            rr = [r for r in rows if r["family"] == fam and r["overhead"] == eps]
            red = np.array([r["PCGS_reduction_vs_RAND"] for r in rr])
            lpr = [r for r in rr if "Lambda_LP_bits" in r and np.isfinite(r["Lambda_LP_bits"])]
            lp = [r["Lambda_LP_bits"] for r in lpr]
            summ.append({"family": fam, "overhead": eps, "instances": len(rr),
                         "mean_Lambda_DET_bits": float(np.mean([r["Lambda_DET_bits"] for r in rr])),
                         "mean_Lambda_RAND_bits": float(np.mean([r["Lambda_RAND_bits"] for r in rr])),
                         "mean_Lambda_PCGS_bits": float(np.mean([r["Lambda_PCGS_bits"] for r in rr])),
                         "lp_instances": len(lp),
                         "lp_paired_mean_Lambda_LP_bits": float(np.mean(lp)) if lp else math.nan,
                         "lp_paired_mean_Lambda_PCGS_bits":
                             float(np.mean([r["Lambda_PCGS_bits"] for r in lpr])) if lp else math.nan,
                         "lp_paired_mean_Lambda_RAND_bits":
                             float(np.mean([r["Lambda_RAND_bits"] for r in lpr])) if lp else math.nan,
                         "share_m_ge_3": float(np.mean([r["frontier_size"] >= 3 for r in rr])),
                         "mean_reduction_m_ge_3": _red([r for r in rr if r["frontier_size"] >= 3]),
                         "mean_reduction": float(1 - np.mean([r["Lambda_PCGS_bits"] for r in rr]) /
                                                 max(np.mean([r["Lambda_RAND_bits"] for r in rr]), 1e-300)),
                         "share_reduction_ge_10pct": float((red >= 0.10).mean()),
                         "never_worse": all(r["PCGS_not_worse"] for r in rr),
                         "max_gap_bits": float(max(r["PCGS_gap_bits"] for r in rr)),
                         "p50_compile_ms": float(np.median([r["compile_ms"] for r in rr]))})
    files = [write_csv(os.path.join(out, "e24_kernel_comparison.csv"), rows),
             write_csv(os.path.join(out, "e24_summary.csv"), summ),
             write_csv(os.path.join(out, "e24_prior_mismatch.csv"), mism),
             write_csv(os.path.join(out, "e24_frontier_curve.csv"), frontier_curve(cfg))]
    files.append(write_csv(os.path.join(out, "e24_gate.csv"), [gate1(rows)]))
    return files


def _red(rr) -> float:
    if not rr:
        return math.nan
    return float(1 - np.mean([r["Lambda_PCGS_bits"] for r in rr]) / max(np.mean([r["Lambda_RAND_bits"] for r in rr]), 1e-300))


def gate1(rows) -> dict:
    """Gate 1: PCGS never worse than RAND at equal cost, and >= 10% lower Lambda in >= 25% of skewed-prior
    instances at 2.35% overhead."""
    never = all(r["PCGS_not_worse"] for r in rows)
    sk = [r for r in rows if r["family"].startswith("skewed") and r["overhead"] == 0.0235]
    share = float(np.mean([r["PCGS_reduction_vs_RAND"] >= 0.10 for r in sk])) if sk else 0.0
    ok = never and share >= 0.25
    return {"gate": "1 (after E24)", "never_worse": never, "skewed_instances": len(sk),
            "share_skewed_reduction_ge_10pct": share, "passed": ok,
            "consequence": "PCGS improvement is a headline result" if ok else
            "keep Theorem 2 as an optimality certificate (RAND within x% of optimal); headline C1 and C3"}


def frontier_curve(cfg) -> list[dict]:
    """Exact leakage-cost frontier on the toy space (uniform and ferromagnetic priors) with the baselines."""
    rows = []
    for prior_name in ("uniform", "ferro", "skewed"):
        rng = np.random.default_rng(cfg.get("seed", 1) + 12)
        pr = None if prior_name == "uniform" else Prior.random_ising((4, 4, 4, 4), rng, "ferro", 1.5,
                                                                      0.7 if prior_name == "ferro" else 1.6)
        _, _, sp = blueprint_toy_space(pr)
        ss = sp.signature_space()
        c_det = ss.cost(ss.kernel_det())

        def rec(mode, param, lam, cost):
            rows.append({"prior": prior_name, "mode": mode, "param": param, "Lambda_bits": lam,
                         "cost_overhead": cost / c_det - 1})

        rec("DET", "", ss.lam(ss.kernel_det()), c_det)
        fl = ss.lambda_floor()
        rec("FLOOR", "", fl["Lambda"], ss.cost(fl["Q"]))
        for mu in np.concatenate([[0.0], np.geomspace(0.01, 50, 40)]):
            Q, _, _, _ = ss.blahut_arimoto(float(mu))
            rec("PCGS", float(mu), ss.lam(Q), ss.cost(Q))
        for lam in np.concatenate([[0.0], np.geomspace(0.01, 50, 40)]):
            Q = ss.kernel_exp(float(lam))
            rec("RAND", float(lam), ss.lam(Q), ss.cost(Q))
        for eps in (0.01, 0.0235, 0.05, 0.1, 0.2):
            K, e = lp_at_budget(sp, c_det * (1 + eps))
            if K is not None:
                m = sp.metrics(K)
                rec("LP", round(e, 4), m["Lambda"], m["exp_cost"])
        rec("BOP", "", 0.0, float(sp.w.sum()))
    return rows
