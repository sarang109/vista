"""E26: do the selection-leakage results hold on priors estimated from login signals? (go/no-go gate 2)

Data: the RBA login dataset (simulator/rba.py). Four claim chains are derived from each user's own login
history; their empirical joint is the prior. Two policy families:
  rba_*     random monotone policies over the four login chains alone (every prior value is observed data)
  hybrid    policies over the four login chains plus the six VISTA chains that no public dataset covers
            (identity, authentication strength, phishing resistance, credential age, dynamic binding, control
            execution); prior = empirical login joint x the v1 population prior on those six chains
For each policy E23 (zero-leakage feasibility, floor vs DET) and E24 (DET / RAND / LP / PCGS at equal cost) are
repeated. The positive-association (FKG) assumption is tested pairwise and on the full joint.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import policy as pol
from ..claims import DEFAULT_SCHEMA, Schema
from ..prior import Prior
from ..progress import PROGRESS
from ..selection import SelectionSpace
from ..simulator import rba
from ..simulator.source import resolve_path
from ..util import write_csv
from .exp_frontier import OVERHEADS, compare_row, floor_row

SYN = ("identity_assurance", "auth_strength", "phishing_resistant", "credential_age", "dynamic_binding",
       "fraud_policy_executed")


def hybrid(prior_rba: Prior):
    from ..simulator.world import World
    w = World(20, 10_000, 7, 20_000)
    idx = [DEFAULT_SCHEMA.index(a) for a in SYN]
    sch = Schema("vista-hybrid-2026.1", list(rba.RBA_SCHEMA.attributes) + [DEFAULT_SCHEMA.attributes[i] for i in idx])
    syn = Prior(tuple(DEFAULT_SCHEMA.dims[i] for i in idx))
    m = np.zeros(len(syn.p))
    np.add.at(m, w.prior.states[:, idx].astype(np.int64) @ syn._radix, w.prior.p)
    syn.p = m / m.sum()
    pr = Prior(sch.dims)
    pr.p = np.outer(prior_rba.p, syn.p).ravel()
    return sch, pr


def random_policy(sch: Schema, rng, real_attrs: int, m_terms: int):
    """DNF whose every term has at least one login-chain atom (so the observed prior matters)."""
    terms = []
    for _ in range(m_terms):
        k = int(rng.integers(1, 4))
        first = int(rng.integers(real_attrs))
        rest = [int(a) for a in rng.choice([a for a in range(sch.n) if a != first], size=k - 1, replace=False)]
        ats = [pol.Atom(i, int(rng.integers(1, sch.attributes[i].d + 1))) for i in [first] + rest]
        terms.append(ats[0] if len(ats) == 1 else pol.And(tuple(ats)))
    return pol.Or(tuple(terms)) if len(terms) > 1 else terms[0]


def instances(sch, prior, count, rng, family):
    made = tries = 0
    while made < count and tries < 50 * count:
        tries += 1
        node = random_policy(sch, rng, 4, int(rng.integers(2, 7)))
        fr = pol.frontier(node, sch.n)
        if len(fr) < 2:
            continue
        w = np.array([prior.surprisal(u) + 0.05 * int((u > 0).sum()) for u in fr])
        if not np.isfinite(w).all():
            continue
        sp = SelectionSpace(prior, fr, w)
        if sp.pA <= 0:
            continue
        made += 1
        yield {"family": family, "frontier_size": len(fr), "policy": pol.to_string(node, sch)}, sp


def run_e26(cfg, out):
    rc = cfg.get("rba", {})
    cache = resolve_path(rc.get("cache", "data/rba_cache"))
    meta = rba.ensure_cache(resolve_path(rc.get("zip", "data/rba/rba-dataset.zip")), cache,
                            rc.get("user_fraction", 0.06), rc.get("warmup_days", 60), True, PROGRESS.update)
    legit, attack = rba.load_states(cache, "legit"), rba.load_states(cache, "attack")
    files = []
    marg = []
    for i, ch in enumerate(rba.CHAINS):
        for lvl, lab in enumerate(rba.RBA_SCHEMA.attributes[i].levels):
            marg.append({"chain": ch, "level": lab, "share_legit": float((legit[:, i] == lvl).mean()),
                         "share_attack": float((attack[:, i] == lvl).mean()) if len(attack) else math.nan})
    files.append(write_csv(os.path.join(out, "e26_chain_marginals.csv"), marg))
    assoc = rba.pairwise_association(legit)
    files.append(write_csv(os.path.join(out, "e26_association.csv"), assoc))
    emp = rba.empirical_prior(legit)
    smooth = rba.empirical_prior(legit, 0.5)
    latt = rba.lattice_condition(emp)

    rng = np.random.default_rng(cfg.get("seed", 1) + 260)
    n_pol = cfg.get("e26_policies", 100)
    lp_left = cfg.get("e26_lp_instances", 30)
    floors, comps = [], []
    fams = [(rba.RBA_SCHEMA, emp, "rba_empirical"), (rba.RBA_SCHEMA, smooth, "rba_smoothed")]
    hs, hp = hybrid(emp)
    fams.append((hs, hp, "hybrid_empirical"))
    for sch, prior, fam in fams:
        for k, (m, sp) in enumerate(instances(sch, prior, n_pol, rng, fam)):
            ss = sp.signature_space()
            floors.append(dict(m, instance=k, **floor_row(sp, ss)))
            use_lp = lp_left > 0 and fam.startswith("rba")
            for eps in OVERHEADS:
                comps.append(dict(m, instance=k, **compare_row(sp, ss, eps, use_lp)))
            lp_left -= use_lp
    files.append(write_csv(os.path.join(out, "e26_zero_leakage.csv"), floors))
    files.append(write_csv(os.path.join(out, "e26_kernel_comparison.csv"), comps))

    summ = []
    for fam in [f for _, _, f in fams]:
        fl = [r for r in floors if r["family"] == fam]
        for eps in OVERHEADS:
            rr = [r for r in comps if r["family"] == fam and r["overhead"] == eps]
            if not rr:
                continue
            mr, mp = np.mean([r["Lambda_RAND_bits"] for r in rr]), np.mean([r["Lambda_PCGS_bits"] for r in rr])
            m3 = [r for r in rr if r["frontier_size"] >= 3]
            summ.append({"family": fam, "overhead": eps, "policies": len(rr),
                         "zero_leakage_feasible": sum(r["zero_leakage_feasible"] for r in fl),
                         "solver_agrees": sum(r["solver_agrees"] for r in fl),
                         "mean_Lambda_DET_bits": float(np.mean([r["Lambda_DET_bits"] for r in rr])),
                         "mean_Lambda_floor_bits": float(np.mean([r["Lambda_floor_bits"] for r in fl])),
                         "mean_Lambda_RAND_bits": float(mr), "mean_Lambda_PCGS_bits": float(mp),
                         "mean_reduction": float(1 - mp / mr) if mr > 0 else 0.0,
                         "mean_reduction_m_ge_3": float(1 - np.mean([r["Lambda_PCGS_bits"] for r in m3]) /
                                                        np.mean([r["Lambda_RAND_bits"] for r in m3])) if m3 else math.nan,
                         "share_reduction_ge_10pct": float(np.mean([r["PCGS_reduction_vs_RAND"] >= 0.1 for r in rr])),
                         "never_worse": all(r["PCGS_not_worse"] for r in rr)})
    files.append(write_csv(os.path.join(out, "e26_summary.csv"), summ))

    fkg_fail = [a for a in assoc if a["tp2_significant_violations"] > 0 or a["pqd_significant_violations"] > 0]
    emp_row = next((s for s in summ if s["family"] == "rba_empirical" and s["overhead"] == 0.0235), None)
    red = emp_row["mean_reduction"] if emp_row else 0.0
    ok = red >= 0.05 or bool(fkg_fail)
    files.append(write_csv(os.path.join(out, "e26_gate.csv"), [{
        "gate": "2 (after E26)", "mean_reduction_rba_empirical_2.35pct": red, "fkg_failing_pairs": len(fkg_fail),
        "fkg_failing_pair_names": "; ".join(f"{a['chain_i']}~{a['chain_j']}" for a in fkg_fail),
        "passed": ok, "passed_by": ("reduction>=5%" if red >= 0.05 else "") + (" FKG fails" if fkg_fail else ""),
        "consequence": "report the PCGS improvement on empirical priors" if red >= 0.05 else
        "report the real-prior frontier as the finding; drop the 'improves' wording"}]))
    files.append(write_csv(os.path.join(out, "e26_data.csv"), [dict(
        {k: v for k, v in meta.items() if k not in ("levels", "chains")},
        lattice_pairs_tested=latt["pairs_tested"], lattice_structural_zero_violations=latt["structural_zero_violations"],
        lattice_point_violations=latt["point_violations"], lattice_significant_violations=latt["significant_violations"],
        lattice_share_significant=latt["share_significant"], empirical_support_states=int((emp.p > 0).sum()),
        state_space=len(emp.p))]))
    return files
