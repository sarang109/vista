"""E17 additivity gap (N4), E18 fixed-charge greedy (N5), E4/E19 MAN (N6), E20 extraction (N7),
E21/E22 audit deterrence and commitment overhead (N8)."""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .. import audit, group_cost as gc, leakage as lk, man, policy as pol
from .. import policy_extraction as pe
from ..claims import synthetic_schema
from ..mdps_exact import solve_frontier
from ..prior import Prior
from ..util import percentiles, write_csv


# ----------------------------------------------------------------------------------- E17 (N4)
def run_e17(cfg, out):
    """Additive leakage weights vs true surprisal. Mis-ranking is only possible when the payer can
    satisfy >= 2 frontier branches, so the pick comparison is restricted to those 'choice' decisions."""
    rng = np.random.default_rng(cfg.get("seed", 1) + 17)
    sch = synthetic_schema(5, 4)
    rows = []
    for kind in ("independent", "ferro", "anti", "mixed"):
        for strength in (0.5, 1.5, 3.0):
            pr = Prior.random_ising(sch.dims, rng, kind if kind != "independent" else "none", strength)
            add = lk.additive_weights(pr)
            n = choice = diff = 0
            regret = []
            gaps = []
            for _ in range(cfg.get("e17_policies", 40)):
                node = pol.random_dnf(sch, int(rng.integers(3, 7)), int(rng.integers(1, 4)), rng)
                fr = pol.frontier(node, sch.n)
                true_w = np.array([pr.surprisal(u) for u in fr])
                add_w = np.array([add(u) for u in fr])
                gaps.extend((add_w - true_w).tolist())
                for x in pr.sample(rng, 30):
                    avail = (fr <= x).all(1)
                    if not avail.any():
                        continue
                    n += 1
                    if avail.sum() < 2:
                        continue
                    choice += 1
                    a = solve_frontier(fr, x, None, add_w)
                    t = solve_frontier(fr, x, None, true_w)
                    diff += a.profile != t.profile
                    regret.append(pr.surprisal(a.profile) - t.cost)
            gaps = np.array(gaps)
            rows.append({"prior": kind, "coupling": strength, "log_supermodular": pr.is_log_supermodular()[0],
                         "decisions": n, "choice_decisions": choice,
                         "additive_picks_different_set": diff / max(choice, 1),
                         "mean_true_leakage_regret_bits": float(np.mean(regret)) if regret else 0.0,
                         "max_true_leakage_regret_bits": float(np.max(regret)) if regret else 0.0,
                         "mean_additivity_gap_bits": float(gaps.mean()), "min_additivity_gap_bits": float(gaps.min()),
                         "frac_underestimates": float((gaps < -1e-9).mean())})
    return [write_csv(os.path.join(out, "e17_additivity_gap.csv"), rows)]


# ----------------------------------------------------------------------------------- E18 (N5)
def measure_sd_costs(reps: int = 200) -> dict:
    """Measure fixed charge F_g and marginal m(c) of the selective-disclosure backend (bytes and ms)."""
    from ..crypto import selective_disclosure as sd
    from ..crypto.signed import SigningKey
    from ..util import canonical_json
    key = SigningKey()
    bind = {"pid": "0" * 64, "aud": "FI001", "nonce": "0" * 32}
    data = []
    for k in range(1, 9):
        labels = [f"attr{i}>=L1" for i in range(k)]
        ms_v = []
        for _ in range(reps):
            cred, disc = sd.issue(key, "g", labels, k, bind)
            pres = sd.present(cred, disc, labels)
            t = time.perf_counter()
            sd.verify(pres, key.public_bytes, bind)
            ms_v.append((time.perf_counter() - t) * 1e3)
        data.append((k, len(canonical_json(pres)), float(np.median(ms_v))))
    ks = np.array([d[0] for d in data])
    by = np.polyfit(ks, [d[1] for d in data], 1)
    ms = np.polyfit(ks, [d[2] for d in data], 1)
    return {"F_bytes": float(by[1]), "m_bytes": float(by[0]), "F_ms": float(ms[1]), "m_ms": float(ms[0]),
            "points": data}


def run_e18(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 18)
    meas = measure_sd_costs(cfg.get("e18_measure_reps", 100))
    rows = [{"kind": "measured_backend", "F_bytes": meas["F_bytes"], "m_bytes": meas["m_bytes"],
             "F_over_m_bytes": meas["F_bytes"] / meas["m_bytes"], "F_ms": meas["F_ms"], "m_ms": meas["m_ms"]}]
    for k in (4, 8, 16, 32, 64):
        inst = gc.adversarial_instance(k)
        rows.append({"kind": "adversarial_family", "k": k, "myopic_ratio": gc.myopic_greedy(inst)[1] / 2,
                     "density_ratio": gc.density_greedy(inst)[1] / 2})
    fm = meas["F_bytes"] / meas["m_bytes"]
    for n_groups in (2, 4, 6, 9, 12):
        for fm_ratio in (1.0, 3.0, fm, 20.0):
            rd, rm = [], []
            for _ in range(cfg.get("e18_reps", 60)):
                inst = gc.random_instance(rng, int(rng.integers(3, 11)), n_groups,
                                          F_range=(0.5 * fm_ratio, 1.5 * fm_ratio), m_range=(0.5, 1.5))
                _, ce = gc.exact_ilp(inst)
                rd.append(gc.density_greedy(inst)[1] / ce)
                rm.append(gc.myopic_greedy(inst)[1] / ce)
            rows.append({"kind": "random", "groups": n_groups, "F_over_m": round(fm_ratio, 2),
                         "density_mean_ratio": float(np.mean(rd)), "density_max_ratio": float(np.max(rd)),
                         "myopic_mean_ratio": float(np.mean(rm)), "myopic_max_ratio": float(np.max(rm)),
                         "density_optimal_rate": float(np.mean(np.abs(np.array(rd) - 1) < 1e-9)),
                         "myopic_optimal_rate": float(np.mean(np.abs(np.array(rm) - 1) < 1e-9))})
    return [write_csv(os.path.join(out, "e18_fixed_charge_greedy.csv"), rows)]


# ----------------------------------------------------------------------------------- E19 (N6)
def run_e19(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 19)
    rows = []
    for depth in (1, 2, 3):
        for n_act in (4, 6, 8, 10):
            for rep in range(cfg.get("e19_reps", 30)):
                acts, nd = man.random_readonce_actions(rng, n_act, depth)
                if isinstance(nd, man.AConst):
                    continue
                strat = {
                    "optimal_dp": man.optimal_dp(nd, acts),
                    "ratio_rule": man.ratio_strategy(nd, acts),
                    "fixed_order": man.fixed_order_strategy(nd, list(range(len(acts)))),
                }
                strength = lambda a: 1.0 / max(a.p, 1e-3) + a.k  # "strongest" = most demanding controls
                bb = man.blanket_branch(nd, acts, strength)
                strat["blanket"] = man.ratio_strategy(bb, acts)
                opt = None
                for name, ch in strat.items():
                    target = bb if name == "blanket" else nd
                    ev = man.evaluate_strategy(target, acts, ch)
                    if name == "optimal_dp":
                        opt = ev
                    rows.append({"depth": man.formula_depth(nd), "n_actions": n_act, "rep": rep, "strategy": name,
                                 "exp_friction": ev["exp_friction"], "success_prob": ev["success_prob"],
                                 "friction_per_success": ev["friction_per_success"], "exp_rounds": ev["exp_rounds"],
                                 "regret_vs_opt": ev["exp_friction"] / opt["exp_friction"] - 1 if name != "blanket" else math.nan})
                # robustness: plan with mis-estimated p, evaluate with true p
                for noise in (0.05, 0.15, 0.3):
                    est = [man.Action(a.name, a.k, float(np.clip(a.p + rng.normal(0, noise), 0.02, 0.99))) for a in acts]
                    ch = man.optimal_dp(nd, est)
                    ev = man.evaluate_strategy(nd, acts, ch)
                    rows.append({"depth": man.formula_depth(nd), "n_actions": n_act, "rep": rep,
                                 "strategy": f"dp_misestimated_sd{noise}", "exp_friction": ev["exp_friction"],
                                 "success_prob": ev["success_prob"], "friction_per_success": ev["friction_per_success"],
                                 "exp_rounds": ev["exp_rounds"], "regret_vs_opt": ev["exp_friction"] / opt["exp_friction"] - 1})
    return [write_csv(os.path.join(out, "e19_stochastic_man.csv"), rows)]


def run_e4(cfg, out, world):
    """Deterministic MAN (v1.0) vs blanket step-up on the interbank policy library."""
    from ..claims import DEFAULT_SCHEMA as S
    from ..simulator.world import ACTIONS
    rng = np.random.default_rng(cfg.get("seed", 1) + 4)
    rows = []
    strongest = [0, 1, 2, 3, 4]  # blanket: request every control in the catalog order of strength
    for stance, tiers in world.policies.items():
        for tier, node in tiers.items():
            fr = pol.frontier(node, S.n)
            for _ in range(cfg.get("e4_reps", 150)):
                x = tuple(int(v) for v in world.prior.sample(rng, 1)[0])
                if pol.evaluate(node, x):
                    continue
                # receiver sees the negotiation-aware partial disclosure min(u, x)
                parts = {tuple(int(min(a, b)) for a, b in zip(u, x)) for u in fr}
                best = None
                for part in parts:
                    Q, c = man.deterministic_man(node, part, ACTIONS)
                    if Q is not None and (best is None or c < best[1]):
                        best = (Q, c, part)
                blanket_c, blanket_ok = 0.0, False
                if best is not None:
                    prof = list(best[2])
                    for a in strongest:
                        blanket_c += ACTIONS[a].k
                        for i, t in ACTIONS[a].delta:
                            prof[i] = max(prof[i], t)
                    blanket_ok = pol.evaluate(node, tuple(prof))
                rows.append({"stance": stance, "tier": tier, "resolvable": best is not None,
                             "man_cost": best[1] if best else math.nan, "man_rounds": len(best[0]) if best else math.nan,
                             "man_actions": "+".join(ACTIONS[a].name for a in best[0]) if best else "",
                             "blanket_cost": blanket_c if best else math.nan, "blanket_resolves": blanket_ok})
    return [write_csv(os.path.join(out, "e4_deterministic_man.csv"), rows)]


# ----------------------------------------------------------------------------------- E20 (N7)
def run_e20(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 20)
    rows = []
    for (n, k, m) in ((8, 2, 3), (8, 3, 3), (10, 3, 4), (12, 3, 4)):
        for budget in (5, 10, 20, 40, 80, None):
            accs, qs, comp = [], [], []
            for _ in range(cfg.get("e20_reps", 12)):
                terms = pe.random_kdnf(rng, n, k, m)
                o = pe.PolicyOracle(n, terms, budget)
                r = pe.extract_dnf(o)
                accs.append(pe.agreement(terms, r["terms"], n, rng, 2000))
                qs.append(r["queries"])
                comp.append(r["complete"])
            logF = pe.counting_bound_exact(n, k, m)
            rows.append({"n": n, "k": k, "m": m, "budget": budget or "unlimited", "mean_queries": float(np.mean(qs)),
                         "extracted_accuracy": float(np.mean(accs)), "complete_rate": float(np.mean(comp)),
                         "log2_F_bits": logF, "info_cap_binary_bits": (budget or float("nan")),
                         "info_cap_token4_bits": (budget * 2 if budget else float("nan")),
                         "info_cap_token8_bits": (budget * 3 if budget else float("nan")),
                         "info_cap_missing_set_bits": (budget * n if budget else float("nan")),
                         "min_queries_binary": pe.min_queries(logF, 2), "min_queries_token4": pe.min_queries(logF, 4),
                         "min_queries_missing_set": pe.min_queries(logF, 2 ** n)})
    return [write_csv(os.path.join(out, "e20_policy_extraction.csv"), rows)]


# ----------------------------------------------------------------------------------- E21/E22 (N8)
def run_e21(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1) + 21)
    rows = []
    senders = cfg.get("e21_senders", 400)
    payments = cfg.get("e21_payments", 200)
    g = rng.lognormal(np.log(20), 0.6, senders)
    F = rng.lognormal(np.log(400), 0.5, senders)
    for delta, label in ((0.99, "commit_before_settle"), (0.1, "no_precommitment")):
        for q in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            lie = np.array([audit.lying_profitable(gi, Fi, q, delta) for gi, Fi in zip(g, F)])
            # realise audits: each lying sender lies on every payment; audited w.p. q, detected w.p. delta
            caught = rng.binomial(payments, q * delta, senders) * lie
            profit = (g * payments * lie) - caught * (g + F)
            rows.append({"regime": label, "delta": delta, "audit_rate": q, "lying_rate": float(lie.mean()),
                         "mean_q_star": float(np.mean([audit.q_star(gi, Fi, delta) for gi, Fi in zip(g, F)])),
                         "audits_per_1k_payments": 1000 * q, "mean_liar_realised_profit": float(profit[lie].mean()) if lie.any() else 0.0,
                         "caught_per_liar": float(caught[lie].mean()) if lie.any() else 0.0})
    return [write_csv(os.path.join(out, "e21_rational_liar.csv"), rows)]


def run_e22(cfg, out):
    from ..util import canonical_json
    rng = np.random.default_rng(cfg.get("seed", 1) + 22)
    rows = []
    log = audit.TransparencyLog()
    for n_claims in (1, 2, 4, 6, 9):
        ms, bytes_ = [], []
        opened_ok = opened_n = 0
        for j in range(cfg.get("e22_reps", 300)):
            ev = [{"type": "auth", "method": "fido2_passkey", "ts": 1788220800 + j, "i": i} for i in range(n_claims)]
            salt = rng.bytes(16)
            t = time.perf_counter()
            leaves = audit.evidence_leaves(ev, salt)
            tree = audit.MerkleTree(leaves)
            r = log.append(f"{j:064x}", "FI001", tree.root, [f"c{i}" for i in range(n_claims)], 1788220800 + j)
            ms.append((time.perf_counter() - t) * 1e3)
            bytes_.append(len(canonical_json(dict(r, root=tree.root.hex()))))
            # audit opening: every committed leaf verifies against the logged root
            opened_ok += all(audit.MerkleTree.verify(l, tree.proof(k), tree.root) for k, l in enumerate(leaves))
            opened_n += 1
        pc = percentiles(ms)
        rows.append({"claims_per_capsule": n_claims, "commit_p50_ms": pc["p50"], "commit_p99_ms": pc["p99"],
                     "receipt_bytes": float(np.mean(bytes_)), "audit_openings_ok": f"{opened_ok}/{opened_n}"})
    rows.append({"claims_per_capsule": "log_chain_verified", "commit_p50_ms": "", "commit_p99_ms": "",
                 "receipt_bytes": "", "log_entries": len(log.entries), "chain_ok": log.verify_chain()})
    return [write_csv(os.path.join(out, "e22_commitment_overhead.csv"), rows)]
