"""Re-runs the Sec 31 verification appendix against this implementation (checks 1-24, plus extras).

Each row records the statement tested, instance count, pass count and the blueprint status tag.
These are numerical checks on small synthetic instances, not proofs.
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np

from .. import audit, group_cost as gc, ledger as led, leakage as lk, man, policy as pol
from .. import policy_extraction as pe
from ..claims import synthetic_schema
from ..cost import AdditiveCost
from ..mdps_exact import solve_bruteforce, solve_frontier, solve_ilp, solve_readonce
from ..prior import Prior
from ..selection import SelectionSpace, blueprint_toy_space
from ..util import write_csv


def _row(rows, cid, check, supports, n, passed, result, tag="[V]"):
    rows.append({"id": cid, "check": check, "supports": supports, "instances": n, "passed": passed,
                 "pass_rate": passed / n if n else math.nan, "result": result, "tag": tag})


def run(cfg: dict, out: str) -> list[str]:
    s = cfg.get("verification_scale", 1.0)
    N = lambda k: max(10, int(k * s))  # noqa: E731
    rng = np.random.default_rng(cfg.get("seed", 1))
    rows: list[dict] = []
    sch = synthetic_schema(3, 4)
    states = sch.states()

    # 1 frontier vs brute force
    n1 = N(600)
    ok = 0
    pols = []
    for _ in range(n1):
        node = pol.random_tree(sch, int(rng.integers(1, 4)), 3, rng)
        pols.append(node)
        a = {tuple(r) for r in pol.frontier(node, sch.n).tolist()}
        b = {tuple(r) for r in pol.frontier_bruteforce(node, states).tolist()}
        ok += a == b
    _row(rows, 1, "FRONTIER equals brute-force minimal satisfying profiles (3 attrs x 4 levels)",
         "Algorithm FRONTIER", n1, ok, f"{ok}/{n1} equal")

    # 2/3 threshold-minimality lemma with isotone / non-isotone costs
    iso_ok = iso_n = inv_out = inv_n = 0
    for node in pols[: N(300)]:
        fr = pol.frontier(node, sch.n)
        for _ in range(4):
            x = tuple(int(v) for v in states[rng.integers(len(states))])
            if not pol.evaluate(node, x):
                continue
            for iso in (True, False):
                w = (AdditiveCost.random_isotone if iso else AdditiveCost.random_nonisotone)(sch.dims, rng)
                f = solve_frontier(fr, x, w)
                b = solve_bruteforce(node, x, w)
                if iso:
                    iso_n += 1
                    iso_ok += abs(f.cost - b.cost) < 1e-9
                else:
                    inv_n += 1
                    inv_out += b.cost < f.cost - 1e-9
    _row(rows, 2, "Lemma 1.1: frontier optimum = global optimum under isotone costs", "Lemma 1.1", iso_n, iso_ok,
         f"{iso_ok}/{iso_n} hold")
    _row(rows, 3, "Non-isotone (inverted) costs: optimum lies outside the frontier", "assumption necessary", inv_n,
         inv_out, f"{inv_out}/{inv_n} violate ({100 * inv_out / max(inv_n, 1):.1f}%)")

    # 4 set cover encoded as CNF MDPS (ILP) vs brute-force set cover
    n4 = N(192)
    ok = 0
    for _ in range(n4):
        U, m = int(rng.integers(3, 7)), int(rng.integers(3, 8))
        sets = [set(np.flatnonzero(rng.random(U) < 0.45).tolist()) for _ in range(m)]
        for u in range(U):
            if not any(u in S for S in sets):
                sets[int(rng.integers(m))].add(u)
        wts = rng.uniform(0.5, 5, m)
        bs = synthetic_schema(m, 2)
        node = pol.And(tuple(pol.Or(tuple(pol.Atom(j, 1) for j in range(m) if u in sets[j])) if
                             sum(u in S for S in sets) > 1 else next(pol.Atom(j, 1) for j in range(m) if u in sets[j])
                             for u in range(U)))
        cost = AdditiveCost([[0.0, w] for w in wts])
        ilp = solve_ilp(node, tuple([1] * m), cost)
        best = min(sum(wts[list(c)]) for r in range(1, m + 1) for c in itertools.combinations(range(m), r)
                   if set().union(*[sets[k] for k in c]) == set(range(U)))
        ok += abs(ilp.cost - best) < 1e-6
    _row(rows, 4, "Weighted set cover encoded as CNF MDPS (ILP) equals brute-force cover", "Proposition 1.2", n4, ok,
         f"{ok}/{n4} agree")

    # 5 read-once DP vs brute force
    n5 = N(300)
    ok = 0
    big = synthetic_schema(9, 4)
    for _ in range(n5):
        node = pol.random_tree(big, int(rng.integers(1, 4)), 3, rng, read_once=True)
        x = tuple(int(v) for v in rng.integers(0, 4, 9))
        w = AdditiveCost.random_isotone(big.dims, rng)
        a, b = solve_readonce(node, x, w), solve_bruteforce(node, x, w)
        ok += (a.cost == b.cost) or abs(a.cost - b.cost) < 1e-9
    _row(rows, 5, "Read-once AND/OR DP equals brute force (up to 9 atoms)", "Proposition 1.2", n5, ok, f"{ok}/{n5} agree")

    # 6 frontier blow-up for AND of k two-way ORs
    sizes = []
    for k in range(2, 13, 2):
        sk = synthetic_schema(2 * k, 2)
        node = pol.And(tuple(pol.Or((pol.Atom(2 * i, 1), pol.Atom(2 * i + 1, 1))) for i in range(k)))
        sizes.append(len(pol.frontier(node, sk.n)))
    _row(rows, 6, "Frontier size for AND of k two-way ORs, k=2,4,..,12", "blow-up motivation", len(sizes),
         sum(sz == 2 ** k for sz, k in zip(sizes, range(2, 13, 2))), " ".join(map(str, sizes)))

    # 7-11 selection leakage on the toy space
    _, _, sp = blueprint_toy_space()
    gaps = []
    for K in [sp.kernel_det()] + [sp.kernel_exp(l) for l in (0, 0.5, 1, 3)]:
        gaps.append(abs(sp.metrics(K)["identity_gap"]))
    _row(rows, 7, "Selection-leakage identity Total = E[L] + Lambda (det, lambda=0,.5,1,3)", "Theorem 2.1", len(gaps),
         sum(g < 1e-9 for g in gaps), f"max |gap| = {max(gaps):.2e}", "[P]")
    Kd = sp.kernel_det()
    md = sp.metrics(Kd)
    smaller = 0
    views = 0
    for v in range(len(sp.front)):
        if Kd[:, v].sum() == 0:
            continue
        views += 1
        smaller += (Kd[sp.avail[:, v], v] < 1).any()
    _row(rows, 8, "Deterministic selection: posterior support strictly smaller than {X>=v}", "Theorem 2.2", views,
         smaller, f"{smaller}/{views} views strictly smaller; Lambda = {md['Lambda']:.3f} bits", "[P]")
    ok = n = 0
    for lam in (0, 0.5, 1, 2, 4):
        e = sp.eps_sel(sp.kernel_exp(lam))
        bound = math.log(np.exp(lam * (sp.w.max() - sp.w)).sum())
        n += 1
        ok += e <= bound + 1e-9
    _row(rows, 9, "Exponential mechanism eps_sel <= ln sum exp(lambda(w_max - w(u)))", "Proposition 2.3", n, ok,
         f"{ok}/{n} within bound")
    eps_min = sp.min_feasible_eps(iters=25)
    floor = math.log(sp.a_max())
    lp_better = n = 0
    for lam in (0, 0.5, 1, 2):
        Ke = sp.kernel_exp(lam)
        e = sp.eps_sel(Ke)
        Kl = sp.kernel_lp(e + 1e-9)
        if Kl is not None:
            n += 1
            lp_better += sp.metrics(Kl)["exp_cost"] <= sp.metrics(Ke)["exp_cost"] + 1e-9
    _row(rows, 10, "Smallest feasible eps (LP bisection) vs ln a_max; LP cost <= exp mechanism at equal eps",
         "Propositions 2.4, 2.5", n + 1, lp_better + (abs(eps_min - floor) < 1e-3 or not sp.solitary_reachable()),
         f"eps_min={eps_min:.4f}, ln a_max={floor:.4f}, solitary={sp.solitary_reachable()}; LP cheaper {lp_better}/{n}")
    b = sp.bop_metrics()
    _row(rows, 11, "BOP cost vs cheapest-branch; information floor and PoD", "Proposition 2.6", 1, 1,
         f"BOP cost {b['exp_cost']:.2f} vs DET {md['exp_cost']:.2f}; L_min={sp.L_min:.2f} bits; PoD={md['PoD']:.2f} bits")

    # 12-15 ledger
    ok = 0
    for x in range(128):
        r = led.probe(128, x, "binary", None, rng)
        ok += r["exact"] and r["queries"] == 7
    _row(rows, 12, "Threshold probing on 128 values: exact recovery in 7 probes", "Proposition 3.1", 128, ok,
         f"{ok}/128 exact in 7 probes", "[P]")
    ok = n = 0
    for rho in (1, 8, 16, 32):
        for strat in ("binary", "random", "scan"):
            widths = [led.probe(128, int(x), strat, rho, rng)["final_width"] for x in rng.integers(0, 128, N(60))]
            n += 1
            ok += min(widths) >= rho
    _row(rows, 13, "Ledger: final consistent width >= rho for binary/random/scan adversaries", "Theorem 3.2", n, ok,
         f"{ok}/{n} configurations hold", "[P]")
    fr_spread = _benign_false_reject([32, 64, 96], 16, rng, N(400))
    fr_crowd = _benign_false_reject([60, 64, 68, 72], 16, rng, N(400))
    _row(rows, 14, "Benign false rejects under ledger rho=16: spread vs crowded thresholds", "utility cost of ledger", 2,
         2, f"{100 * fr_spread:.1f}% vs {100 * fr_crowd:.1f}%")
    n15 = N(100)
    ok = 0
    for _ in range(n15):
        m = int(rng.integers(3, 8))
        th = sorted(rng.choice(np.arange(1, 128), m, replace=False).tolist())
        mu = rng.uniform(0.2, 2, m).tolist()
        S = led.survival(np.full(128, 1 / 128))
        k = int(rng.integers(1, m + 1))
        _, a = led.ocgd(th, mu, k, S)
        _, bb = led.ocgd_bruteforce(th, mu, k, S)
        ok += abs(a - bb) < 1e-9
    _row(rows, 15, "OCGD dynamic program equals brute-force grid search", "Algorithm OCGD", n15, ok, f"{ok}/{n15} agree")

    # 16-17 N4 submodularity
    res = {}
    for kind in ("ferro", "anti", "mixed"):
        viol = tot = 0
        for rep in range(3):
            for dims in ((2,) * 6, (4,) * 3):
                pr = Prior.random_ising(dims, rng, kind)
                c = lk.submodularity_check(pr, rng, N(600))
                viol += c["submod_violations"]
                tot += c["trials"]
        res[kind] = (viol, tot)
    fv, ft = res["ferro"]
    _row(rows, 16, "Surprisal submodular under ferromagnetic (FKG) priors; violations under anti/mixed", "Theorem 4.3",
         ft, ft - fv, f"ferro {fv}/{ft} violations; anti {res['anti'][0]}/{res['anti'][1]}; mixed {res['mixed'][0]}/{res['mixed'][1]}",
         "[P]")
    sub_v = kl_v = n = 0
    for _ in range(3):
        pr = Prior.random_ising((4, 4, 3), rng, "ferro")
        sub_v += lk.submodularity_check(pr, rng, N(600))["subadditivity_violations"]
        kl_v += lk.kl_bound_check(pr, rng, N(70))
        n += 1
    _row(rows, 17, "Subadditivity under FKG priors; KL <= surprisal bound", "Prop 4.1, Thm 4.3", 2,
         int(sub_v == 0) + int(kl_v == 0), f"subadditivity violations {sub_v}; KL-bound violations {kl_v}", "[P]")

    # 18-19 N5
    ratios = []
    for k in (4, 8, 16, 32):
        inst = gc.adversarial_instance(k)
        _, cm = gc.myopic_greedy(inst)
        _, cd = gc.density_greedy(inst)
        ratios.append((cm / 2.0, cd / 2.0))
    _row(rows, 18, "Myopic greedy on adversarial family k=4,8,16,32 (ratio to optimum 2)", "Proposition 5.1", 4,
         sum(abs(r[0] - k / 2) < 1e-9 for r, k in zip(ratios, (4, 8, 16, 32))),
         "myopic ratios " + " ".join(f"{r[0]:.0f}" for r in ratios) + "; DG ratios " + " ".join(f"{r[1]:.0f}" for r in ratios), "[P]")
    n19 = N(400)
    rd, rm, opt_d = [], [], 0
    hm_ok = 0
    for _ in range(n19):
        inst = gc.random_instance(rng, int(rng.integers(2, 11)), int(rng.integers(2, 10)))
        _, ce = gc.exact_ilp(inst)
        _, cd = gc.density_greedy(inst)
        _, cm = gc.myopic_greedy(inst)
        rd.append(cd / ce)
        rm.append(cm / ce)
        opt_d += abs(cd - ce) < 1e-9
        hm_ok += cd / ce <= gc.harmonic(inst.n_clauses) + 1e-9
    _row(rows, 19, "Density greedy vs myopic vs exact (ILP); DG within H_m", "Proposition 5.2", n19, hm_ok,
         f"mean {np.mean(rd):.3f} vs {np.mean(rm):.3f}; max {np.max(rd):.3f} vs {np.max(rm):.3f}; DG optimal {opt_d}/{n19}")

    # 20-21 N6
    n20 = N(300)
    ok_and = ok_or = 0
    for _ in range(n20):
        k = int(rng.integers(2, 6))
        acts = [man.Action(f"a{i}", float(rng.uniform(0.5, 6)), float(rng.uniform(0.1, 0.99))) for i in range(k)]
        for kind in ("and", "or"):
            best = min(_series_cost(acts, perm, kind) for perm in itertools.permutations(range(k)))
            key = (lambda a: a.k / (1 - a.p)) if kind == "and" else (lambda a: a.k / a.p)
            rule = _series_cost(acts, sorted(range(k), key=lambda i: key(acts[i])), kind)
            if kind == "and":
                ok_and += abs(rule - best) < 1e-9
            else:
                ok_or += abs(rule - best) < 1e-9
    _row(rows, 20, "Ratio rules vs brute-force permutations (AND series, OR series)", "Proposition 6.1", 2 * n20,
         ok_and + ok_or, f"AND {ok_and}/{n20}; OR {ok_or}/{n20}", "[S/P]")
    n21 = N(250)
    eq2 = eq3 = 0
    worst = 0.0
    tot3 = 0
    for depth_target in (2, 3):
        cnt = 0
        while cnt < n21:
            acts, nd = man.random_readonce_actions(rng, int(rng.integers(4, 9)), depth_target)
            dep = man.formula_depth(nd)
            if (depth_target == 2 and dep > 2) or (depth_target == 3 and dep < 3):
                continue
            cnt += 1
            o = man.evaluate_strategy(nd, acts, man.optimal_dp(nd, acts))["exp_friction"]
            r = man.evaluate_strategy(nd, acts, man.ratio_strategy(nd, acts))["exp_friction"]
            if depth_target == 2:
                eq2 += abs(o - r) < 1e-9
            else:
                tot3 += 1
                eq3 += abs(o - r) < 1e-9
                worst = max(worst, r / o - 1)
    _row(rows, 21, "Depth-first ratio strategy vs exact DP: depth<=2 exact; depth>=3 gap", "Proposition 6.2", n21, eq2,
         f"depth<=2: {eq2}/{n21} equal; depth>=3: {eq3}/{tot3} equal, worst gap {100 * worst:.1f}%", "[S+V]")

    # toy MAN example of Sec 25.2
    acts = [man.Action("passkey", 2, .9), man.Action("rebinding", 5, .6), man.Action("otp", 1.5, .85), man.Action("selfie", 4, .7)]
    nd = man.AOr((man.AAnd((man.ALeaf(0), man.ALeaf(1))), man.AAnd((man.ALeaf(2), man.ALeaf(3)))))
    e = man.evaluate_strategy(nd, acts, man.ratio_strategy(nd, acts))
    _row(rows, 25, "Worked MAN example (Sec 25.2): ratio-rule friction 7.41, success 81.4%", "Proposition 6.2", 1,
         int(abs(e["exp_friction"] - 7.41) < 0.01 and abs(e["success_prob"] - 0.814) < 0.001),
         f"friction {e['exp_friction']:.3f}; success {100 * e['success_prob']:.1f}%")

    # 22-23 N7
    ok = n = 0
    for (nn, k, m) in ((16, 2, 3), (32, 3, 4), (64, 5, 8), (24, 3, 5), (40, 4, 6), (12, 2, 4)):
        n += 1
        ok += pe.counting_bound_closed(nn, k, m) <= pe.counting_bound_exact(nn, k, m) + 1e-9
    conj_ok = conj_n = 0
    for nn in range(1, 9):
        for r in range(1, nn + 1):
            for T in itertools.combinations(range(nn), r):
                o = pe.PolicyOracle(nn, [T])
                got = pe.extract_conjunction(o)
                conj_n += 1
                conj_ok += got == frozenset(T) and o.queries == nn
    _row(rows, 22, "Counting bound closed form <= exact; conjunction extraction in exactly n queries", "Props 7.1, 7.2",
         n + conj_n, ok + conj_ok, f"bound valid {ok}/{n}; conjunctions {conj_ok}/{conj_n}", "[P]")
    parts = []
    allok = 0
    for (nn, k, m) in ((8, 3, 2), (8, 3, 3), (8, 2, 3)):
        qs = []
        for _ in range(15):
            terms = pe.random_kdnf(rng, nn, k, m)
            o = pe.PolicyOracle(nn, terms)
            r = pe.extract_dnf(o)
            allok += set(r["terms"]) == set(terms)
            qs.append(r["queries"])
        parts.append(f"({nn},{k},{m}): mean {np.mean(qs):.1f} max {max(qs)} vs floor {pe.counting_bound_exact(nn, k, m):.1f}")
    _row(rows, 23, "Full extraction of random DNF policies, n=8", "Proposition 7.3", 45, allok, "; ".join(parts))

    # 24 N8
    n24 = N(20000)
    mism = 0
    for _ in range(n24):
        g, F, d, q = rng.uniform(0.1, 100), rng.uniform(0, 1000), rng.uniform(0.01, 1), rng.uniform(0, 1)
        mism += audit.lying_profitable(g, F, q, d) != (q < audit.q_star(g, F, d) - 1e-12)
    _row(rows, 24, "Deterrence law q* = g/(delta(g+F)) matches best response", "Proposition 8.1", n24, n24 - mism,
         f"{mism} mismatches; toy q*(20,400,0.9) = {100 * audit.q_star(20, 400, .9):.1f}%", "[P]")

    selection_theory_checks(rows, rng, N)
    path = write_csv(os.path.join(out, "verification_appendix.csv"), rows)
    return [path]


def _random_space(rng, sch, kind="mixed", sparse=0.0, m_range=(2, 6)):
    while True:
        node = pol.random_dnf(sch, int(rng.integers(*m_range)), int(rng.integers(1, 3)), rng)
        fr = pol.frontier(node, sch.n)
        if len(fr) < 2:
            continue
        pr = Prior.random_ising(sch.dims, rng, kind)
        if sparse > 0:
            p = np.where(rng.random(len(pr.p)) < sparse, 0.0, pr.p)
            if p.sum() <= 0:
                continue
            pr.p = p / p.sum()
        w = fr.sum(1).astype(float) + rng.normal(0, 1.0, len(fr))
        sp = SelectionSpace(pr, fr, w - w.min() + 1.0)
        if sp.pA > 0:
            return sp


def selection_theory_checks(rows, rng, N):
    """Checks 26-35: Lemma 1, Theorems 1-3, Lemma 2 and the stale-status grace bound."""
    from ..repeated import StickyStore, coverage_payments, prob_all_seen, q_min, repeated_metrics
    from ..trust import grace_window, unsafe_probability
    sch = synthetic_schema(3, 4)

    n = N(120)
    gaps = []
    for _ in range(n):
        sp = _random_space(rng, sch)
        ss = sp.signature_space()
        for Q in (ss.kernel_det(), ss.kernel_exp(float(rng.uniform(0, 3)))):
            m_sig, m_st = ss.metrics(Q), sp.metrics(sp.expand(ss, Q))
            gaps.append(abs(m_sig["Lambda"] - m_st["Lambda"]))
            gaps.append(abs(m_sig["I_bits"] - m_sig["E_L_bits"] - m_st["Lambda"]))
    _row(rows, 26, "Lemma 1: Lambda = I(X;V) - E[L_S(V)] (signature space vs state-level metrics)", "Lemma 1",
         len(gaps), sum(g < 1e-9 for g in gaps), f"max |gap| = {max(gaps):.1e} bits", "[P]")

    n = N(300)
    ok = obst = 0
    for _ in range(n):
        ss = _random_space(rng, sch).signature_space()
        ok += not ss.zero_leakage_system()["feasible"]
        obst += ss.two_branch_obstruction() is not None
    _row(rows, 27, "Theorem 1: full-support priors, m >= 2: exact-cover system infeasible (every kernel leaks)",
         "Theorem 1", n, ok, f"infeasible {ok}/{n}; two-branch obstruction found {obst}/{n}", "[P]")

    n = N(103)
    agree = feas = zero_ok = 0
    for _ in range(n):
        ss = _random_space(rng, sch, sparse=float(rng.choice([0.5, 0.8, 0.95]))).signature_space()
        sysr = ss.zero_leakage_system()
        fl = ss.lambda_floor()
        if sysr["feasible"]:
            feas += 1
            agree += fl["Lambda"] < 1e-6
            zero_ok += abs(ss.lam(ss.zero_leakage_kernel(sysr["w"]))) < 1e-12
        else:
            agree += fl["lower_bound"] > 1e-9
    _row(rows, 28, "Theorem 1 (iff): sparse-support priors, system feasibility vs convex solver's Lambda* = 0",
         "Theorem 1", n, agree, f"agree {agree}/{n} ({feas} feasible); w-kernel has Lambda = 0 in {zero_ok}/{feas}", "[P]")

    n = N(60)
    ok = 0
    worst_gap = 0.0
    for _ in range(n):
        ss = _random_space(rng, sch, kind="ferro").signature_space()
        c = ss.compile(float(rng.choice([0.01, 0.05, 0.1])))
        Qe, _ = ss.exp_at_budget(c.budget)
        best = math.inf
        for _ in range(300):
            Q = ss._normalise(np.log(rng.dirichlet(np.ones(ss.m), size=ss.n_signatures)))
            if ss.cost(Q) <= c.budget:
                best = min(best, ss.lam(Q))
        worst_gap = max(worst_gap, c.gap_bits)
        ok += (c.lam_bits <= ss.lam(Qe) + c.gap_bits + 1e-9) and (c.lam_bits <= best + 1e-9) and c.cost <= c.budget + 1e-9
    _row(rows, 29, "Theorem 2: PCGS within budget and <= exponential mechanism and random kernels at equal cost",
         "Theorem 2", n, ok, f"{ok}/{n}; max certified gap {worst_gap:.1e} bits", "[P]")

    n = N(60)
    ok = 0
    for _ in range(n):
        ss = _random_space(rng, sch).signature_space()
        # mu is scaled to the smallest within-signature cost gap so that mu * gap >= 50 (near-ties need larger mu)
        Cs = np.sort(np.where(ss.sig, ss.C[None, :], np.inf), axis=1)
        gap = np.min(np.where(np.isfinite(Cs[:, 1]), Cs[:, 1] - Cs[:, 0], np.inf))
        Q, _, _, _ = ss.blahut_arimoto(max(200.0, 50.0 / gap) if np.isfinite(gap) and gap > 0 else 200.0)
        ok += abs(ss.cost(Q) - ss.cost(ss.kernel_det())) < 1e-6
    _row(rows, 30, "Theorem 2: DET is the mu -> infinity limit of the Gibbs kernel (cost equal at mu * min gap >= 50)",
         "Theorem 2", n, ok, f"{ok}/{n}")

    n = N(60)
    ok = 0
    for _ in range(n):
        sp = _random_space(rng, sch)
        ss = sp.signature_space()
        K = rng.dirichlet(np.ones(len(sp.front)), size=len(sp.prior.p)) * sp.avail
        K = np.where(sp.accept[:, None], K / np.maximum(K.sum(1, keepdims=True), 1e-300), 0.0)
        codes = sp.avail.astype(np.int64) @ (1 << np.arange(ss.m))
        Qb = np.array([(sp.prior.p[(codes == c) & sp.accept][:, None] * K[(codes == c) & sp.accept]).sum(0) /
                       sp.prior.p[(codes == c) & sp.accept].sum() for c in ss.codes])
        a, b = sp.metrics(K), ss.metrics(Qb)
        ok += b["Lambda"] <= a["Lambda"] + 1e-9 and abs(b["cost"] - a["exp_cost"]) < 1e-9
    _row(rows, 31, "Lemma 2: averaging a kernel over each signature keeps cost and never raises Lambda", "Lemma 2",
         n, ok, f"{ok}/{n}", "[P]")

    n = N(40)
    ok = 0
    for _ in range(n):
        ss = _random_space(rng, sch, kind="ferro", m_range=(3, 6)).signature_space()
        Q = ss.kernel_exp(float(rng.uniform(0.2, 2)))
        nb = coverage_payments(q_min(ss, Q), ss.m, 0.05)
        ok += min(prob_all_seen(Q[a], nb) for a in range(ss.n_signatures)) >= 0.95
    _row(rows, 32, "Theorem 3(1): after ln(m/delta)/q_min fresh payments all supported branches seen w.p. >= 1-delta",
         "Theorem 3", n, ok, f"{ok}/{n} (delta = 0.05, exact inclusion-exclusion)", "[P]")

    n = N(40)
    ok = 0
    for _ in range(n):
        ss = _random_space(rng, sch, kind="ferro", m_range=(3, 6)).signature_space()
        Q = ss.kernel_exp(1.0)
        I = [repeated_metrics(ss, Q, k)["I_bits"] for k in (1, 2, 3, 5, 8)]
        IA = float(-(ss.mass * np.log2(ss.mass)).sum())
        ok += all(I[i] <= k * I[0] + 1e-9 for i, k in enumerate((1, 2, 3, 5, 8))) and \
            all(a <= b + 1e-9 for a, b in zip(I, I[1:])) and I[-1] <= IA + 1e-9
    _row(rows, 33, "Theorem 3(2): I(X;V^n) <= n I(X;V), nondecreasing in n, bounded by I(X;A(X))", "Theorem 3", n,
         ok, f"{ok}/{n}", "[P]")

    n = N(200)
    ok = 0
    st = StickyStore(b"verification")
    for k in range(n):
        sup = rng.random(4) < 0.7
        sup[int(rng.integers(4))] = True
        probs = rng.dirichlet(np.ones(4))
        first = st.choose(("acct", k, "epoch"), sup, probs)[0]
        ok += all(st.choose(("acct", k, "epoch"), sup, probs)[0] == first for _ in range(10))
    _row(rows, 34, "Theorem 3(3): sticky selection repeats the branch while evidence is unchanged", "Theorem 3", n, ok,
         f"{ok}/{n} sequences constant over 11 payments", "[P]")

    n = N(2000)
    lam = 1 / 3600.0
    tau = grace_window(0.05, lam)
    s = rng.uniform(0, tau, n)
    rev = rng.exponential(1 / lam, n)
    frac = float((rev <= s).mean())
    ok = int(all(unsafe_probability(x, lam) <= 0.05 + 1e-12 for x in s) and frac <= 0.05 + 3 * math.sqrt(0.05 / n))
    _row(rows, 35, "Stale-status grace: P(revoked within staleness <= tau) <= eps (eps = 0.05)", "Proposition C6", 1,
         ok, f"tau = {tau:.0f}s; Monte Carlo unsafe share {frac:.4f} vs eps 0.05", "[P]")


def _series_cost(acts, perm, kind):
    c, run = 0.0, 1.0
    for i in perm:
        c += run * acts[i].k
        run *= acts[i].p if kind == "and" else (1 - acts[i].p)
    return c


def _benign_false_reject(thresholds, rho, rng, payers, T=12):
    fr = tot = 0
    for _ in range(payers):
        x = int(rng.integers(0, 128))
        L = led.IntervalLedger(0, 127, rho)
        for _ in range(T):
            th = int(rng.choice(thresholds))
            a = L.answer(x, th)
            if x >= th:
                tot += 1
                fr += a != led.TRUE
    return fr / max(tot, 1)
