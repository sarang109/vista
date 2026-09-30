import math

import numpy as np

from vista import audit, group_cost as gc, ledger as led, leakage as lk
from vista import policy_extraction as pe
from vista.prior import Prior
from vista.selection import blueprint_toy_space


def test_selection_identity_and_floor():
    _, _, sp = blueprint_toy_space()
    for K in (sp.kernel_det(), sp.kernel_exp(0.0), sp.kernel_exp(1.0)):
        m = sp.metrics(K)
        assert abs(m["identity_gap"]) < 1e-9
        assert m["E_L"] >= sp.L_min - 1e-9
    assert math.isinf(sp.eps_sel(sp.kernel_det()))
    assert abs(sp.eps_sel(sp.kernel_exp(0.0)) - math.log(3)) < 1e-9
    eps = sp.min_feasible_eps(iters=25)
    assert abs(eps - math.log(sp.a_max())) < 1e-3
    K = sp.kernel_lp(1.5)
    assert sp.eps_sel(K) <= 1.5 + 1e-6


def test_ledger_invariant(rng):
    for rho in (4, 16):
        for strat in ("binary", "random", "scan"):
            for x in rng.integers(0, 128, 30):
                r = led.probe(128, int(x), strat, rho, rng)
                assert r["final_width"] >= rho and r["consistent"]


def test_ocgd_matches_bruteforce(rng):
    S = led.survival(np.full(128, 1 / 128))
    for _ in range(20):
        th = sorted(rng.choice(np.arange(1, 128), 5, replace=False).tolist())
        mu = rng.uniform(0.2, 2, 5).tolist()
        for k in (1, 2, 3):
            assert abs(led.ocgd(th, mu, k, S)[1] - led.ocgd_bruteforce(th, mu, k, S)[1]) < 1e-9


def test_fkg_submodular(rng):
    pr = Prior.random_ising((3, 3, 3, 2), rng, "ferro")
    assert pr.is_log_supermodular()[0]
    assert lk.submodularity_check(pr, rng, 400)["submod_violations"] == 0


def test_group_cost(rng):
    inst = gc.adversarial_instance(8)
    assert gc.myopic_greedy(inst)[1] == 8 and gc.density_greedy(inst)[1] == 2
    for _ in range(30):
        inst = gc.random_instance(rng, 5, 4)
        ce = gc.exact_ilp(inst)[1]
        assert abs(ce - gc.exact_enum(inst)[1]) < 1e-9
        assert gc.density_greedy(inst)[1] <= gc.harmonic(5) * ce + 1e-9


def test_policy_extraction(rng):
    terms = pe.random_kdnf(rng, 8, 3, 3)
    r = pe.extract_dnf(pe.PolicyOracle(8, terms))
    assert set(r["terms"]) == set(terms) and r["complete"]
    r = pe.extract_dnf(pe.PolicyOracle(8, terms, budget=5))
    assert r["queries"] <= 5


def test_audit(rng):
    assert abs(audit.q_star(20, 400, 0.9) - 20 / (0.9 * 420)) < 1e-12
    leaves = [bytes([i]) * 10 for i in range(7)]
    t = audit.MerkleTree(leaves)
    assert all(audit.MerkleTree.verify(l, t.proof(i), t.root) for i, l in enumerate(leaves))
    assert not audit.MerkleTree.verify(b"x", t.proof(0), t.root)
    log = audit.TransparencyLog()
    for i in range(5):
        log.append(f"{i}", "FI", t.root, ["c"], i)
    assert log.verify_chain()
    log.entries[2]["claims"] = ["forged"]
    assert not log.verify_chain()
