import math

import numpy as np
import pytest

from vista import policy as pol
from vista.claims import synthetic_schema
from vista.pcgs import SignatureSpace
from vista.prior import Prior
from vista.repeated import StickyStore, coverage_payments, prob_all_seen, q_min, repeated_metrics, sticky_metrics
from vista.selection import SelectionSpace, blueprint_toy_space
from vista.trust import TrustError, TrustRegistry, grace_window, unsafe_probability


def _space(rng, sparse=0.0, kind="ferro"):
    sch = synthetic_schema(3, 4)
    while True:
        node = pol.random_dnf(sch, int(rng.integers(2, 6)), int(rng.integers(1, 3)), rng)
        fr = pol.frontier(node, sch.n)
        if len(fr) < 2:
            continue
        pr = Prior.random_ising(sch.dims, rng, kind)
        if sparse:
            p = np.where(rng.random(len(pr.p)) < sparse, 0.0, pr.p)
            if p.sum() == 0:
                continue
            pr.p = p / p.sum()
        w = fr.sum(1) + rng.normal(0, 1, len(fr))
        sp = SelectionSpace(pr, fr, w - w.min() + 1)
        if sp.pA > 0:
            return sp


def test_signature_metrics_match_state_level(rng):
    for _ in range(20):
        sp = _space(rng)
        ss = sp.signature_space()
        for Q in (ss.kernel_det(), ss.kernel_exp(0.7)):
            a, b = ss.metrics(Q), sp.metrics(sp.expand(ss, Q))
            assert abs(a["Lambda"] - b["Lambda"]) < 1e-9
            assert abs(a["cost"] - b["exp_cost"]) < 1e-9
            assert abs(a["I_bits"] - a["E_L_bits"] - a["Lambda"]) < 1e-12


def test_no_free_selection_full_support(rng):
    for _ in range(30):
        ss = _space(rng).signature_space()
        assert not ss.zero_leakage_system()["feasible"]
        assert ss.two_branch_obstruction() is not None
        assert ss.lambda_floor()["lower_bound"] > 0


def test_zero_leakage_iff_system_feasible(rng):
    seen_feasible = 0
    for _ in range(40):
        ss = _space(rng, sparse=0.9, kind="mixed").signature_space()
        sysr = ss.zero_leakage_system()
        fl = ss.lambda_floor()
        if sysr["feasible"]:
            seen_feasible += 1
            assert abs(ss.lam(ss.zero_leakage_kernel(sysr["w"]))) < 1e-12
            assert fl["Lambda"] < 1e-6
        else:
            assert fl["lower_bound"] > 0
    assert seen_feasible > 0


def test_pcgs_is_optimal_within_budget(rng):
    for _ in range(10):
        ss = _space(rng).signature_space()
        c = ss.compile(0.05)
        assert c.cost <= c.budget + 1e-9
        assert c.gap_bits < 1e-6
        Qe, _ = ss.exp_at_budget(c.budget)
        assert c.lam_bits <= ss.lam(Qe) + 1e-9
        for _ in range(200):
            Q = ss._normalise(np.log(rng.dirichlet(np.ones(ss.m), size=ss.n_signatures)))
            if ss.cost(Q) <= c.budget:
                assert c.lam_bits <= ss.lam(Q) + 1e-9


def test_pcgs_state_kernel_and_det_limit():
    _, _, sp = blueprint_toy_space()
    K = sp.kernel_pcgs(0.02)
    m = sp.metrics(K)
    det = sp.metrics(sp.kernel_det())
    assert m["exp_cost"] <= det["exp_cost"] * 1.02 + 1e-9
    assert m["Lambda"] < det["Lambda"]
    ss = sp.signature_space()
    Q, _, _, _ = ss.blahut_arimoto(500.0)
    assert abs(ss.cost(Q) - ss.cost(ss.kernel_det())) < 1e-6
    assert not sp.zero_leakage_feasible()


def test_repeated_fresh_grows_sticky_constant():
    _, _, sp = blueprint_toy_space()
    ss = sp.signature_space()
    Q = ss.kernel_exp(1.0)
    lam = [repeated_metrics(ss, Q, n)["Lambda"] for n in (1, 2, 3, 6)]
    assert abs(lam[0] - ss.lam(Q)) < 1e-9
    assert all(a < b for a, b in zip(lam, lam[1:]))
    assert lam[2] > ss.lam(ss.kernel_det())  # fresh RAND passes DET by the third payment on the toy space
    assert sticky_metrics(ss, Q, 6)["Lambda"] == pytest.approx(lam[0])
    I1 = repeated_metrics(ss, Q, 1)["I_bits"]
    assert repeated_metrics(ss, Q, 5)["I_bits"] <= 5 * I1 + 1e-9
    mc = repeated_metrics(ss, Q, 4, max_rows=1, mc_samples=60_000, rng=np.random.default_rng(3))
    assert abs(mc["Lambda"] - repeated_metrics(ss, Q, 4)["Lambda"]) < 0.02


def test_coverage_bound():
    _, _, sp = blueprint_toy_space()
    ss = sp.signature_space()
    Q = ss.kernel_exp(0.5)
    nb = coverage_payments(q_min(ss, Q), ss.m, 0.05)
    assert all(prob_all_seen(Q[a], nb) >= 0.95 for a in range(ss.n_signatures))
    assert prob_all_seen(np.array([0.5, 0.5]), 1) == pytest.approx(0.0)
    assert prob_all_seen(np.array([0.5, 0.5]), 2) == pytest.approx(0.5)


def test_sticky_store_redraws_only_when_unsupported():
    st = StickyStore(b"k")
    probs = np.array([0.3, 0.3, 0.4])
    v, ev = st.choose("key", np.array([True, True, True]), probs)
    assert ev == "first"
    for _ in range(5):
        assert st.choose("key", np.array([True, True, True]), probs) == (v, "hit")
    sup = np.ones(3, dtype=bool)
    sup[v] = False
    v2, ev2 = st.choose("key", sup, probs)
    assert ev2 == "redraw" and v2 != v
    # reproducible from the key alone (audit)
    assert StickyStore(b"k").choose("key", np.array([True, True, True]), probs)[0] == v


def test_grace_window_and_registry():
    lam = 1 / (365 * 86400)
    tau = grace_window(1e-4, lam)
    assert unsafe_probability(tau, lam) == pytest.approx(1e-4)
    from vista.crypto.signed import SigningKey
    reg = TrustRegistry(max_status_staleness_s=300)
    k = SigningKey()
    reg.register("B1", k, "issuer", 0, 10**9)
    reg.resolve(k.key_id, 1000)
    reg.online = False
    with pytest.raises(TrustError):
        reg.resolve(k.key_id, 1000 + 301)
    assert reg.resolve(k.key_id, 1000 + 301, max_staleness=int(tau)).institution == "B1"


def test_signature_space_from_states_aggregates():
    avail = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    p = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    ss = SignatureSpace.from_states(avail, p, np.array([1.0, 2.0]))
    assert ss.n_signatures == 3
    assert math.isclose(ss.mass.sum(), 1.0)
    assert np.allclose(sorted(ss.mass), sorted(np.array([0.3, 0.3, 0.3]) / 0.9))
