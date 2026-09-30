import itertools
import math

import numpy as np
import pytest

from vista import man, payment as pm, policy as pol
from vista.claims import DEFAULT_SCHEMA as S, synthetic_schema
from vista.cost import AdditiveCost
from vista.mdps_exact import solve_bruteforce, solve_frontier, solve_ilp, solve_readonce
from vista.mdps_greedy import greedy
from vista.sen import normalize

T0 = 1_788_220_800


def test_pid_deterministic_and_nfc():
    p = pm.Payment("FI1", "FI2", "Café", "B1", 1000, "usd", T0 * 1000, "TX1")
    q = pm.Payment("FI1", "FI2", "Café", "B1", 1000, "USD", T0 * 1000, "TX1")  # decomposed accent
    assert pm.pid(p) == pm.pid(q)
    assert len(pm.pid(p)) == 64


def test_payment_rejects_floats_and_bad_currency():
    with pytest.raises(ValueError):
        pm.encode(pm.Payment("a", "b", "c", "d", 10.5, "USD", 0, "t"))
    with pytest.raises(ValueError):
        pm.encode(pm.Payment("a", "b", "c", "d", 10, "US", 0, "t"))


def test_policy_parse_blueprint_example():
    node = pol.parse("(auth_strength >= A3 AND credential_age >= 7d) OR "
                     "(auth_strength >= A2 AND account_age >= 365d AND device_continuity = established)", S)
    x = [0] * S.n
    x[S.index("auth_strength")] = 3
    x[S.index("credential_age")] = 2
    assert pol.evaluate(node, x)
    x[S.index("credential_age")] = 1
    assert not pol.evaluate(node, x)
    fr = pol.frontier(node, S.n)
    assert len(fr) == 2


def test_parse_errors():
    with pytest.raises(SyntaxError):
        pol.parse("auth_strength < A3", S)
    with pytest.raises(KeyError):
        pol.parse("nonexistent >= 1", S)


def test_frontier_matches_bruteforce(rng):
    sch = synthetic_schema(3, 4)
    st = sch.states()
    for _ in range(80):
        node = pol.random_tree(sch, int(rng.integers(1, 4)), 3, rng)
        a = {tuple(r) for r in pol.frontier(node, sch.n).tolist()}
        b = {tuple(r) for r in pol.frontier_bruteforce(node, st).tolist()}
        assert a == b


def test_mdps_solvers_agree(rng):
    sch = synthetic_schema(6, 4)
    for _ in range(60):
        node = pol.random_tree(sch, int(rng.integers(1, 3)), 3, rng, read_once=True)
        x = tuple(int(v) for v in rng.integers(0, 4, 6))
        w = AdditiveCost.random_isotone(sch.dims, rng)
        b = solve_bruteforce(node, x, w)
        i = solve_ilp(node, x, w)
        d = solve_readonce(node, x, w)
        f = solve_frontier(pol.frontier(node, sch.n), x, w)
        if not b.feasible:
            assert not i.feasible and not d.feasible and not f.feasible
            continue
        for r in (i, d, f):
            assert math.isclose(r.cost, b.cost, abs_tol=1e-9)
        g = greedy(node, x, w)
        assert g.cost >= b.cost - 1e-9 and pol.evaluate(node, g.profile)


def test_sen_conservative_and_invalidation():
    pid = "p" * 64
    now = T0 + 100
    ev = [{"type": "auth", "method": "fido2_passkey", "ts": now - 10, "tx_bound": True, "pid": pid},
          {"type": "credential", "last_reset_ts": now - 30 * 86400}]
    x = normalize(ev, pid, now).profile
    assert x[S.index("auth_strength")] == 3 and x[S.index("dynamic_binding")] == 1
    assert x[S.index("auth_freshness")] == 3
    ev[1]["last_reset_ts"] = now - 5  # reset after authentication invalidates auth claims
    r = normalize(ev, pid, now)
    assert r.profile[S.index("auth_strength")] == 0 and r.invalidated
    ev2 = [{"type": "auth", "method": "fido2_passkey", "ts": now - 10, "tx_bound": True, "pid": "other"}]
    assert normalize(ev2, pid, now).profile[S.index("dynamic_binding")] == 0
    ev3 = [{"type": "auth", "method": "unknown_method", "ts": now - 10}]
    assert normalize(ev3, pid, now).profile[S.index("auth_strength")] == 0


def test_man_toy_example():
    acts = [man.Action("passkey", 2, .9), man.Action("rebinding", 5, .6), man.Action("otp", 1.5, .85),
            man.Action("selfie", 4, .7)]
    nd = man.AOr((man.AAnd((man.ALeaf(0), man.ALeaf(1))), man.AAnd((man.ALeaf(2), man.ALeaf(3)))))
    r = man.evaluate_strategy(nd, acts, man.ratio_strategy(nd, acts))
    o = man.evaluate_strategy(nd, acts, man.optimal_dp(nd, acts))
    assert abs(r["exp_friction"] - 7.41) < 0.01 and abs(r["success_prob"] - 0.814) < 1e-3
    assert abs(o["exp_friction"] - r["exp_friction"]) < 1e-9
    mc = man.simulate(nd, acts, man.ratio_strategy(nd, acts), np.random.default_rng(0), 20000)
    assert abs(mc["mc_exp_friction"] - 7.41) < 0.15


def test_kleene_matches_bruteforce(rng):
    for _ in range(40):
        acts, nd = man.random_readonce_actions(rng, 5, 3)
        for s in itertools.product((0, 1, 2), repeat=5):
            v = man.eval3(nd, s)
            lo = man.eval3(nd, tuple(2 if t == 0 else t for t in s))
            hi = man.eval3(nd, tuple(1 if t == 0 else t for t in s))
            if v is None:
                assert lo != hi
            else:
                assert lo == hi == v
