import csv
import json
import zipfile

import numpy as np

from vista import payment as pm
from vista.governor import GovernorConfig, evidence_epoch
from vista.simulator import rba
from vista.simulator.source import make_source
from vista.simulator.world import World


def _run(world, cfg, n=600, seed=1):
    """cfg=None runs the v1 DET sender; otherwise the governor (mode IDG) with that configuration."""
    src = make_source(world, {"n_payments": n, "seed": 7, "days": 7})
    pay = src.take(np.arange(n), world)
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        p = world.payment(pay, j)
        s, r = int(pay["sender"][j]), int(pay["receiver"][j])
        pid = pm.pid(p)
        now = int(pay["ts"][j]) + 1
        ev = world.evidence(pay, j, pid, rng)
        snd = world.sender(s, "DET") if cfg is None else world.sender(s, "IDG", gov=cfg)
        rcv = world.receiver(r)
        node = world.policy_for(r, p.amount_minor)
        o = snd.build(p, ev, node, rcv.verifier.nonces.issue(snd.id, 1, now, rng)[0], now)
        out.append((o, dict(snd.last_info), rcv.decide(p, o.blob, now)))
    return out


def test_governor_det_matches_v1_and_verifies():
    res = _run(World(8, 2000, 7, 2000), GovernorConfig(name="det", kernel="det", sticky=False))
    ref = _run(World(8, 2000, 7, 2000), None)
    assert all(d["verify_reason"] == "ok" for _, _, d in res)
    assert [r[0].disclosed for r in res] == [r[0].disclosed for r in ref]
    assert [r[2]["decision"] for r in res] == [r[2]["decision"] for r in ref]


def test_full_governor_components():
    w = World(8, 2000, 7, 2000)
    cfg = GovernorConfig(name="idg", ledger_rho={"account_age": 2}, fixed_charge={"weight": 0.5, "F": {}, "per_claim": 0.25},
                         budget=True, probe_limit=1, audit=True)
    res = _run(w, cfg, 800)
    events = {info["event"] for _, info, _ in res}
    assert {"first", "unsat"} <= events
    for o, info, d in res:
        if info["event"] in ("first", "hit", "redraw"):
            assert d["decision"] == "ACCEPT"
            assert "receipt" in info and info["fixed_charge"] > 0
    govs = [s.governor for k, s in w._senders.items() if k[-1] == "idg"]
    assert sum(g.log.count for g in govs) == sum(1 for _, i, _ in res if "receipt" in i)
    assert sum(g.budget.suppressed for g in govs) > 0  # probe_limit=1 suppresses repeated step-up probes


def test_evidence_epoch_changes_on_reset():
    ev = [{"type": "credential", "last_reset_ts": 10}, {"type": "device", "first_seen_ts": 5, "bound": True},
          {"type": "auth", "method": "totp", "ts": 99}]
    ev2 = [dict(e) for e in ev]
    ev2[2]["ts"] = 120  # a new authentication does not change the epoch
    assert evidence_epoch(ev) == evidence_epoch(ev2)
    ev2[0]["last_reset_ts"] = 50
    assert evidence_epoch(ev) != evidence_epoch(ev2)


def test_rba_cache_from_small_csv(tmp_path):
    header = ["index", "Login Timestamp", "User ID", "Round-Trip Time [ms]", "IP Address", "Country", "Region", "City",
              "ASN", "User Agent String", "Browser Name and Version", "OS Name and Version", "Device Type",
              "Login Successful", "Is Attack IP", "Is Account Takeover"]
    rows = []
    t0 = 1_580_000_000
    for u in range(30):
        for k in range(12):
            ts = t0 + u * 3600 + k * 5 * 86400
            dev = ("Chrome 80.1", "Windows 10") if k % 4 else ("Firefox 70", "Linux")
            rows.append([len(rows), f"2020-02-{1 + (ts - t0) // 86400 % 28:02d} 10:00:00.000", str(-1000 - u), "", "1.2.3.4",
                         "NO", "-", "-", "29695", "ua", dev[0], dev[1], "desktop", "True", "False", str(k == 11)])
    path = tmp_path / "rba.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    zp = tmp_path / "rba.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(path, "rba-dataset.csv")
    meta = rba.build_cache(str(zp), str(tmp_path / "cache"), user_fraction=1.0, warmup_days=0)
    assert meta["users_sampled"] == 30 and meta["rows_sampled"] == 360
    L = rba.load_states(str(tmp_path / "cache"))
    A = rba.load_states(str(tmp_path / "cache"), "attack")
    assert len(A) == 30 and len(L) == 330
    assert L[:, 0].max() <= 3 and (L[:, 1] >= 0).all()
    pr = rba.empirical_prior(L)
    assert abs(pr.p.sum() - 1) < 1e-12
    assert len(rba.pairwise_association(L)) == 6
    json.dumps(rba.lattice_condition(pr))
