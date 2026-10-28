"""E28: does a risk-bounded stale-status grace cap exposure during a registry outage?

The v1 registry-outage scenario (E10 timing) is replayed through the protocol path twice: fail-closed (v1) and
with a grace window tau = -ln(1 - eps)/lam_r for the micro and standard tiers (high and very_high stay
fail-closed). Revocations during the outage are drawn per issuer key from the Poisson model; a payment is an
unsafe accept when it is accepted on a key revoked before it was verified (the verifier cannot see this
while the registry is down). Exposure is checked against the bound eps x (value accepted under grace) both on
the protocol run and by Monte Carlo over many revocation draws.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict

import numpy as np

from .. import payment as pm
from ..trust import grace_window, unsafe_probability
from ..util import write_csv
from ..vsac import VerifyConfig
from . import exp_e2e

YEAR = 365 * 86_400
LOW = ("micro", "standard")


def _timeline(cfg, tau: float | None, label: str):
    wc = dict(cfg["world"], n_banks=min(cfg["world"]["n_banks"], 12), n_payers=min(cfg["world"]["n_payers"], 20_000))
    n_req = cfg.get("e28_payments", cfg.get("e10_payments", 6000))
    wc["n_payments"] = exp_e2e.source_size(cfg["world"]) if wc.get("source") == "paysim" else n_req
    world, source = exp_e2e.build_world(wc)
    stride = max(1, source.n // n_req)
    pay = source.take(np.arange(0, source.n, stride)[:n_req], world)
    rng = np.random.default_rng(wc["seed"] + 28)
    n = len(pay["ts"])
    t_start = int(pay["ts"][0])
    OUT_A, OUT_B = 10, 25
    vc = VerifyConfig(status_grace=(lambda p: int(tau) if world.tier(p.amount_minor) in LOW else None) if tau else None)
    per_min = defaultdict(Counter)
    accepted = []  # (issuer key id, verify time, staleness, amount usd, tier is low)
    for k in range(n):
        now = t_start + int(k * 0.5)
        mn = int(k * 0.5 // 60)
        world.trust.online = not (OUT_A <= mn < OUT_B)
        p = world.payment(pay, k).with_(timestamp_ms=now * 1000)
        s, r = int(pay["sender"][k]), int(pay["receiver"][k])
        pid = pm.pid(p)
        from ..simulator import workload as wl
        ev = wl.evidence_for(world.pop, int(pay["payer"][k]), now - 1, int(pay["scenario"][k]), pid,
                             world.arche["archetypes"][world.banks[s].archetype], rng)
        snd = world.sender(s, "DET", "signed")
        rcv = world.receiver(r, vc)
        node = world.policy_for(r, p.amount_minor)
        nonce = rcv.verifier.nonces.issue(snd.id, 1, now, rng)[0]
        o = snd.build(p, ev, node, nonce, now)
        d = rcv.decide(p, o.blob, now)
        low = world.tier(p.amount_minor) in LOW
        c = per_min[mn]
        grp = "low" if low else "high"
        c[f"sat_{grp}"] += o.satisfiable
        c[f"acc_{grp}"] += (d["decision"] == "ACCEPT") and o.satisfiable
        c["payments"] += 1
        if d["decision"] == "ACCEPT" and OUT_A <= mn < OUT_B:
            accepted.append((snd.issuer.key_id, now, now - world.trust.snapshot_time, p.amount_minor / 100, low))
    rows = []
    for mn in sorted(per_min):
        c = per_min[mn]
        rows.append({"config": label, "minute": mn, "phase": "registry_outage" if OUT_A <= mn < OUT_B else "normal",
                     "payments": c["payments"],
                     "availability_low_tiers": c["acc_low"] / max(c["sat_low"], 1),
                     "availability_high_tiers": c["acc_high"] / max(c["sat_high"], 1)})
    return rows, accepted


def _exposure(accepted, lam: float, eps: float, tau: float, trials: int, rng, label: str) -> dict:
    """Unsafe accepts under grace: analytic expectation, the eps bound, and Monte Carlo revocation draws."""
    grace = [a for a in accepted if a[4]]
    if not grace:
        return {"setting": label, "grace_accepts": 0}
    keys = sorted({a[0] for a in grace})
    kidx = {k: i for i, k in enumerate(keys)}
    stale = np.array([a[2] for a in grace], dtype=float)
    val = np.array([a[3] for a in grace])
    ki = np.array([kidx[a[0]] for a in grace])
    p_unsafe = np.array([unsafe_probability(s, lam) for s in stale])
    mc_cnt, mc_val = [], []
    for _ in range(trials):
        rev = rng.exponential(1 / lam, size=len(keys))  # revocation time after the last snapshot
        bad = rev[ki] <= stale
        mc_cnt.append(bad.sum())
        mc_val.append(val[bad].sum())
    mc_cnt, mc_val = np.array(mc_cnt, float), np.array(mc_val)
    return {"setting": label, "eps": eps, "revocation_rate_per_year": lam * YEAR, "tau_s": tau,
            "grace_accepts": len(grace), "grace_accepted_value_usd": float(val.sum()),
            "max_staleness_s": float(stale.max()), "expected_unsafe_accepts": float(p_unsafe.sum()),
            "bound_unsafe_accepts": eps * len(grace), "expected_unsafe_value_usd": float((p_unsafe * val).sum()),
            "bound_unsafe_value_usd": eps * float(val.sum()), "mc_trials": trials,
            "mc_mean_unsafe_accepts": float(mc_cnt.mean()),
            "mc_ci95_unsafe_accepts": float(1.96 * mc_cnt.std(ddof=1) / math.sqrt(trials)),
            "mc_mean_unsafe_value_usd": float(mc_val.mean()),
            "bound_holds": bool(mc_val.mean() <= eps * val.sum() + 1.96 * mc_val.std(ddof=1) / math.sqrt(trials))}


def run_e28(cfg, out):
    g = cfg.get("grace", {})
    eps, rate_year = g.get("eps", 1e-4), g.get("revocations_per_key_year", 1.0)
    tau = grace_window(eps, rate_year / YEAR)
    rows, _ = _timeline(cfg, None, "v1_fail_closed")
    rows_g, acc = _timeline(cfg, tau, f"grace_low_tiers_tau={tau:.0f}s")
    rng = np.random.default_rng(cfg.get("seed", 1) + 280)
    trials = cfg.get("e28_trials", 4000)
    expo = [_exposure(acc, rate_year / YEAR, eps, tau, trials, rng, "deployment (eps=1e-4; 1 revocation per key-year)")]
    # stress setting: revocations frequent enough that unsafe accepts actually occur within the outage
    eps_s = g.get("stress_eps", 0.05)
    lam_s = -math.log1p(-eps_s) / g.get("stress_tau_s", 900)
    expo.append(_exposure(acc, lam_s, eps_s, g.get("stress_tau_s", 900), trials, rng,
                          f"stress (eps={eps_s}; tau={g.get('stress_tau_s', 900)}s)"))
    return [write_csv(os.path.join(out, "e28_grace_timeline.csv"), rows + rows_g),
            write_csv(os.path.join(out, "e28_grace_exposure.csv"), expo)]
