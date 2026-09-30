"""End-to-end evaluation on a realistic interbank transaction stream (H1-H5, E3, E7, E8, E10).

Every payment goes through the real protocol path: SEN -> MDPS/selection -> VSAC generation (real
Ed25519 / selective disclosure / hash-chain / OR-proof crypto) -> simulated rail latency ->
verification -> receiver policy -> MAN-S step-up rounds executed by the (simulated) payer.
Work is sharded across processes; each worker rebuilds the same deterministic World.
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import payment as pm
from ..claims import DEFAULT_SCHEMA as S, Schema
from ..crypto.signed import SigningKey
from ..policy import evaluate, evaluate_many
from ..sen import normalize
from ..simulator import workload as wl
from ..simulator.world import ACTION_SECONDS, ACTIONS, World
from ..util import bootstrap_ci, canonical_json, percentiles, write_csv

SCEN = ("normal", "ato_like", "scam_like")
MAX_ROUNDS = 5
_G: dict = {}


def build_world(wc: dict):
    world = World(wc["n_banks"], wc["n_payers"], wc["seed"], wc.get("prior_sample", 20_000))
    rng = np.random.default_rng(wc["seed"] + 100)
    ps = wc.get("paysim_path")
    if ps and not os.path.isabs(ps):
        from ..simulator.world import ROOT
        ps = os.path.join(ROOT, ps)
    if ps and os.path.exists(ps):
        pay = wl.load_paysim(ps, world.pop, world.banks, wc["n_payments"], rng)
    else:
        pay = wl.generate_payments(world.pop, world.banks, wc["n_payments"], wc.get("days", 7), rng)
    return world, pay


def _init(wc):
    _G["world"], _G["pay"] = build_world(wc)
    _G["wc"] = wc
    _G["lmin"] = {}


def _lmin(world, node):
    k = id(node)
    if k not in _G["lmin"]:
        m = float(world.prior.p[evaluate_many(node, world.prior.states)].sum())
        _G["lmin"][k] = -math.log2(m) if m > 0 else math.inf
    return _G["lmin"][k]


def _risk_label(x) -> str:
    """B1: a sender's opaque risk label from its own scoring (weights unknown to the receiver)."""
    u = [x[i] / max(S.attributes[i].d, 1) for i in range(S.n)]
    w = [0.10, 0.25, 0.10, 0.15, 0.10, 0.05, 0.05, 0.15, 0.05]
    sc = sum(a * b for a, b in zip(u, w))
    return "LOW" if sc >= 0.62 else "MEDIUM" if sc >= 0.42 else "HIGH"


def _process(task):
    start, stop, mode, backend, detail = task
    world, pay, wc = _G["world"], _G["pay"], _G["wc"]
    t_start = time.time()
    rows = []
    fixed = world.sender(0, "FIXED").fixed_bundle
    for j in range(start, stop):
        rng = np.random.default_rng([wc["seed"], j])
        p = world.payment(pay, j)
        s, r = int(pay["sender"][j]), int(pay["receiver"][j])
        payer, scen = int(pay["payer"][j]), int(pay["scenario"][j])
        pid = pm.pid(p)
        now = int(pay["ts"][j]) + 1
        ev = world.evidence(pay, j, pid, rng)
        snd = world.sender(s, mode, backend)
        rcv = world.receiver(r)
        node = world.policy_for(r, p.amount_minor)
        nonce = rcv.verifier.nonces.issue(snd.id, 1, now, rng)[0]
        t = time.perf_counter()
        out = snd.build(p, ev, node, nonce, now)
        gen_ms = (time.perf_counter() - t) * 1e3
        net_ms = float(rng.lognormal(math.log(12.0), 0.5))  # simulated one-way rail latency
        d = rcv.decide(p, out.blob, now)
        initial = d["decision"]
        final, rounds, friction, user_s = initial, 0, 0.0, 0
        failed: set = set()
        plan = d.get("plan")
        u = rng.random(len(ACTIONS))  # pre-drawn payer outcomes, shared with the blanket counterfactual
        ok_a = [world.can_complete(payer, s, a, scen) and u[k] < a.p for k, a in enumerate(ACTIONS)]
        ev0 = list(ev)
        # the payer stays in the step-up dialogue while the residual policy is still satisfiable
        while final in ("STEP_UP", "HOLD") and plan and plan["p_star"] > 0 and plan["first_action"] is not None \
                and rounds < MAX_ROUNDS:
            act = ACTIONS[plan["first_action"]]
            rounds += 1
            friction += act.k
            user_s += ACTION_SECONDS[act.name]
            now2 = now + user_s
            if ok_a[plan["first_action"]]:
                ev = world.apply_action(ev, act, pid, now2, s)
            else:
                failed.add(act.name)
            n2 = rcv.verifier.nonces.issue(snd.id, 1, now2, rng)[0]
            o2 = snd.build(p, ev, node, n2, now2)
            d2 = rcv.decide(p, o2.blob, now2, frozenset(failed))
            final, plan = d2["decision"], d2.get("plan")
        if final == "STEP_UP":
            final = "HOLD"
        bl = {}
        if initial in ("STEP_UP", "HOLD"):  # B6 blanket step-up: strongest control first, until satisfied or exhausted
            prof_b = world.arche["archetypes"][world.banks[s].archetype]["profile"]
            evb, fb, rb, tb = list(ev0), 0.0, 0, 0
            for k, a in enumerate(ACTIONS):
                if evaluate(node, normalize(evb, pid, now + tb, prof_b, S).profile):
                    break
                rb += 1
                fb += a.k
                tb += ACTION_SECONDS[a.name]
                if ok_a[k]:
                    evb = world.apply_action(evb, a, pid, now + tb, s)
            bl = {"blanket_accept": evaluate(node, normalize(evb, pid, now + tb, prof_b, S).profile),
                  "blanket_friction": fb, "blanket_rounds": rb, "blanket_user_seconds": tb}
        x = out.x
        ideal = evaluate(node, x)
        tier = world.tier(p.amount_minor)
        label = _risk_label(x)
        b1 = label == "LOW" or (label == "MEDIUM" and tier in ("micro", "standard"))
        disclosed = out.disclosed
        fr, w, _ = snd.cache.get(node)
        det = world.sender(s, "DET").select(x, fr, w)
        leak_sent = _lmin(world, node) if disclosed is None else world.surprisal(tuple(disclosed))
        row = {
            "tx_idx": j, "tx_id": p.tx_id, "ts": int(pay["ts"][j]), "sender_fi": p.sender_fi,
            "receiver_fi": p.receiver_fi, "sender_archetype": world.banks[s].archetype,
            "receiver_stance": world.banks[r].stance, "amount_usd": p.amount_minor / 100, "tier": tier,
            "scenario": SCEN[scen], "mode": mode, "backend": backend or "bank_default",
            "profile_x": "".join(map(str, x)), "policy_satisfiable": ideal,
            "claims_disclosed": 0 if disclosed is None else sum(1 for v in disclosed if v > 0),
            "leakage_bits": leak_sent, "capsule_bytes": len(out.blob) if out.blob else 0,
            "gen_ms": gen_ms, "net_ms": net_ms, "verify_ms": d["verify_ms"], "policy_ms": d.get("policy_ms", 0.0),
            "man_ms": d.get("man_ms", 0.0),
            "e2e_ms": gen_ms + net_ms + d["verify_ms"] + d.get("policy_ms", 0.0) + d.get("man_ms", 0.0),
            "verify_reason": d["verify_reason"], "decision_initial": initial, "decision_final": final,
            "stepup_rounds": rounds, "stepup_friction": friction, "stepup_user_seconds": user_s,
            "deficit": (d.get("plan") or {}).get("deficit", 0.0), "p_star": (d.get("plan") or {}).get("p_star", 1.0),
            "vista_matches_receiver_policy": (initial == "ACCEPT") == ideal,
            "b0_metadata_accept": tier == "micro", "b1_label": label, "b1_accept": b1,
            "blanket_accept": bl.get("blanket_accept", ""), "blanket_friction": bl.get("blanket_friction", ""),
            "blanket_rounds": bl.get("blanket_rounds", ""), "blanket_user_seconds": bl.get("blanket_user_seconds", ""),
        }
        if detail:
            row.update({
                "leak_B2_raw_bits": -math.log2(world.prior.p[world.prior.index_of(x)]),
                "leak_B4_all_bits": world.surprisal(tuple(x)),
                "leak_B3_fixed_bits": world.surprisal(tuple(min(a, b) for a, b in zip(fixed, x))),
                "leak_DET_bits": world.surprisal(det) if det is not None else math.nan,
                "leak_BOP_floor_bits": _lmin(world, node) if ideal else math.nan,
                "claims_B4_all": sum(1 for v in x if v > 0),
                "claims_B3_fixed": sum(1 for a, b in zip(fixed, x) if min(a, b) > 0),
                "claims_DET": sum(1 for v in det if v > 0) if det is not None else 0,
                "raw_evidence_bytes": len(canonical_json(ev)) + 64,
            })
        rows.append(row)
    return rows, t_start, time.time()


def _run_pool(wc, tasks, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(wc,)) as ex:
        list(ex.map(_warm, range(workers * 2)))
        t0 = time.time()
        res = list(ex.map(_process, tasks))
        wall = time.time() - t0
    rows = [r for rr, _, _ in res for r in rr]
    return rows, wall


def _warm(_):
    time.sleep(0.2)
    return os.getpid()


def _tasks(n, chunk, mode, backend, detail, start=0):
    return [(a, min(a + chunk, n), mode, backend, detail) for a in range(start, n, chunk)]


# ------------------------------------------------------------------------------------------------ main runs
def run(cfg, out):
    wc = cfg["world"]
    workers = cfg.get("workers", os.cpu_count() or 4)
    n = wc["n_payments"]
    files = []
    # dump the raw workload for inspection / reuse
    world, pay = build_world(wc)
    wl.dump_workload_csv(os.path.join(out, "workload_transactions.csv"), pay, world.banks)
    files.append(os.path.join(out, "workload_transactions.csv"))
    files.append(write_csv(os.path.join(out, "banks.csv"), [
        {"bank": b.id, "archetype": b.archetype, "market_share": b.share, "receiver_stance": b.stance,
         "default_backend": world.arche["archetypes"][b.archetype]["backend"]} for b in world.banks]))

    # 1) main stream: DET MDPS, bank-default backends, full detail
    rows, wall = _run_pool(wc, _tasks(n, cfg.get("chunk", 2000), "DET", None, True), workers)
    rows.sort(key=lambda r: r["tx_idx"])
    files.append(write_csv(os.path.join(out, "e2e_transactions.csv"), rows))
    files += _summaries(rows, wall, workers, out)

    # 2) E3 disclosure modes x backends on a subsample (bytes/latency/leakage per mode)
    m = min(n, cfg.get("e3_payments", 20_000))
    e3 = []
    for mode, backend in [("ALL", "signed"), ("FIXED", "signed"), ("DET", "signed"), ("RAND", "signed"),
                          ("DET", "sd"), ("DET", "hashchain"), ("BOP", "bop")]:
        rr, wall_m = _run_pool(wc, _tasks(m, cfg.get("chunk", 2000), mode, None if mode == "BOP" else backend, False),
                               workers)
        sat = [r for r in rr if r["policy_satisfiable"]]
        e3.append({"mode": mode, "backend": backend, "payments": len(rr),
                   "mean_claims_disclosed_when_satisfiable": float(np.mean([r["claims_disclosed"] for r in sat])),
                   "mean_leakage_bits_when_satisfiable": float(np.mean([r["leakage_bits"] for r in sat])),
                   "mean_capsule_bytes": float(np.mean([r["capsule_bytes"] for r in rr])),
                   "gen_p50_ms": percentiles([r["gen_ms"] for r in rr])["p50"],
                   "gen_p99_ms": percentiles([r["gen_ms"] for r in rr])["p99"],
                   "verify_p50_ms": percentiles([r["verify_ms"] for r in rr])["p50"],
                   "verify_p99_ms": percentiles([r["verify_ms"] for r in rr])["p99"],
                   "accept_rate_initial": float(np.mean([r["decision_initial"] == "ACCEPT" for r in rr])),
                   "matches_receiver_policy": float(np.mean([r["vista_matches_receiver_policy"] for r in rr])),
                   "throughput_tps": len(rr) / wall_m})
    # raw-attribute baseline (B2) bytes/leakage from the main detailed rows
    sub = rows[:m]
    sat = [r for r in sub if r["policy_satisfiable"]]
    e3.append({"mode": "B2_RAW", "backend": "signed", "payments": len(sub),
               "mean_claims_disclosed_when_satisfiable": S.n,
               "mean_leakage_bits_when_satisfiable": float(np.mean([r["leak_B2_raw_bits"] for r in sat])),
               "mean_capsule_bytes": float(np.mean([r["raw_evidence_bytes"] for r in sub])),
               "matches_receiver_policy": 1.0})
    files.append(write_csv(os.path.join(out, "e3_disclosure_modes.csv"), e3))

    # 3) E7 throughput scaling
    ns = min(n, cfg.get("scaling_payments", 16_000))
    sc = []
    for wk in cfg.get("scaling_workers", [1, 2, 4, 8]):
        if wk > (os.cpu_count() or 1):
            continue
        rr, wall_s = _run_pool(wc, _tasks(ns, max(250, ns // (wk * 4)), "DET", "signed", False), wk)
        pc = percentiles([r["e2e_ms"] - r["net_ms"] for r in rr])
        sc.append({"workers": wk, "payments": len(rr), "wall_s": wall_s, "throughput_tps": len(rr) / wall_s,
                   "cpu_path_p50_ms": pc["p50"], "cpu_path_p99_ms": pc["p99"]})
    files.append(write_csv(os.path.join(out, "e7_throughput_scaling.csv"), sc))
    return files


def _summaries(rows, wall, workers, out):
    files = []
    n = len(rows)
    lat = {k: percentiles([r[k] for r in rows]) for k in ("gen_ms", "net_ms", "verify_ms", "policy_ms", "man_ms", "e2e_ms")}
    files.append(write_csv(os.path.join(out, "e7_latency.csv"), [
        dict({"component": k}, **v) for k, v in lat.items()] + [
        {"component": "throughput", "n": n, "mean": n / wall, "p50": "", "p95": "", "p99": "",
         "max": f"{workers} workers, wall {wall:.1f}s"}]))

    # H1: decision fidelity to the receiver's own policy on the sender's true evidence
    ideal = np.array([r["policy_satisfiable"] for r in rows])
    fid = []
    for name, dec in (("B0_metadata_only", [r["b0_metadata_accept"] for r in rows]),
                      ("B1_sender_risk_label", [r["b1_accept"] for r in rows]),
                      ("VISTA_initial", [r["decision_initial"] == "ACCEPT" for r in rows])):
        dec = np.array(dec)
        agree = dec == ideal
        lo, hi = bootstrap_ci(agree.astype(float), n_boot=500)
        fid.append({"approach": name, "payments": n, "agreement_with_receiver_policy": float(agree.mean()),
                    "ci95_lo": lo, "ci95_hi": hi,
                    "false_accept_rate": float((dec & ~ideal).sum() / max((~ideal).sum(), 1)),
                    "false_reject_rate": float((~dec & ideal).sum() / max(ideal.sum(), 1)),
                    "accept_rate": float(dec.mean())})
    files.append(write_csv(os.path.join(out, "h1_decision_fidelity.csv"), fid))

    # H2/E3 on the full stream: leakage and claims per disclosure strategy (satisfiable payments)
    sat = [r for r in rows if r["policy_satisfiable"]]
    lk = []
    for name, lcol, ccol in (("B2_raw_attributes", "leak_B2_raw_bits", None),
                             ("B4_vsac_disclose_all", "leak_B4_all_bits", "claims_B4_all"),
                             ("B3_fixed_bundle", "leak_B3_fixed_bits", "claims_B3_fixed"),
                             ("VISTA_MDPS_DET", "leak_DET_bits", "claims_DET"),
                             ("VISTA_BOP_floor", "leak_BOP_floor_bits", None)):
        v = np.array([r[lcol] for r in sat], dtype=float)
        lo, hi = bootstrap_ci(v, n_boot=500)
        lk.append({"strategy": name, "payments": len(sat), "mean_leakage_bits": float(v.mean()), "ci95_lo": lo,
                   "ci95_hi": hi, "p95_leakage_bits": float(np.percentile(v, 95)),
                   "mean_claims": float(np.mean([r[ccol] for r in sat])) if ccol else (S.n if "raw" in name else 0.0)})
    files.append(write_csv(os.path.join(out, "h2_leakage_by_strategy.csv"), lk))

    # H3: MAN-S vs blanket step-up on identical payer outcomes (paired)
    su = [r for r in rows if r["decision_initial"] in ("STEP_UP", "HOLD")]
    if su:
        h3 = []
        for name, acc, fr_, rd, us in (
                ("MAN_S", [r["decision_final"] == "ACCEPT" for r in su], [r["stepup_friction"] for r in su],
                 [r["stepup_rounds"] for r in su], [r["stepup_user_seconds"] for r in su]),
                ("B6_blanket", [bool(r["blanket_accept"]) for r in su], [r["blanket_friction"] for r in su],
                 [r["blanket_rounds"] for r in su], [r["blanket_user_seconds"] for r in su])):
            acc, fr_ = np.array(acc, float), np.array(fr_, float)
            h3.append({"strategy": name, "stepup_payments": len(su), "resolution_rate": float(acc.mean()),
                       "mean_friction": float(fr_.mean()), "mean_rounds": float(np.mean(rd)),
                       "mean_user_seconds": float(np.mean(us)),
                       "friction_per_resolved": float(fr_.sum() / max(acc.sum(), 1))})
        diff = np.array([r["stepup_friction"] - r["blanket_friction"] for r in su], float)
        lo, hi = bootstrap_ci(diff, n_boot=500)
        h3.append({"strategy": "paired_diff_MAN_minus_blanket", "stepup_payments": len(su),
                   "mean_friction": float(diff.mean()), "resolution_rate": float(
                       np.mean([r["decision_final"] == "ACCEPT" for r in su]) - np.mean([bool(r["blanket_accept"]) for r in su])),
                   "friction_ci95": f"[{lo:.3f}, {hi:.3f}]"})
        files.append(write_csv(os.path.join(out, "h3_man_vs_blanket.csv"), h3))

    # decisions by scenario / tier / archetype / stance (E8 heterogeneity)
    for keyname, keyf in (("scenario", lambda r: r["scenario"]), ("tier", lambda r: r["tier"]),
                          ("heterogeneity", lambda r: f"{r['sender_archetype']}->{r['receiver_stance']}")):
        grp = defaultdict(list)
        for r in rows:
            grp[keyf(r)].append(r)
        tab = []
        for k, rr in sorted(grp.items()):
            ci = Counter(r["decision_initial"] for r in rr)
            cf = Counter(r["decision_final"] for r in rr)
            su = [r for r in rr if r["decision_initial"] in ("STEP_UP", "HOLD")]
            tab.append({keyname: k, "payments": len(rr), "satisfiable_rate": float(np.mean([r["policy_satisfiable"] for r in rr])),
                        "initial_accept": ci["ACCEPT"] / len(rr), "initial_stepup": ci["STEP_UP"] / len(rr),
                        "initial_hold": ci["HOLD"] / len(rr), "initial_reject": ci["REJECT"] / len(rr),
                        "final_accept": cf["ACCEPT"] / len(rr), "final_hold": cf["HOLD"] / len(rr),
                        "final_reject": cf["REJECT"] / len(rr),
                        "stepup_resolved_rate": float(np.mean([r["decision_final"] == "ACCEPT" for r in su])) if su else math.nan,
                        "mean_stepup_rounds": float(np.mean([r["stepup_rounds"] for r in su])) if su else math.nan,
                        "mean_stepup_friction": float(np.mean([r["stepup_friction"] for r in su])) if su else math.nan,
                        "mean_leakage_bits": float(np.mean([r["leakage_bits"] for r in rr])),
                        "mean_capsule_bytes": float(np.mean([r["capsule_bytes"] for r in rr])),
                        "verification_failures": sum(r["verify_reason"] != "ok" for r in rr)})
        files.append(write_csv(os.path.join(out, f"e8_decisions_by_{keyname}.csv"), tab))
    return files


# ------------------------------------------------------------------------------------------------ E10
def run_e10(cfg, out):
    """Failure injection timeline: registry outage, key revocation + rotation, schema rollout, drops."""
    wc = dict(cfg["world"], n_payments=cfg.get("e10_payments", 6000), n_banks=min(cfg["world"]["n_banks"], 12),
              n_payers=min(cfg["world"]["n_payers"], 20_000))
    world, pay = build_world(wc)
    rng = np.random.default_rng(wc["seed"] + 10)
    n = len(pay["ts"])
    t_start = int(pay["ts"][0])
    dt = 0.5
    minute = lambda k: int(k * dt // 60)  # noqa: E731
    total_min = minute(n - 1) + 1
    OUT_A, OUT_B = 10, 25
    REVOKE, ROTATE = 30, 33
    SCH_SEND, SCH_RECV = 38, 42
    victim = 0  # largest bank's issuer key is revoked
    v2 = Schema("vista-claims-2026.2", S.attributes)
    per_min = defaultdict(Counter)
    unsafe = 0
    rotated = upgraded_s = upgraded_r = False
    for k in range(n):
        now = t_start + int(k * dt)
        mn = minute(k)
        world.trust.online = not (OUT_A <= mn < OUT_B)
        if mn >= REVOKE and world.keys[world.banks[victim].id][0].key_id in world.trust.keys and \
                world.trust.keys[world.keys[world.banks[victim].id][0].key_id].revoked_at is None:
            world.trust.revoke(world.keys[world.banks[victim].id][0].key_id, now)
        if mn >= ROTATE and not rotated:
            nk = SigningKey()
            world.trust.online = True
            world.trust.rotate(world.banks[victim].id, "issuer", nk, now, 10**7)
            for (b, *_), snd in world._senders.items():
                if b == victim:
                    snd.issuer = nk
            world.keys[world.banks[victim].id] = (nk, world.keys[world.banks[victim].id][1])
            rotated = True
        if mn >= SCH_SEND and not upgraded_s:
            for snd in world._senders.values():
                snd.schema = v2
            upgraded_s = True
        if mn >= SCH_RECV and not upgraded_r:
            for rc in world._receivers.values():
                rc.verifier.schemas[v2.version] = v2
            upgraded_r = True
        p = world.payment(pay, k).with_(timestamp_ms=now * 1000)
        s, r = int(pay["sender"][k]), int(pay["receiver"][k])
        pid = pm.pid(p)
        ev = wl.evidence_for(world.pop, int(pay["payer"][k]), now - 1, int(pay["scenario"][k]), pid,
                             world.arche["archetypes"][world.banks[s].archetype], rng)
        snd = world.sender(s, "DET", "signed")
        if upgraded_s:
            snd.schema = v2
        rcv = world.receiver(r)
        node = world.policy_for(r, p.amount_minor)
        nonce = rcv.verifier.nonces.issue(snd.id, 1, now, rng)[0]
        o = snd.build(p, ev, node, nonce, now)
        blob = None if rng.random() < 0.005 else o.blob  # dropped VSAC in transit
        d = rcv.decide(p, blob, now)
        c = per_min[mn]
        c["payments"] += 1
        c["satisfiable"] += o.satisfiable
        c["accepted"] += d["decision"] == "ACCEPT"
        c["accepted_when_satisfiable"] += (d["decision"] == "ACCEPT") and o.satisfiable
        c[f"reason:{d['verify_reason']}"] += 1
        if d["decision"] == "ACCEPT" and (s == victim and REVOKE <= mn < ROTATE):
            unsafe += 1
    rows = []
    for mn in range(total_min):
        c = per_min[mn]
        if not c["payments"]:
            continue
        phase = ("registry_outage" if OUT_A <= mn < OUT_B else "key_revoked" if REVOKE <= mn < ROTATE else
                 "schema_skew" if SCH_SEND <= mn < SCH_RECV else "normal")
        row = {"minute": mn, "phase": phase, "payments": c["payments"],
               "availability_when_satisfiable": c["accepted_when_satisfiable"] / max(c["satisfiable"], 1),
               "accept_rate": c["accepted"] / c["payments"]}
        for key in ("ok", "status_unavailable", "revoked_key", "unknown_schema", "missing_capsule", "expired_key"):
            row[f"n_{key}"] = c[f"reason:{key}"]
        rows.append(row)
    rows.append({"minute": "summary", "phase": f"unsafe_accepts_during_revocation={unsafe}",
                 "payments": n})
    return [write_csv(os.path.join(out, "e10_failure_injection.csv"), rows)]
