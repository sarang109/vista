"""End-to-end evaluation on a realistic interbank transaction stream (H1-H5, E3, E7, E8, E10).

Every payment goes through the real protocol path: SEN -> MDPS/selection -> VSAC generation (real
Ed25519 / selective disclosure / hash-chain / OR-proof crypto) -> simulated rail latency ->
verification -> receiver policy -> MAN-S step-up rounds executed by the (simulated) payer.

Memory is bounded: workers process fixed-size chunks read from a memory-mapped source, write their detailed
rows as pre-formatted CSV text, and return only small numeric columns that StreamAgg folds into counters and
histograms. Results are consumed in order through a sliding window, so nothing accumulates in the parent.
"""
from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import payment as pm
from ..claims import DEFAULT_SCHEMA as S, Schema
from ..crypto.signed import SigningKey
from ..policy import evaluate, evaluate_many
from ..progress import PROGRESS
from ..sen import normalize
from ..simulator import workload as wl
from ..simulator.source import make_source
from ..simulator.world import ACTION_SECONDS, ACTIONS, World
from ..util import _fmt, canonical_json, percentiles, write_csv
from .stream_agg import ARCH, DEC, SCEN, STANCE, TIERS, StreamAgg

MAX_ROUNDS = 5
_G: dict = {}

NUM = ("gen_ms", "net_ms", "verify_ms", "policy_ms", "man_ms", "e2e_ms", "capsule_bytes", "claims_disclosed",
       "leakage_bits", "stepup_rounds", "stepup_friction", "stepup_user_seconds", "blanket_friction",
       "blanket_rounds", "blanket_user_seconds", "leak_B2_raw_bits", "leak_B4_all_bits", "leak_B3_fixed_bits",
       "leak_DET_bits", "leak_BOP_floor_bits", "claims_B4_all", "claims_B3_fixed", "claims_DET",
       "raw_evidence_bytes")


def build_world(wc: dict):
    world = World(wc["n_banks"], wc["n_payers"], wc["seed"], wc.get("prior_sample", 20_000))
    return world, make_source(world, wc)


def source_size(wc: dict) -> int:
    if wc.get("source") == "paysim":
        from ..simulator.source import resolve_path
        with open(os.path.join(resolve_path(wc.get("paysim_cache", "data/paysim_cache")), "meta.json")) as fh:
            n = json.load(fh)["rows_usable"]
        return min(n, wc["n_payments"]) if wc.get("n_payments") else n
    return wc["n_payments"]


def _init(wc):
    _G["world"], _G["src"] = build_world(wc)
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


def _num(v) -> float:
    if v is None or v == "":
        return math.nan
    return float(v)


def _columns(rows: list[dict]) -> dict:
    c = {k: np.array([_num(r.get(k)) for r in rows], dtype=np.float32) for k in NUM}
    c["scenario"] = np.array([SCEN.index(r["scenario"]) for r in rows], dtype=np.int8)
    c["tier"] = np.array([TIERS.index(r["tier"]) for r in rows], dtype=np.int8)
    c["archetype"] = np.array([ARCH.index(r["sender_archetype"]) for r in rows], dtype=np.int8)
    c["stance"] = np.array([STANCE.index(r["receiver_stance"]) for r in rows], dtype=np.int8)
    c["decision_initial"] = np.array([DEC.index(r["decision_initial"]) for r in rows], dtype=np.int8)
    c["decision_final"] = np.array([DEC.index(r["decision_final"]) for r in rows], dtype=np.int8)
    c["sat"] = np.array([r["policy_satisfiable"] for r in rows], dtype=bool)
    c["match"] = np.array([r["vista_matches_receiver_policy"] for r in rows], dtype=bool)
    c["b0"] = np.array([r["b0_metadata_accept"] for r in rows], dtype=bool)
    c["b1"] = np.array([r["b1_accept"] for r in rows], dtype=bool)
    c["blanket_accept"] = np.array([-1 if r["blanket_accept"] == "" else int(bool(r["blanket_accept"])) for r in rows],
                                   dtype=np.int8)
    c["verify_ok"] = np.array([r["verify_reason"] == "ok" for r in rows], dtype=bool)
    return c


def _process(task):
    start, stop, stride, mode, backend, detail, want_csv = task
    world, wc = _G["world"], _G["wc"]
    # Receivers (and their replay caches) persist inside a worker process. E3 re-sends the same payments
    # (same PID and deterministic nonce) in several modes, so without this reset the later modes are
    # rejected as replays. Replay defence itself is exercised separately in E5.
    for _r in getattr(world, "_receivers", {}).values():
        _r.verifier.seen_pids.clear()
    idx = np.arange(start, stop, stride)
    pay = _G["src"].take(idx, world)
    rows = []
    fixed = world.sender(0, "FIXED").fixed_bundle
    for j in range(len(idx)):
        gid = int(pay["gidx"][j])
        rng = np.random.default_rng([wc["seed"], gid])
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
            "tx_idx": gid, "tx_id": p.tx_id, "ts": int(pay["ts"][j]), "sender_fi": p.sender_fi,
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
    keys = text = None
    if want_csv:
        keys = list(rows[0].keys())
        buf = io.StringIO()
        wr = csv.writer(buf, lineterminator="\n")
        for r_ in rows:
            wr.writerow([_fmt(r_.get(k, "")) for k in keys])
        text = buf.getvalue()
    return _columns(rows), text, keys, len(rows)


def _warm(_):
    time.sleep(0.2)
    return os.getpid()


def _tasks(n_rows, chunk, mode, backend, detail, want_csv=False, stride=1):
    """Tasks over n_rows positions sampled every `stride` global rows (stride 1 = every row)."""
    return [(a * stride, min(a + chunk, n_rows) * stride, stride, mode, backend, detail, want_csv)
            for a in range(0, n_rows, chunk)]


def _stream(ex, tasks, window, on_result):
    """Ordered, bounded-memory map: at most `window` chunks are in flight or waiting at once."""
    it = iter(tasks)
    pend: deque = deque(ex.submit(_process, t) for t in itertools.islice(it, window))
    while pend:
        res = pend.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            pend.append(ex.submit(_process, nxt))
        on_result(res)


def _collect(ex, tasks, window, bump):
    cols: list[dict] = []

    def on(res):
        cols.append(res[0])
        bump(res[3])

    t0 = time.time()
    _stream(ex, tasks, window, on)
    wall = time.time() - t0
    return {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}, wall


def _new_pool(wc, workers):
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(wc,))
    list(ex.map(_warm, range(workers * 2)))
    return ex


# ------------------------------------------------------------------------------------------------ main runs
E3_MODES = [("ALL", "signed"), ("FIXED", "signed"), ("DET", "signed"), ("RAND", "signed"), ("DET", "sd"),
            ("DET", "hashchain"), ("BOP", "bop")]


def run(cfg, out):
    wc = cfg["world"]
    workers = cfg.get("workers", os.cpu_count() or 4)
    chunk = cfg.get("chunk", 2500)
    window = workers * 3
    n = source_size(wc)
    e3_n = min(n, cfg.get("e3_payments", 20_000))
    sc_n = min(n, cfg.get("scaling_payments", 16_000))
    sc_workers = [w for w in cfg.get("scaling_workers", [1, 2, 4, 8]) if w <= (os.cpu_count() or 1)]
    total_units = n + e3_n * len(E3_MODES) + sc_n * len(sc_workers)
    done = [0]

    def bump(k):
        done[0] += k
        PROGRESS.update(done[0], total_units, "payments")

    files = []
    from ..simulator.world import ROOT
    arche = wl.load_archetypes(os.path.join(ROOT, "configs", "banks", "archetypes.json"))
    banks = wl.generate_banks(wc["n_banks"], arche, np.random.default_rng(wc["seed"]))  # same banks as the workers
    files.append(write_csv(os.path.join(out, "banks.csv"), [
        {"bank": b.id, "archetype": b.archetype, "market_share": b.share, "receiver_stance": b.stance,
         "default_backend": arche["archetypes"][b.archetype]["backend"]} for b in banks]))
    if wc.get("source") != "paysim":  # synthetic stream: also save the raw generated workload
        _, src_full = build_world(wc)
        wl.dump_workload_csv(os.path.join(out, "workload_transactions.csv"), src_full.pay, banks)
        files.append(os.path.join(out, "workload_transactions.csv"))
        del src_full

    gz = cfg.get("e2e_csv", "gz" if n > 1_000_000 else "plain") == "gz"
    path = os.path.join(out, "e2e_transactions.csv" + (".gz" if gz else ""))
    agg = StreamAgg()
    state = {"header": False}

    ex = _new_pool(wc, workers)
    try:
        t0 = time.time()
        fh = gzip.open(path, "wt", compresslevel=3, newline="") if gz else open(path, "w", newline="")
        with fh:
            def on_main(res):
                cols, text, keys, k = res
                agg.add(cols)
                if not state["header"]:
                    fh.write(",".join(keys) + "\n")
                    state["header"] = True
                fh.write(text)
                bump(k)

            _stream(ex, _tasks(n, chunk, "DET", None, True, True), window, on_main)
        wall = time.time() - t0
        files.append(path)
        files += agg.write(out, wall, workers)

        # E3: disclosure modes x backends on a uniform subsample
        stride = max(1, n // e3_n)
        e3 = []
        for mode, backend in E3_MODES:
            c, wall_m = _collect(ex, _tasks(e3_n, chunk, mode, None if mode == "BOP" else backend, False, False, stride),
                                 window, bump)
            sat = c["sat"].astype(bool)
            e3.append({"mode": mode, "backend": backend, "payments": len(sat),
                       "mean_claims_disclosed_when_satisfiable": float(c["claims_disclosed"][sat].mean()),
                       "mean_leakage_bits_when_satisfiable": float(c["leakage_bits"][sat].mean()),
                       "mean_capsule_bytes": float(c["capsule_bytes"].mean()),
                       "gen_p50_ms": float(np.percentile(c["gen_ms"], 50)), "gen_p99_ms": float(np.percentile(c["gen_ms"], 99)),
                       "verify_p50_ms": float(np.percentile(c["verify_ms"], 50)),
                       "verify_p99_ms": float(np.percentile(c["verify_ms"], 99)),
                       "accept_rate_initial": float((c["decision_initial"] == 0).mean()),
                       "matches_receiver_policy": float(c["match"].mean()), "throughput_tps": len(sat) / wall_m})
    finally:
        ex.shutdown(wait=True)
    e3.append({"mode": "B2_RAW", "backend": "signed", "payments": agg.n, "mean_claims_disclosed_when_satisfiable": S.n,
               "mean_leakage_bits_when_satisfiable": agg.leak["B2_raw_attributes"][0].mean,
               "mean_capsule_bytes": agg.raw_bytes.mean, "matches_receiver_policy": 1.0})
    files.append(write_csv(os.path.join(out, "e3_disclosure_modes.csv"), e3))

    # E7 throughput scaling (fresh pools with different worker counts)
    sc = []
    stride = max(1, n // sc_n)
    for wk in sc_workers:
        ex2 = _new_pool(wc, wk)
        try:
            c, wall_s = _collect(ex2, _tasks(sc_n, max(250, sc_n // (wk * 4)), "DET", "signed", False, False, stride),
                                 wk * 3, bump)
        finally:
            ex2.shutdown(wait=True)
        cpu = c["e2e_ms"] - c["net_ms"]
        sc.append({"workers": wk, "payments": len(cpu), "wall_s": wall_s, "throughput_tps": len(cpu) / wall_s,
                   "cpu_path_p50_ms": float(np.percentile(cpu, 50)), "cpu_path_p99_ms": float(np.percentile(cpu, 99))})
    files.append(write_csv(os.path.join(out, "e7_throughput_scaling.csv"), sc))

    if wc.get("source") == "paysim":
        from ..simulator.source import PaySimSource, resolve_path
        info = PaySimSource(resolve_path(wc.get("paysim_cache", "data/paysim_cache")), wc.get("n_payments"),
                            float(wc.get("paysim_amount_scale", 1.0))).describe()
    else:
        info = {"kind": "synthetic"}
    info.update({"rows_used": n, "payers_in_population": wc["n_payers"], "banks": wc["n_banks"],
                 "percentile_method": "log histogram, ~0.8% relative bin width (latencies); 0.01-bit bins (leakage)"})
    with open(os.path.join(out, "data_source.json"), "w") as f2:
        json.dump(info, f2, indent=2)
    files.append(os.path.join(out, "data_source.json"))
    return files


# ------------------------------------------------------------------------------------------------ E10
def run_e10(cfg, out):
    """Failure injection timeline: registry outage, key revocation + rotation, schema rollout, drops."""
    wc = dict(cfg["world"], n_banks=min(cfg["world"]["n_banks"], 12), n_payers=min(cfg["world"]["n_payers"], 20_000))
    n_req = cfg.get("e10_payments", 6000)
    if wc.get("source") == "paysim":
        wc["n_payments"] = source_size(cfg["world"])
    else:
        wc["n_payments"] = n_req
    world, source = build_world(wc)
    stride = max(1, source.n // n_req)
    pay = source.take(np.arange(0, source.n, stride)[:n_req], world)
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
