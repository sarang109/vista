"""E27: does the integrated disclosure governor cost anything? Ablation ladder on the full protocol path.

Every configuration sends the same payments through SEN -> governor -> VSAC build -> simulated rail ->
verification -> receiver policy -> MAN-S step-up, with the components switched on one at a time:
  DET_v1, RAND_v1 (exponential, fresh), PCGS (fresh), +SBS, +ledger, +fixed_charge, +budget, +audit (= IDG)
Payments are sharded by sender institution (each sender runs its own governor), so per-account state is exact.

Leakage is reported three ways: the disclosed claims' surprisal (atomic information), the policy-level
single-payment Lambda of the deployed kernel, and an exposure-weighted Lambda that charges each payment the
Lambda_n of its position n in its (payer, receiver, policy) sequence (Theorem 3 with unchanged evidence; an
upper envelope for fresh kernels, exact for DET and SBS).
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import payment as pm
from ..claims import DEFAULT_SCHEMA as S
from ..governor import GovernorConfig
from ..progress import PROGRESS
from ..repeated import repeated_metrics
from ..simulator.world import ACTION_SECONDS, ACTIONS
from ..util import write_csv
from . import exp_e2e
from .shards import bank_shards, iter_shard

MAX_ROUNDS = exp_e2e.MAX_ROUNDS
NMAX = 10


def ladder(cfg) -> list[GovernorConfig]:
    g = cfg.get("governor", {})
    rho = g.get("ledger_rho", {"credential_age": 2, "account_age": 2})
    fc = g.get("fixed_charge", {"weight": 0.5, "F": {}, "per_claim": 0.25})
    base = {"overhead": g.get("overhead", 0.0235), "lam": g.get("lam", 1.0)}
    steps = [("DET_v1", {"kernel": "det", "sticky": False}), ("RAND_v1", {"kernel": "exp", "sticky": False}),
             ("PCGS", {"kernel": "pcgs", "sticky": False}), ("+SBS", {"kernel": "pcgs", "sticky": True}),
             ("+ledger", {"ledger_rho": rho}), ("+fixed_charge", {"fixed_charge": fc}),
             ("+budget", {"budget": True, "probe_limit": g.get("probe_limit", 10)}), ("+audit (IDG)", {"audit": True})]
    out, acc = [], dict(base)
    for name, over in steps:
        acc.update(over)
        out.append(GovernorConfig.from_dict(dict(acc, name=name)))
    return out


def _lam_curve(t, fresh: bool) -> list[float]:
    if len(t.front) < 2:
        return [0.0] * NMAX
    if not fresh:
        return [t.ss.lam(t.Q)] * NMAX
    return [repeated_metrics(t.ss, t.Q, n)["Lambda"] for n in range(1, NMAX + 1)]


def _run(task):
    shard, n_rows, frac, gdict = task
    world, src, wc = exp_e2e._G["world"], exp_e2e._G["src"], exp_e2e._G["wc"]
    gcfg = GovernorConfig.from_dict(gdict)
    for r_ in getattr(world, "_receivers", {}).values():
        r_.verifier.seen_pids.clear()
    fresh = not gcfg.sticky and gcfg.kernel != "det"
    for b in shard:  # offline step (Algorithm 1): compile every policy before the online path is timed
        g = world.sender(b, "IDG", None, gov=gcfg).governor
        for tiers in world.policies.values():
            for node in tiers.values():
                g.table(node)
    curves: dict = {}
    seqs: dict = {}
    cols = defaultdict(list)
    t_wall = time.perf_counter()
    for pay, j in iter_shard(src, world, shard, n_rows, frac):
        gid = int(pay["gidx"][j])
        rng = np.random.default_rng([wc["seed"], gid])
        p = world.payment(pay, j)
        s, r = int(pay["sender"][j]), int(pay["receiver"][j])
        payer, scen = int(pay["payer"][j]), int(pay["scenario"][j])
        pid = pm.pid(p)
        now = int(pay["ts"][j]) + 1
        ev = world.evidence(pay, j, pid, rng)
        snd = world.sender(s, "IDG", None, gov=gcfg)
        rcv = world.receiver(r)
        node = world.policy_for(r, p.amount_minor)
        nonce = rcv.verifier.nonces.issue(snd.id, 1, now, rng)[0]
        t0 = time.perf_counter()
        o = snd.build(p, ev, node, nonce, now)
        gen_ms = (time.perf_counter() - t0) * 1e3
        info = snd.last_info
        net_ms = float(rng.lognormal(math.log(12.0), 0.5))
        d = rcv.decide(p, o.blob, now)
        initial = final = d["decision"]
        rounds, plan, failed = 0, d.get("plan"), set()
        u = rng.random(len(ACTIONS))
        ok_a = [world.can_complete(payer, s, a, scen) and u[k] < a.p for k, a in enumerate(ACTIONS)]
        user_s = 0
        while final in ("STEP_UP", "HOLD") and plan and plan["p_star"] > 0 and plan["first_action"] is not None \
                and rounds < MAX_ROUNDS:
            act = ACTIONS[plan["first_action"]]
            rounds += 1
            user_s += ACTION_SECONDS[act.name]
            if ok_a[plan["first_action"]]:
                ev = world.apply_action(ev, act, pid, now + user_s, s)
            else:
                failed.add(act.name)
            n2 = rcv.verifier.nonces.issue(snd.id, 1, now + user_s, rng)[0]
            d2 = rcv.decide(p, snd.build(p, ev, node, n2, now + user_s).blob, now + user_s, frozenset(failed))
            final, plan = d2["decision"], d2.get("plan")
        t = snd.governor.table(node)
        if id(t) not in curves:
            curves[id(t)] = _lam_curve(t, fresh)
        key = (payer, r, t.hash)
        sq = seqs.setdefault(key, [0, 0])
        sq[0] += 1
        if info.get("event") not in ("unsat", "ledger_refused", None):
            sq[1] |= 1 << info["branch"]
        n = min(sq[0], NMAX)
        disclosed = o.disclosed
        cols["gen_ms"].append(gen_ms)
        cols["governor_ms"].append(info.get("governor_ms", 0.0))
        cols["verify_ms"].append(d["verify_ms"])
        cols["e2e_ms"].append(gen_ms + net_ms + d["verify_ms"] + d.get("policy_ms", 0.0) + d.get("man_ms", 0.0))
        cols["sat"].append(o.satisfiable)
        cols["accept_initial"].append(initial == "ACCEPT")
        cols["accept_final"].append(final == "ACCEPT")
        cols["ledger_refused"].append(info.get("event") == "ledger_refused")
        cols["fallback"].append(bool(info.get("fallback")))
        cols["redraw"].append(info.get("event") == "redraw")
        cols["fixed_charge"].append(info.get("fixed_charge", math.nan))
        cols["leak_bits"].append(world.surprisal(tuple(disclosed)) if disclosed is not None and o.satisfiable
                                 and info.get("event") != "ledger_refused" else math.nan)
        lam_ok = o.satisfiable and info.get("event") != "ledger_refused"
        cols["lam1"].append(curves[id(t)][0] if lam_ok else math.nan)
        cols["lamn"].append(curves[id(t)][n - 1] if lam_ok else math.nan)
        cols["seq_pos"].append(n)
        cols["distinct"].append(bin(sq[1]).count("1") if sq[0] > 1 and sq[1] else math.nan)
    gov = [snd_.governor for (b, *_rest, gname), snd_ in world._senders.items() if gname == gcfg.name and b in shard]
    extra = {"budget_suppressed": sum(g.budget.suppressed for g in gov),
             "audit_entries": sum(g.log.count for g in gov if g.log is not None),
             "audit_bytes": sum(g.log.bytes for g in gov if g.log is not None),
             "compile_ms": [1e3 * t.compile_s for g in gov for t in g.tables.values()],
             "shard_seconds": time.perf_counter() - t_wall}
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}, extra


def run_e27(cfg, out):
    wc = cfg["world"]
    workers = cfg.get("workers", os.cpu_count() or 4)
    n_rows = exp_e2e.source_size(wc)
    frac = cfg.get("e27_payer_fraction", 1.0)
    if cfg.get("e27_payments"):
        n_rows = min(n_rows, cfg["e27_payments"])
    world, _ = exp_e2e.build_world(dict(wc, n_payments=min(n_rows, 1000)) if wc.get("source") != "paysim" else wc)
    shards = bank_shards(world.banks, workers * 3)
    configs = ladder(cfg)
    rows = []
    ex = ProcessPoolExecutor(max_workers=workers, initializer=exp_e2e._init, initargs=(wc,))
    try:
        for ci, g in enumerate(configs):
            t0 = time.time()
            parts = list(ex.map(_run, [(sh, n_rows, frac, g.__dict__) for sh in shards]))
            wall = time.time() - t0
            PROGRESS.update(ci + 1, len(configs), "governor configs")
            c = {k: np.concatenate([pp[0][k] for pp in parts if k in pp[0]]) for k in parts[0][0]}
            n = len(c["sat"])
            sat = c["sat"].astype(bool)
            rep = c["seq_pos"] > 1
            comp = [x for pp in parts for x in pp[1]["compile_ms"]]
            rows.append({
                "config": g.name, "payments": n, "satisfiable_rate": float(sat.mean()),
                "accept_rate_initial": float(c["accept_initial"].mean()),
                "accept_rate_final": float(c["accept_final"].mean()),
                "ledger_false_reject_rate": float(c["ledger_refused"][sat].mean()) if sat.any() else 0.0,
                "ledger_fallback_rate": float(c["fallback"][sat].mean()) if sat.any() else 0.0,
                "sticky_redraw_rate": float(c["redraw"][sat].mean()) if sat.any() else 0.0,
                "mean_disclosed_surprisal_bits": float(np.nanmean(c["leak_bits"])),
                "mean_Lambda1_bits": float(np.nanmean(c["lam1"])),
                "mean_exposure_weighted_Lambda_bits": float(np.nanmean(c["lamn"])),
                "repeat_payment_share": float(rep.mean()),
                "mean_distinct_branches_repeat_seq": float(np.nanmean(c["distinct"])) if rep.any() else math.nan,
                "mean_fixed_charge": float(np.nanmean(c["fixed_charge"])),
                "governor_p50_ms": float(np.percentile(c["governor_ms"], 50)),
                "governor_p99_ms": float(np.percentile(c["governor_ms"], 99)),
                "gen_p99_ms": float(np.percentile(c["gen_ms"], 99)),
                "verify_p99_ms": float(np.percentile(c["verify_ms"], 99)),
                "e2e_p50_ms": float(np.percentile(c["e2e_ms"], 50)), "e2e_p99_ms": float(np.percentile(c["e2e_ms"], 99)),
                "throughput_tps": n / wall, "wall_s": wall,
                "max_shard_s": max(pp[1]["shard_seconds"] for pp in parts),
                "policy_compile_p50_ms": float(np.median(comp)) if comp else math.nan,
                "budget_suppressed_probes": sum(pp[1]["budget_suppressed"] for pp in parts),
                "audit_entries": sum(pp[1]["audit_entries"] for pp in parts),
                "audit_bytes_per_entry": (sum(pp[1]["audit_bytes"] for pp in parts) /
                                          max(sum(pp[1]["audit_entries"] for pp in parts), 1))})
    finally:
        ex.shutdown(wait=True)
    base = rows[0]
    for r in rows:
        r["e2e_p99_delta_ms_vs_DET"] = r["e2e_p99_ms"] - base["e2e_p99_ms"]
        r["gen_p99_delta_ms_vs_DET"] = r["gen_p99_ms"] - base["gen_p99_ms"]
    return [write_csv(os.path.join(out, "e27_governor_ablation.csv"), rows)]
