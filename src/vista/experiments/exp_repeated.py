"""E25: does fresh randomisation leak over repeated payments? (Theorem 3)

Part 1 (exact, unchanged evidence): Lambda_n, I(X;V^n) and Bayes attacker accuracy against n for DET, fresh
exponential RAND, fresh PCGS and PCGS with sticky selection (SBS), on 64-state toy spaces with three- and
four-branch policies, on the v1 E11 instances and on the receiver policy library under the world prior.

Part 2 (workload, drifting evidence): the payment stream (PaySim-backed or synthetic) replayed per
(payer, receiver, policy) sequence through the real SEN path. Per position n: branches revealed so far, an
empirical attacker trained on half of the payers and tested on the other half (target: the cheapest branch is
unavailable), linkability (same branch as the previous payment), sticky redraw rate and evidence drift.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import payment as pm
from .. import policy as pol
from ..claims import DEFAULT_SCHEMA as S, synthetic_schema
from ..governor import GovernorConfig, PolicyTable, evidence_epoch
from ..prior import Prior
from ..progress import PROGRESS
from ..repeated import StickyStore, coverage_payments, draw, prob_all_seen, q_min, repeated_metrics, sticky_metrics
from ..selection import SelectionSpace
from ..sen import normalize
from ..util import sha256, write_csv
from . import exp_e2e
from .exp_selection import e11_instances
from .shards import bank_shards, iter_shard, payer_selected

STRATS = ("DET", "RAND_fresh", "PCGS_fresh", "PCGS_SBS")
NMAX = 10


# ------------------------------------------------------------------------------------------ part 1
def toy_instances(cfg, count: int):
    rng = np.random.default_rng(cfg.get("seed", 1) + 250)
    sch = synthetic_schema(3, 4)
    made = 0
    while made < count:
        node = pol.random_dnf(sch, int(rng.integers(3, 6)), int(rng.integers(1, 3)), rng)
        fr = pol.frontier(node, sch.n)
        if len(fr) not in (3, 4):
            continue
        pr = Prior.random_ising(sch.dims, rng, "ferro", 1.5)
        w = fr.sum(1).astype(float) + rng.normal(0, 1.0, len(fr))
        sp = SelectionSpace(pr, fr, w - w.min() + 1.0)
        if sp.pA <= 0:
            continue
        made += 1
        yield {"family": "toy_64_state", "frontier_size": len(fr)}, sp


def theory_rows(family_iter, n_max: int, lam: float, overhead: float):
    rows, cover = [], []
    for k, (meta, sp) in enumerate(family_iter):
        ss = sp.signature_space()
        kernels = {"DET": ss.kernel_det(), "RAND_fresh": ss.kernel_exp(lam), "PCGS_fresh": ss.compile(overhead).Q}
        for n in range(1, n_max + 1):
            for name, Q in kernels.items():
                r = repeated_metrics(ss, Q, n) if name != "DET" else sticky_metrics(ss, Q, n)
                rows.append(dict(meta, instance=k, strategy=name, n=n, **{x: r[x] for x in (
                    "Lambda", "I_bits", "E_L_bits", "signature_recovery_acc", "branch_attack_acc", "branch_attack_base")}))
            r = sticky_metrics(ss, kernels["PCGS_fresh"], n)
            rows.append(dict(meta, instance=k, strategy="PCGS_SBS", n=n, **{x: r[x] for x in (
                "Lambda", "I_bits", "E_L_bits", "signature_recovery_acc", "branch_attack_acc", "branch_attack_base")}))
        for name in ("RAND_fresh", "PCGS_fresh"):
            qm = q_min(ss, kernels[name])
            if qm <= 0:
                continue
            nb = coverage_payments(qm, ss.m, 0.05)
            worst = min(prob_all_seen(kernels[name][a], nb) for a in range(ss.n_signatures))
            cover.append(dict(meta, instance=k, strategy=name, q_min=qm, n_bound_delta05=nb,
                              min_prob_all_seen_at_bound=worst, holds=worst >= 0.95 - 1e-12))
    return rows, cover


def summarise(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[(r["family"], r["strategy"], r["n"])].append(r)
    out = []
    for (fam, st, n), rr in sorted(acc.items()):
        out.append({"family": fam, "strategy": st, "n": n, "instances": len(rr),
                    "mean_Lambda_bits": float(np.mean([r["Lambda"] for r in rr])),
                    "mean_I_bits": float(np.mean([r["I_bits"] for r in rr])),
                    "mean_signature_recovery_acc": float(np.mean([r["signature_recovery_acc"] for r in rr])),
                    "mean_branch_attack_acc": float(np.mean([r["branch_attack_acc"] for r in rr])),
                    "mean_branch_attack_base": float(np.mean([r["branch_attack_base"] for r in rr]))})
    cross = []
    for fam in sorted({r["family"] for r in out}):
        det = {r["n"]: r["mean_Lambda_bits"] for r in out if r["family"] == fam and r["strategy"] == "DET"}
        for st in ("RAND_fresh", "PCGS_fresh"):
            cur = {r["n"]: r["mean_Lambda_bits"] for r in out if r["family"] == fam and r["strategy"] == st}
            nstar = next((n for n in sorted(cur) if cur[n] > det[n]), None)
            cross.append({"family": fam, "strategy": st, "passes_DET_at_n": nstar if nstar else f">{max(cur)}",
                          "Lambda_1": cur[1], "Lambda_max_n": cur[max(cur)], "Lambda_DET": det[1]})
    return out, cross


def library_family():
    from .exp_frontier import library_instances
    return library_instances()


# ------------------------------------------------------------------------------------------ part 2
def _scan(task):
    shard, n_rows, payer_fraction, lam, overhead = task
    world, src, wc = exp_e2e._G["world"], exp_e2e._G["src"], exp_e2e._G["wc"]
    tables: dict = {}
    gcfg = GovernorConfig(kernel="pcgs", overhead=overhead, sticky=True)
    stores = StickyStore(sha256(f"{wc['seed']}|e25".encode()))
    state: dict = {}
    by_n = defaultdict(lambda: np.zeros(6))  # count, distinct, linkable, redraw, drift, epoch_change
    attack = defaultdict(lambda: np.zeros(2))
    for pay, j in iter_shard(src, world, shard, n_rows, payer_fraction):
        gid = int(pay["gidx"][j])
        rng = np.random.default_rng([wc["seed"], gid, 25])
        p = world.payment(pay, j)
        s, r = int(pay["sender"][j]), int(pay["receiver"][j])
        pid = pm.pid(p)
        ev = world.evidence(pay, j, pid, rng)
        prof = world.arche["archetypes"][world.banks[s].archetype]["profile"]
        x = normalize(ev, pid, int(pay["ts"][j]) + 1, prof, S).profile
        node = world.policy_for(r, p.amount_minor)
        t = tables.get(id(node))
        if t is None:
            t = tables[id(node)] = PolicyTable(node, S, world.prior, world.cost, gcfg, None)
        if len(t.front) < 2:
            continue  # a single branch has no selection channel
        avail = (t.front <= np.asarray(x)).all(1)
        if not avail.any():
            continue
        code = int(avail.astype(np.int64) @ t.bits)
        y = int(not avail[int(np.argmin(t.base))])
        epoch = evidence_epoch(ev)
        key = (int(pay["payer"][j]), r, t.hash)
        split = int(payer_selected(np.array([pay["payer"][j]]), 0.5, salt=7_777_777)[0])
        exp_p = np.where(avail, np.exp(-lam * (t.base - t.base.min())), 0.0)
        choice = {"DET": int(t.order[np.flatnonzero(avail[t.order])[0]]),
                  "RAND_fresh": draw(exp_p / exp_p.sum(), rng.random()),
                  "PCGS_fresh": draw(t.probs(avail), rng.random())}
        # sticky key as in the governor: (account, receiver policy, epoch), shared by receivers with that policy
        v, ev_kind = stores.choose((key[0], t.hash, epoch), avail, t.probs(avail))
        choice["PCGS_SBS"] = v
        stt = state.setdefault(key, {"n": 0, "sig": None, "epoch": None, **{st: [0, -1] for st in STRATS}})
        stt["n"] += 1
        n = min(stt["n"], NMAX)
        drift = stt["sig"] is not None and stt["sig"] != code
        ech = stt["epoch"] is not None and stt["epoch"] != epoch
        stt["sig"], stt["epoch"] = code, epoch
        pidx = sorted(world.policies).index(world.banks[r].stance) * 4 + ("micro", "standard", "high",
                                                                         "very_high").index(world.tier(p.amount_minor))
        for st in STRATS:
            seen, last = stt[st]
            b = choice[st]
            seen |= 1 << b
            stt[st] = [seen, b]
            agg = by_n[(st, n)]
            agg += (1, bin(seen).count("1"), float(last == b) if stt["n"] > 1 else 0.0,
                    float(st == "PCGS_SBS" and ev_kind == "redraw"), float(drift), float(ech))
            attack[(st, split, pidx, n, seen, b, y)][0] += 1
    return {k: v.tolist() for k, v in by_n.items()}, {k: v[0] for k, v in attack.items()}, len(state)


def run_workload(cfg, out):
    wc = cfg["world"]
    workers = cfg.get("workers", os.cpu_count() or 4)
    n_rows = exp_e2e.source_size(wc)
    frac = cfg.get("e25_payer_fraction", 1.0)
    lam, ovh = cfg.get("e25_lambda", 1.0), cfg.get("e25_overhead", 0.0235)
    world, _ = exp_e2e.build_world(dict(wc, n_payments=min(n_rows, 1000)) if wc.get("source") != "paysim" else wc)
    shards = bank_shards(world.banks, workers * 2)
    by_n: dict = defaultdict(lambda: np.zeros(6))
    attack: dict = defaultdict(float)
    pairs = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=exp_e2e._init, initargs=(wc,)) as ex:
        done = 0
        for bn, at, npairs in ex.map(_scan, [(sh, n_rows, frac, lam, ovh) for sh in shards]):
            for k, v in bn.items():
                by_n[k] += np.array(v)
            for k, v in at.items():
                attack[k] += v
            pairs += npairs
            done += 1
            PROGRESS.update(done, len(shards), "shards")
    rows = []
    for st in STRATS:
        # empirical attacker: Bayes table from the training payers, accuracy on the held-out payers
        table = defaultdict(lambda: np.zeros(2))
        base = defaultdict(lambda: np.zeros(2))
        for (s_, split, pidx, n, seen, b, y), c in attack.items():
            if s_ == st and split == 1:
                table[(pidx, n, seen, b)][y] += c
                base[(pidx, n)][y] += c
        for n in range(1, NMAX + 1):
            a = by_n.get((st, n))
            if a is None or a[0] == 0:
                continue
            hit = hb = tot = 0.0
            for (s_, split, pidx, nn, seen, b, y), c in attack.items():
                if s_ != st or split != 0 or nn != n:
                    continue
                tb = table.get((pidx, n, seen, b))
                bb = base.get((pidx, n))
                pred = int(tb[1] > tb[0]) if tb is not None and tb.sum() > 0 else (int(bb[1] > bb[0]) if bb is not None else 0)
                predb = int(bb[1] > bb[0]) if bb is not None else 0
                hit += c * (pred == y)
                hb += c * (predb == y)
                tot += c
            rows.append({"strategy": st, "n": n if n < NMAX else f"{NMAX}+", "payments": int(a[0]),
                         "mean_distinct_branches_revealed": a[1] / a[0],
                         "linkability_same_branch_as_previous": a[2] / a[0] if n > 1 else math.nan,
                         "sticky_redraw_rate": a[3] / a[0] if st == "PCGS_SBS" else math.nan,
                         "evidence_signature_drift_rate": a[4] / a[0], "epoch_change_rate": a[5] / a[0],
                         "attacker_acc_test": hit / tot if tot else math.nan,
                         "attacker_base_test": hb / tot if tot else math.nan,
                         "attacker_lift": (hit - hb) / tot if tot else math.nan})
    return rows, pairs


def run_e25(cfg, out):
    files = []
    fams = [("toy", toy_instances(cfg, cfg.get("e25_toy", 40))),
            ("e11", ((dict({"family": f"v1_e11_{m['prior']}"}, frontier_size=m["frontier_size"]), sp)
                     for m, sp, _ in e11_instances(cfg))),
            ("library", library_family())]
    all_rows, all_cover = [], []
    for _, it in fams:
        rows, cover = theory_rows(it, cfg.get("e25_nmax", NMAX), cfg.get("e25_lambda", 1.0),
                                  cfg.get("e25_overhead", 0.0235))
        all_rows += rows
        all_cover += cover
    summ, cross = summarise(all_rows)
    files.append(write_csv(os.path.join(out, "e25_theory.csv"), summ))
    files.append(write_csv(os.path.join(out, "e25_crossover.csv"), cross))
    files.append(write_csv(os.path.join(out, "e25_coverage_bound.csv"), all_cover))
    rows, pairs = run_workload(cfg, out)
    for r in rows:
        r["sequences"] = pairs
    files.append(write_csv(os.path.join(out, "e25_workload.csv"), rows))
    return files
