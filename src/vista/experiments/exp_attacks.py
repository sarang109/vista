"""E5 attack resistance with ablations (Sec 9.3): which invariant stops which attack."""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from .. import payment as paymod, vsac
from ..claims import DEFAULT_SCHEMA as S
from ..crypto.signed import SigningKey
from ..simulator import attacks
from ..simulator import workload as wl
from ..util import write_csv

ABLATIONS = {
    "full": {},
    "no_transaction_binding": {"check_binding": False},
    "no_audience_binding": {"check_audience": False},
    "no_freshness": {"check_freshness": False},
    "no_nonce_no_replay_cache": {"check_nonce": False, "check_replay": False},
    "no_status_check": {"check_status": False},
    "no_signature_check": {"check_signature": False},
}


def run(cfg, out, world):
    rng = np.random.default_rng(cfg.get("seed", 1) + 5)
    n_base = cfg.get("e5_base_payments", 150)
    pay = wl.generate_payments(world.pop, world.banks, n_base * 2, 7, rng)
    rows = []
    agg = defaultdict(lambda: [0, 0])
    reasons = defaultdict(lambda: defaultdict(int))
    for backend in cfg.get("e5_backends", ["signed", "sd", "hashchain", "bop"]):
        for j in range(n_base):
            s, r = int(pay["sender"][j]), int(pay["receiver"][j])
            r2 = (r + 1) % len(world.banks)
            if r2 == s:
                r2 = (r2 + 1) % len(world.banks)
            p = world.payment(pay, j)
            p2 = world.payment(pay, j + n_base).with_(sender_fi=p.sender_fi, receiver_fi=p.receiver_fi)
            now = int(pay["ts"][j]) + 1
            pid, pid2 = paymod.pid(p), paymod.pid(p2)
            ev = world.evidence(pay, j, pid, rng)
            if backend == "bop":
                snd = world.sender(s, "BOP")
            else:
                snd = world.sender(s, "ALL", backend)  # disclose everything so injected claims are visible
            node = world.policy_for(r, p.amount_minor)
            issued = {}

            def build(payment, evid, nonce, when, sender=snd):
                return sender.build(payment, evid, node, nonce, when).blob

            nonces = {k: rng.bytes(16).hex() for k in ("main", "stale", "revoked", "redirect")}
            blob = build(p, ev, nonces["main"], now)
            # stale evidence: every auth-derived claim is older than its TTL
            stale_p = p.with_(tx_id=p.tx_id + "S")
            stale_blob = vsac.build(p=stale_p, schema=S, issuer=snd.issuer, attestor=snd.attestor, backend="signed",
                                    claims=[(S.index("auth_strength"), 2)], x=tuple(a.d for a in S.attributes),
                                    event_time={"auth_strength": now - 2000}, nonce=nonces["stale"], now=now,
                                    lifetime_s=3000, mode="DET")
            foreign = world.keys[world.banks[r2].id][0]
            foreign_blob = vsac.build(p=p, schema=S, issuer=foreign, attestor=snd.attestor, backend="signed",
                                      claims=S.profile_claims(tuple(a.d for a in S.attributes)),
                                      x=tuple(a.d for a in S.attributes), event_time={}, nonce=nonces["main"],
                                      now=now, lifetime_s=120, mode="DET")
            rk = SigningKey()
            world.trust.register(p.sender_fi, rk, "issuer", now - 100, now + 10_000)
            world.trust.revoke(rk.key_id, now - 50)
            revoked_p = p.with_(tx_id=p.tx_id + "R")
            revoked_blob = vsac.build(p=revoked_p, schema=S, issuer=rk, attestor=snd.attestor, backend="signed",
                                      claims=[(S.index("auth_strength"), 1)], x=tuple(a.d for a in S.attributes),
                                      event_time={"auth_strength": now}, nonce=nonces["revoked"], now=now,
                                      lifetime_s=120, mode="DET")
            nf_p = p.with_(tx_id=p.tx_id + "N")
            nf_blob = build(nf_p, ev, "f" * 32, now)
            ctx = {"p": p, "blob": blob, "p2": p2, "pid2": pid2,
                   "other_receiver_p": p.with_(receiver_fi=world.banks[r2].id), "now": now, "later": now + 3600,
                   "stale_p": stale_p, "stale_blob": stale_blob, "foreign_blob": foreign_blob,
                   "foreign_kid": foreign.key_id, "revoked_p": revoked_p, "revoked_blob": revoked_blob,
                   "nf_p": nf_p, "nf_blob": nf_blob}
            for case in attacks.attack_cases(ctx):
                if (case["attack"] != "replay_same_payment" and case["blob"] == blob and case["payment"] == p
                        and case["now"] == now):
                    continue  # mutation was a no-op (e.g. stripping claims from a claim-less BOP capsule)
                for abl, flags in ABLATIONS.items():
                    cfgv = vsac.VerifyConfig(**flags)
                    rid = case["payment"].receiver_fi if case["attack"] == "audience_redirect" else p.receiver_fi
                    v = vsac.Verifier(rid, {S.version: S}, world.trust, cfgv, vsac.NonceStore(), None, world.bop_sizes)
                    for nn in nonces.values():
                        v.nonces.live[(p.sender_fi, nn)] = now + 3600
                    if case["attack"] == "replay_same_payment":
                        v.verify(p, blob, now)  # legitimate first use
                    res = v.verify(case["payment"], case["blob"], case["now"])
                    a = agg[(backend, case["attack"], abl)]
                    a[0] += res.ok
                    a[1] += 1
                    reasons[(backend, case["attack"], abl)][res.reason] += 1
            world.trust.keys.pop(rk.key_id, None)
    mitig = attacks.MITIGATIONS
    for (backend, attack, abl), (acc, n) in sorted(agg.items()):
        rs = reasons[(backend, attack, abl)]
        rows.append({"backend": backend, "attack": attack, "ablation": abl, "trials": n, "accepted": acc,
                     "acceptance_rate": acc / n, "expected_defense": mitig.get(attack, ""),
                     "top_reject_reason": max(rs, key=rs.get)})
    return [write_csv(os.path.join(out, "e5_attack_resistance.csv"), rows)]

