"""E6 crypto backend performance (signed / sd / hashchain / bop) and E13 BOP cost vs branch count."""
from __future__ import annotations

import os
import time

import numpy as np

from .. import payment as paymod, policy as pol, vsac
from ..claims import DEFAULT_SCHEMA, synthetic_schema
from ..crypto.signed import SigningKey
from ..trust import TrustRegistry
from ..util import percentiles, write_csv

T0 = 1_788_220_800


def _setup(schema):
    iss, att = SigningKey(), SigningKey()
    tr = TrustRegistry()
    tr.register("FI000", iss, "issuer", T0 - 10**6, T0 + 10**7)
    tr.register("FI000", att, "attestor", T0 - 10**6, T0 + 10**7)
    return iss, att, tr


def _payment(j):
    return paymod.Payment("FI000", "FI001", f"P{j:07d}", "B0000001", 12_345 + j, "USD", T0 * 1000, f"TX{j:010d}")


def run_e6(cfg, out):
    S = DEFAULT_SCHEMA
    iss, att, tr = _setup(S)
    rng = np.random.default_rng(cfg.get("seed", 1) + 6)
    x = tuple(a.d for a in S.attributes)
    reps = cfg.get("e6_reps", 200)
    rows = []
    fr_node = pol.parse("(auth_strength >= A3 AND credential_age >= 7d) OR (auth_strength >= A2 AND account_age >= 365d AND device_continuity >= established)", S)
    fr = pol.frontier(fr_node, S.n)
    ph = pol.policy_hash(fr_node, S)
    for backend in ("signed", "sd", "hashchain", "bop"):
        for k in (range(1, S.n + 1) if backend != "bop" else [len(fr)]):
            claims = [(i, x[i]) for i in range(k)]
            gen, ver, size = [], [], []
            ok = 0
            v = vsac.Verifier("FI001", {S.version: S}, tr, vsac.VerifyConfig(), vsac.NonceStore(),
                              bop_policies={ph: len(fr)})
            for j in range(reps):
                p = _payment(j)
                nonce = v.nonces.issue("FI000", 1, T0, rng)[0]
                t = time.perf_counter()
                blob = vsac.build(p=p, schema=S, issuer=iss, attestor=att, backend=backend, claims=claims, x=x,
                                  event_time={}, nonce=nonce, now=T0, lifetime_s=120, mode="DET",
                                  bop={"front": fr, "policy_hash": ph} if backend == "bop" else None)
                gen.append((time.perf_counter() - t) * 1e3)
                t = time.perf_counter()
                r = v.verify(p, blob, T0 + 1)
                ver.append((time.perf_counter() - t) * 1e3)
                ok += r.ok
                size.append(len(blob))
            g, vv = percentiles(gen), percentiles(ver)
            rows.append({"backend": backend, "claims_or_branches": k, "reps": reps, "verified_ok": ok,
                         "gen_p50_ms": g["p50"], "gen_p99_ms": g["p99"], "verify_p50_ms": vv["p50"],
                         "verify_p99_ms": vv["p99"], "bytes_mean": float(np.mean(size))})
    return [write_csv(os.path.join(out, "e6_crypto_backends.csv"), rows)]


def run_e13(cfg, out):
    sch = synthetic_schema(16, 2, "bop-bench")
    iss, att, tr = _setup(sch)
    rng = np.random.default_rng(cfg.get("seed", 1) + 13)
    reps = cfg.get("e13_reps", 60)
    rows = []
    x = tuple([1] * 16)
    for b in (1, 2, 3, 4, 6, 8, 12, 16):
        node = pol.Atom(0, 1) if b == 1 else pol.Or(tuple(pol.Atom(i, 1) for i in range(b)))
        fr = pol.frontier(node, sch.n)
        ph = pol.policy_hash(node, sch)
        for backend in ("bop", "signed_cheapest_branch"):
            v = vsac.Verifier("FI001", {sch.version: sch}, tr, vsac.VerifyConfig(), vsac.NonceStore(),
                              bop_policies={ph: len(fr)})
            gen, ver, size, ok = [], [], [], 0
            for j in range(reps):
                p = _payment(j)
                nonce = v.nonces.issue("FI000", 1, T0, rng)[0]
                t = time.perf_counter()
                blob = vsac.build(p=p, schema=sch, issuer=iss, attestor=att,
                                  backend="bop" if backend == "bop" else "signed",
                                  claims=[] if backend == "bop" else [(0, 1)], x=x, event_time={}, nonce=nonce,
                                  now=T0, lifetime_s=120, mode=backend,
                                  bop={"front": fr, "policy_hash": ph} if backend == "bop" else None)
                gen.append((time.perf_counter() - t) * 1e3)
                t = time.perf_counter()
                ok += v.verify(p, blob, T0 + 1).ok
                ver.append((time.perf_counter() - t) * 1e3)
                size.append(len(blob))
            rows.append({"branches": b, "backend": backend, "verified_ok": ok, "reps": reps,
                         "gen_p50_ms": float(np.median(gen)), "verify_p50_ms": float(np.median(ver)),
                         "gen_p99_ms": float(np.percentile(gen, 99)), "verify_p99_ms": float(np.percentile(ver, 99)),
                         "bytes_mean": float(np.mean(size))})
    return [write_csv(os.path.join(out, "e13_bop_cost.csv"), rows)]
