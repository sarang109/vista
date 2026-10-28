"""Verifiable Security Assurance Capsule (Sec 4, Algorithm D Sec 5.4).

Wire format: canonical JSON {"header": ..., "proofs": ..., "sig": b64(Ed25519(header||proofs))}.
Invariants enforced by `Verifier` (Sec 4.2), each switchable for ablations (Sec 9.3):
  transaction binding (PID), audience binding, freshness (iat/exp + per-claim TTL),
  single-use receiver nonce, replay cache, schema versioning, issuer key status, fail-closed proofs.
Backends: signed | sd | hashchain | bop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import payment as paymod
from .claims import Schema
from .crypto import hashchain, orproof, selective_disclosure as sd, signed
from .trust import TrustError, TrustRegistry
from .util import b64d, b64e, canonical_json

VERSION = "vsac-1"
BACKENDS = ("signed", "sd", "hashchain", "bop")


# ============================================================================ build (sender side)
def binding(pid: str, aud: str, nonce: str) -> dict:
    return {"pid": pid, "aud": aud, "nonce": nonce}


def build(*, p: paymod.Payment, schema: Schema, issuer: signed.SigningKey, attestor: signed.SigningKey,
          backend: str, claims: list[tuple[int, int]], x, event_time: dict, nonce: str, now: int,
          lifetime_s: int, mode: str, lam: str = "", groups: dict | None = None,
          bop: dict | None = None, log_receipt: dict | None = None) -> bytes:
    """claims: disclosed (i, theta) list (strongest per attribute). x: SEN profile (private).
    bop: {"front": array, "policy_hash": str} for branch-oblivious proofs."""
    pid = paymod.pid(p)
    bind = binding(pid, p.receiver_fi, nonce)
    ttl_exp = [event_time.get(schema.attributes[i].name, now) + schema.attributes[i].ttl_s
               for i, _ in claims if schema.attributes[i].ttl_s]
    exp = min([now + lifetime_s] + ttl_exp)  # capsule validity horizon = min over claims (P8)
    header = {
        "v": VERSION, "pid": pid, "sender": p.sender_fi, "aud": p.receiver_fi, "iat": now, "exp": exp,
        "nonce": nonce, "schema": schema.version, "mode": mode, "lam": lam, "backend": backend,
        "kid": issuer.key_id, "attestor_kid": attestor.key_id,
        # event times are only carried for TTL-bound claims; exact times of static claims
        # (e.g. account opening) would leak far more than the disclosed threshold
        "claims": [dict({"c": schema.claim_label(i, t)},
                        **({"t": int(event_time.get(schema.attributes[i].name, now))}
                           if schema.attributes[i].ttl_s else {})) for i, t in claims],
    }
    if log_receipt:
        header["log"] = log_receipt
    proofs: dict = {}
    if backend == "sd":
        groups = groups or {a.name: a.name for a in schema.attributes}
        by_g: dict = {}
        for i, t in claims:
            by_g.setdefault(groups[schema.attributes[i].name], []).append((i, t))
        pres = []
        for g, cl in sorted(by_g.items()):
            members = [k for k, a in enumerate(schema.attributes) if groups[a.name] == g]
            labels = [schema.claim_label(k, t) for k in members for t in range(1, int(x[k]) + 1)]
            pad = sum(schema.attributes[k].d for k in members)
            cred, disc = sd.issue(attestor, g, labels, pad, bind)
            pres.append(sd.present(cred, disc, [schema.claim_label(i, t) for i, t in cl]))
        proofs["sd"] = pres
    elif backend == "hashchain":
        hc = []
        for i, t in claims:
            name = schema.attributes[i].name
            com, seed = hashchain.commit(attestor, name, int(x[i]), bind)
            hc.append({"c": schema.claim_label(i, t), "com": com,
                       "w": hashchain.prove(seed, int(x[i]), t, name, bind)})
        proofs["hc"] = hc
    elif backend == "bop":
        front = bop["front"]
        header["claims"] = []
        header["policy_hash"] = bop["policy_hash"]
        keys = [orproof.keygen() for _ in range(len(front))]
        pks = [pk for _, pk in keys]
        avail = [j for j, u in enumerate(front) if all(int(a) <= int(b) for a, b in zip(u, x))]
        if not avail:
            raise ValueError("BOP requires Phi(x) = 1")
        j = min(avail)  # any available witness; the proof hides which one
        stmt = {"binding": bind, "policy_hash": bop["policy_hash"], "pks": [orproof.to_hex(v) for v in pks]}
        att_sig = attestor.sign(canonical_json(stmt))
        ctx = canonical_json({"binding": bind, "policy_hash": bop["policy_hash"]})
        cs, zs = orproof.prove(pks, j, keys[j][0], ctx)
        proofs["bop"] = {"stmt": stmt, "att_sig": b64e(att_sig),
                         "cs": [format(c, "x") for c in cs], "zs": [format(z, "x") for z in zs]}
    elif backend != "signed":
        raise ValueError(f"unknown backend {backend}")
    sig = issuer.sign(canonical_json({"header": header, "proofs": proofs}))
    return canonical_json({"header": header, "proofs": proofs, "sig": b64e(sig)})


# ============================================================================ verify (receiver side)
@dataclass
class VerifyConfig:
    check_binding: bool = True
    check_audience: bool = True
    check_freshness: bool = True
    check_nonce: bool = True
    check_replay: bool = True
    check_status: bool = True
    check_signature: bool = True
    check_proofs: bool = True
    require_commitment: bool = False
    clock_skew_s: int = 5
    max_lifetime_s: int = 900
    status_grace: object = None  # optional payment -> allowed status staleness (s), e.g. by amount tier


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    profile: tuple | None = None
    policy_satisfied_bop: bool = False
    claims: list = field(default_factory=list)


class NonceStore:
    """Receiver-issued single-use challenges, pre-fetched in batches by each sender (RTP-friendly)."""

    def __init__(self, ttl_s: int = 3600):
        self.ttl = ttl_s
        self.live: dict = {}

    def issue(self, sender: str, k: int, now: int, rng) -> list[str]:
        out = []
        for _ in range(k):
            n = rng.bytes(16).hex()
            self.live[(sender, n)] = now + self.ttl
            out.append(n)
        return out

    def consume(self, sender: str, nonce: str, now: int) -> bool:
        exp = self.live.pop((sender, nonce), None)
        return exp is not None and now <= exp


class Verifier:
    def __init__(self, receiver_id: str, schemas: dict, trust: TrustRegistry, cfg: VerifyConfig | None = None,
                 nonces: NonceStore | None = None, log=None, bop_policies: dict | None = None):
        self.rid = receiver_id
        self.schemas = schemas  # version -> Schema
        self.trust = trust
        self.cfg = cfg or VerifyConfig()
        self.nonces = nonces or NonceStore()
        self.seen_pids: set = set()
        self.log = log
        self.bop_policies = bop_policies or {}  # policy_hash -> frontier size

    def _fail(self, reason):
        return VerifyResult(False, reason)

    def verify(self, p: paymod.Payment, blob: bytes | None, now: int) -> VerifyResult:
        c = self.cfg
        if blob is None:
            return self._fail("missing_capsule")
        try:
            obj = json.loads(blob)
            h, proofs, sig = obj["header"], obj["proofs"], obj["sig"]
            _ = h["pid"], h["sender"], h["aud"], h["iat"], h["exp"], h["nonce"], h["claims"], h["kid"]
        except Exception:
            return self._fail("malformed")
        if h.get("v") != VERSION:
            return self._fail("unknown_version")
        schema = self.schemas.get(h.get("schema"))
        if schema is None:
            return self._fail("unknown_schema")
        # issuer authentication and status
        try:
            grace = c.status_grace(p) if c.status_grace is not None else None
            rec = self.trust.resolve(h["kid"], now, role="issuer", check_status=c.check_status, max_staleness=grace)
        except TrustError as e:
            return self._fail(e.reason)
        if rec.institution != h["sender"]:
            return self._fail("issuer_mismatch")
        if c.check_signature:
            try:
                ok = signed.verify(rec.public_bytes, b64d(sig), canonical_json({"header": h, "proofs": proofs}))
            except Exception:
                ok = False
            if not ok:
                return self._fail("bad_signature")
        # transaction binding
        pid = paymod.pid(p)
        if c.check_binding and (h["pid"] != pid or h["sender"] != p.sender_fi):
            return self._fail("pid_mismatch")
        if c.check_audience and (h["aud"] != self.rid or p.receiver_fi != self.rid):
            return self._fail("audience_mismatch")
        # freshness
        if c.check_freshness:
            if now + c.clock_skew_s < h["iat"]:
                return self._fail("not_yet_valid")
            if now > h["exp"] or h["exp"] - h["iat"] > c.max_lifetime_s:
                return self._fail("expired")
        claims = []
        try:
            for cl in h["claims"]:
                i, t = schema.parse_claim_label(cl["c"])
                ttl = schema.attributes[i].ttl_s
                if c.check_freshness and ttl and ("t" not in cl or now - int(cl["t"]) > ttl):
                    return self._fail("stale_claim")
                claims.append((i, t))
        except Exception:
            return self._fail("unknown_claim")
        if c.require_commitment:
            lg = h.get("log")
            if not lg or self.log is None:
                return self._fail("missing_commitment")
            try:
                e = self.log.lookup(int(lg["idx"]))
            except Exception:
                return self._fail("bad_commitment")
            if e["entry_hash"] != lg["entry_hash"] or e["pid"] != h["pid"]:
                return self._fail("bad_commitment")
        # nonce + replay (consumed only after every other check passes)
        if c.check_replay and (h["pid"], h["nonce"]) in self.seen_pids:
            return self._fail("replay")
        # backend proofs (fail closed)
        backend = h.get("backend", "signed")
        bind = binding(h["pid"], h["aud"], h["nonce"])
        bop_ok = False
        if c.check_proofs:
            try:
                if backend in ("sd", "hashchain", "bop"):
                    att = self.trust.resolve(h["attestor_kid"], now, role="attestor", check_status=c.check_status,
                                             max_staleness=grace)
                    if att.institution != h["sender"]:
                        return self._fail("attestor_mismatch")
                if backend == "sd":
                    revealed = set()
                    for pres in proofs["sd"]:
                        revealed |= set(sd.verify(pres, att.public_bytes, bind))
                    if not all(cl["c"] in revealed for cl in h["claims"]):
                        return self._fail("bad_proof")
                elif backend == "hashchain":
                    got = {}
                    for item in proofs["hc"]:
                        i, t = schema.parse_claim_label(item["c"])
                        if item["com"]["body"]["attr"] != schema.attributes[i].name:
                            return self._fail("bad_proof")
                        if not hashchain.verify(item["com"], att.public_bytes, t, item["w"], bind):
                            return self._fail("bad_proof")
                        got[item["c"]] = True
                    if not all(cl["c"] in got for cl in h["claims"]):
                        return self._fail("bad_proof")
                elif backend == "bop":
                    b = proofs["bop"]
                    stmt = b["stmt"]
                    if stmt["binding"] != bind or stmt["policy_hash"] != h.get("policy_hash"):
                        return self._fail("bad_proof")
                    if h["policy_hash"] not in self.bop_policies:
                        return self._fail("unknown_policy")
                    if len(stmt["pks"]) != self.bop_policies[h["policy_hash"]]:
                        return self._fail("bad_proof")
                    if not signed.verify(att.public_bytes, b64d(b["att_sig"]), canonical_json(stmt)):
                        return self._fail("bad_proof")
                    pks = [orproof.from_hex(v) for v in stmt["pks"]]
                    ctx = canonical_json({"binding": bind, "policy_hash": h["policy_hash"]})
                    if not orproof.verify(pks, ([int(v, 16) for v in b["cs"]], [int(v, 16) for v in b["zs"]]), ctx):
                        return self._fail("bad_proof")
                    bop_ok = True
                elif backend != "signed":
                    return self._fail("unknown_backend")
            except TrustError as e:
                return self._fail(e.reason)
            except Exception:
                return self._fail("bad_proof")
        if c.check_nonce and not self.nonces.consume(h["sender"], h["nonce"], now):
            return self._fail("nonce_unknown_or_reused")
        if c.check_replay:
            self.seen_pids.add((h["pid"], h["nonce"]))  # a step-up re-submission uses a fresh nonce
        prof = [0] * schema.n
        for i, t in claims:
            prof[i] = max(prof[i], t)
        return VerifyResult(True, "ok", tuple(prof), bop_ok, claims)
