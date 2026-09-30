"""Hash-chain threshold proofs (in the spirit of PayWord / HashWires).

The attestor commits top = H^x(seed) for a fresh per-capsule seed and signs it with a binding.
To prove x >= theta, reveal w = H^(x - theta)(seed); the verifier checks H^theta(w) == top.
Producing a witness for theta > x requires inverting H. The verifier learns only x >= theta
(w is pseudorandom). An auditor given the seed can later check the exact committed x (N8).
"""
from __future__ import annotations

import hashlib
import secrets

from ..util import b64d, b64e, canonical_json
from . import signed


def _h(ctx: bytes, v: bytes) -> bytes:
    return hashlib.sha256(ctx + v).digest()


def _iter(ctx: bytes, v: bytes, k: int) -> bytes:
    for _ in range(k):
        v = _h(ctx, v)
    return v


def commit(key: signed.SigningKey, attr: str, x: int, binding: dict):
    ctx = canonical_json({"attr": attr, "binding": binding})
    seed = secrets.token_bytes(32)
    top = _iter(ctx, seed, x)
    body = {"typ": "vista-hc-v1", "attr": attr, "binding": binding, "top": b64e(top), "kid": key.key_id}
    return {"body": body, "sig": b64e(key.sign(canonical_json(body)))}, seed


def prove(seed: bytes, x: int, theta: int, attr: str, binding: dict) -> str:
    if theta > x:
        raise ValueError("cannot prove a false threshold")
    ctx = canonical_json({"attr": attr, "binding": binding})
    return b64e(_iter(ctx, seed, x - theta))


def verify(commitment: dict, public_bytes: bytes, theta: int, witness: str, expected_binding: dict) -> bool:
    body = commitment["body"]
    if body.get("typ") != "vista-hc-v1" or body.get("binding") != expected_binding:
        return False
    if not signed.verify(public_bytes, b64d(commitment["sig"]), canonical_json(body)):
        return False
    ctx = canonical_json({"attr": body["attr"], "binding": expected_binding})
    return b64e(_iter(ctx, b64d(witness), theta)) == body["top"]


def audit_open(commitment: dict, seed: bytes, x_claimed: int, binding: dict) -> bool:
    ctx = canonical_json({"attr": commitment["body"]["attr"], "binding": binding})
    return b64e(_iter(ctx, seed, x_claimed)) == commitment["body"]["top"]
