"""Attack transformations (Sec 8.2 item 12, E5) against issued capsules.

Each attack takes a legitimately issued (payment, capsule) and produces what an adversary would
submit to a receiver. All of these must be rejected by a fully configured verifier; ablations
switch individual checks off to show which invariant stops which attack.
"""
from __future__ import annotations

import json

from ..util import b64d, b64e, canonical_json


def _mut(blob: bytes, fn) -> bytes:
    obj = json.loads(blob)
    fn(obj)
    return canonical_json(obj)


def _flip_sig(obj):
    s = bytearray(b64d(obj["sig"]))
    s[5] ^= 0x01
    obj["sig"] = b64e(bytes(s))


MITIGATIONS = {
    "replay_same_payment": "nonce+replay cache",
    "amount_substitution": "PID binding",
    "beneficiary_substitution": "PID binding",
    "currency_substitution": "PID binding",
    "cross_tx_transplant": "PID binding",
    "audience_redirect": "audience binding",
    "expired_capsule": "expiry / freshness",
    "stale_evidence": "per-claim TTL",
    "claim_injection": "envelope signature",
    "claim_upgrade": "envelope signature",
    "claim_stripping": "signature + explicit policy (fail closed)",
    "pid_rewrite": "envelope signature",
    "signature_bitflip": "signature",
    "issuer_substitution": "issuer-institution match",
    "kid_swap": "key resolution + signature",
    "unknown_schema": "schema versioning",
    "version_downgrade": "version check",
    "malformed_json": "parser fail-closed",
    "revoked_issuer_key": "status / revocation",
    "nonce_forgery": "receiver challenge",
    "missing_capsule": "fail closed",
}


def attack_cases(ctx: dict) -> list[dict]:
    """ctx: p, blob, p2 (other payment, same sender/receiver), blob2 (capsule for p2),
    other_receiver_p (same tx re-addressed), foreign_blob (valid capsule by a different issuer),
    stale_blob (capsule built with evidence older than TTL), now, later, revoked_blob, schema."""
    p, blob = ctx["p"], ctx["blob"]
    out = []

    def add(name, payment, capsule, when=None, desc="", mitigation=""):
        out.append({"attack": name, "payment": payment, "blob": capsule, "now": when or ctx["now"],
                    "desc": desc, "mitigation": mitigation})

    add("replay_same_payment", p, blob, desc="resubmit an accepted capsule", mitigation="nonce+replay cache")
    add("amount_substitution", p.with_(amount_minor=p.amount_minor * 10), blob, mitigation="PID binding")
    add("beneficiary_substitution", p.with_(beneficiary_ref="BATTACKER01"), blob, mitigation="PID binding")
    add("currency_substitution", p.with_(currency="EUR"), blob, mitigation="PID binding")
    add("cross_tx_transplant", ctx["p2"], blob, desc="capsule of P1 attached to P2", mitigation="PID binding")
    add("audience_redirect", ctx["other_receiver_p"], blob, desc="same capsule to another receiver",
        mitigation="audience binding")
    add("expired_capsule", p, blob, when=ctx["later"], mitigation="expiry / freshness")
    add("stale_evidence", ctx["stale_p"], ctx["stale_blob"], mitigation="per-claim TTL")
    add("claim_injection", p, _mut(blob, lambda o: o["header"]["claims"].append({"c": "auth_strength>=A3", "t": ctx["now"]})),
        mitigation="envelope signature")
    add("claim_upgrade", p, _mut(blob, lambda o: [c.__setitem__("c", c["c"].replace(">=A1", ">=A3").replace(">=A2", ">=A3"))
                                             for c in o["header"]["claims"]]), mitigation="envelope signature")
    add("claim_stripping", p, _mut(blob, lambda o: o["header"]["claims"].clear()),
        desc="strip claims to force a different decision", mitigation="signature + explicit policy (fail closed)")
    add("pid_rewrite", ctx["p2"], _mut(blob, lambda o: o["header"].__setitem__("pid", ctx["pid2"])),
        mitigation="envelope signature")
    add("signature_bitflip", p, _mut(blob, _flip_sig), mitigation="signature")
    add("issuer_substitution", p, ctx["foreign_blob"], desc="valid capsule signed by another FI's key",
        mitigation="issuer-institution match")
    add("kid_swap", p, _mut(blob, lambda o: o["header"].__setitem__("kid", ctx["foreign_kid"])),
        mitigation="key resolution + signature")
    add("unknown_schema", p, _mut(blob, lambda o: o["header"].__setitem__("schema", "vista-claims-1999.0")),
        mitigation="schema versioning")
    add("version_downgrade", p, _mut(blob, lambda o: o["header"].__setitem__("v", "vsac-0")),
        mitigation="version check")
    add("malformed_json", p, blob[: len(blob) // 2], mitigation="parser fail-closed")
    add("revoked_issuer_key", ctx["revoked_p"], ctx["revoked_blob"], mitigation="status / revocation")
    add("nonce_forgery", ctx["nf_p"], ctx["nf_blob"], desc="capsule with a nonce the receiver never issued",
        mitigation="receiver challenge")
    add("missing_capsule", p, None, desc="strip the VSAC in transit", mitigation="fail closed")
    return out
