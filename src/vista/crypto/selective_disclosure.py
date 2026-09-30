"""Salted-digest selective disclosure (SD-JWT-style) with decoy padding.

An internal evidence attestor (e.g., the authentication server / HSM of the sending FI) issues a
credential per claim group at authorization time: it signs a list of salted digests, one per
threshold claim x_i >= t for every t <= x_i, padded with decoys to the attribute's full chain
length so the digest count does not reveal x_i. The VSAC generator later reveals only the chosen
thresholds. One signature per credential = the fixed charge F_g of N5; one hash per claim = m(c).
"""
from __future__ import annotations

import secrets

from ..util import b64d, b64e, canonical_json, sha256
from . import signed


def _digest(salt: bytes, label: str) -> bytes:
    return sha256(canonical_json([b64e(salt), label]))


def issue(key: signed.SigningKey, group: str, labels: list[str], pad_to: int, binding: dict):
    """Returns (credential dict, disclosures {label: salt_b64})."""
    disclosures = {}
    digests = []
    for lab in labels:
        s = secrets.token_bytes(16)
        disclosures[lab] = b64e(s)
        digests.append(_digest(s, lab))
    while len(digests) < pad_to:
        digests.append(secrets.token_bytes(32))
    digests.sort()
    body = {"typ": "vista-sd-v1", "group": group, "binding": binding,
            "digests": [b64e(d) for d in digests], "kid": key.key_id}
    sig = key.sign(canonical_json(body))
    return {"body": body, "sig": b64e(sig)}, disclosures


def present(cred: dict, disclosures: dict, reveal: list[str]) -> dict:
    return {"cred": cred, "disclosed": [[disclosures[l], l] for l in reveal]}


def verify(pres: dict, public_bytes: bytes, expected_binding: dict) -> list[str]:
    body = pres["cred"]["body"]
    if body.get("typ") != "vista-sd-v1":
        raise ValueError("sd: bad type")
    if body.get("binding") != expected_binding:
        raise ValueError("sd: binding mismatch")
    if not signed.verify(public_bytes, b64d(pres["cred"]["sig"]), canonical_json(body)):
        raise ValueError("sd: bad credential signature")
    digs = set(body["digests"])
    out = []
    for salt, lab in pres["disclosed"]:
        if b64e(_digest(b64d(salt), lab)) not in digs:
            raise ValueError("sd: disclosure not committed")
        out.append(lab)
    return out
