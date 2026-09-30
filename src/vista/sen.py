"""Security Evidence Normalization (Sec 5.1, Prop 1.4 conservative SEN).

Maps institution-specific private evidence E_S to a canonical profile x over DEFAULT_SCHEMA using
an auditable per-institution mapping profile (JSON config, not an ML classifier). SEN is sound
(never over-claims): missing or contradictory evidence yields level 0, and event-based
invalidation (credential reset after the authentication) removes authentication claims (Sec 5.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .claims import DEFAULT_SCHEMA, Schema

DAY = 86_400
CRED_AGE_DAYS = (0, 1, 7, 30, 90)
ACCOUNT_AGE_DAYS = (0, 30, 90, 180, 365, 730)
FRESHNESS_S = (None, 300, 60, 30)
AUTH_TTL_S = 600

DEFAULT_PROFILE = {
    "auth_methods": {
        "fido2_passkey": {"auth_strength": "A3", "phishing_resistant": "true"},
        "totp": {"auth_strength": "A2"},
        "push_number_match": {"auth_strength": "A2"},
        "push": {"auth_strength": "A1"},
        "sms_otp": {"auth_strength": "A1"},
        "password": {"auth_strength": "A1"},
    },
    "kyc_levels": {"basic": "L1", "standard": "L2", "enhanced": "L3"},
    "device_established_days": 30,
    "supports_dynamic_binding": True,
    "supports_policy_attestation": True,
    "supports_device_binding": True,
}


@dataclass
class SENResult:
    profile: tuple
    provenance: dict = field(default_factory=dict)  # attr -> evidence index
    event_time: dict = field(default_factory=dict)  # attr -> unix seconds
    invalidated: list = field(default_factory=list)


def _bucket(v: float, edges) -> int:
    lvl = 0
    for k, e in enumerate(edges):
        if v >= e:
            lvl = k
    return lvl


def normalize(evidence: list[dict], pid: str, now: int, profile: dict | None = None,
              schema: Schema = DEFAULT_SCHEMA) -> SENResult:
    prof = profile or DEFAULT_PROFILE
    x = [0] * schema.n
    prov, evt, inval = {}, {}, []
    I = schema.index

    def put(attr, level, idx, t):
        i = I(attr)
        if level > x[i]:
            x[i] = level
            prov[attr] = idx
            evt[attr] = t

    by_type: dict = {}
    for idx, e in enumerate(evidence):
        by_type.setdefault(e.get("type"), []).append((idx, e))

    last_reset = max((e.get("last_reset_ts", 0) for _, e in by_type.get("credential", [])), default=0)

    for idx, e in by_type.get("kyc", []):
        lab = prof["kyc_levels"].get(e.get("level"))
        if lab:
            put("identity_assurance", schema.level("identity_assurance", lab), idx, e["ts"])

    auths = [(idx, e) for idx, e in by_type.get("auth", []) if 0 <= now - e["ts"] <= AUTH_TTL_S]
    if auths:
        idx, a = max(auths, key=lambda t: t[1]["ts"])
        if last_reset > a["ts"]:
            inval.append("auth_invalidated_by_credential_reset")
        else:
            m = prof["auth_methods"].get(a.get("method"))
            if m:
                for attr, lab in m.items():
                    put(attr, schema.level(attr, lab), idx, a["ts"])
            age = now - a["ts"]
            fl = 0
            for k, lim in enumerate(FRESHNESS_S):
                if lim is not None and age <= lim:
                    fl = k
            if m:
                put("auth_freshness", fl, idx, a["ts"])
            if prof.get("supports_dynamic_binding") and a.get("tx_bound") and a.get("pid") == pid:
                put("dynamic_binding", 1, idx, a["ts"])

    for idx, e in by_type.get("credential", []):
        days = (now - e["last_reset_ts"]) / DAY
        put("credential_age", _bucket(days, CRED_AGE_DAYS), idx, e["last_reset_ts"])

    if prof.get("supports_device_binding"):
        for idx, e in by_type.get("device", []):
            if not e.get("bound"):
                continue
            days = (now - e["first_seen_ts"]) / DAY
            lvl = 2 if days >= prof["device_established_days"] else 1
            put("device_continuity", lvl, idx, e["first_seen_ts"])

    for idx, e in by_type.get("account", []):
        days = (now - e["opened_ts"]) / DAY
        put("account_age", _bucket(days, ACCOUNT_AGE_DAYS), idx, e["opened_ts"])

    if prof.get("supports_policy_attestation"):
        for idx, e in by_type.get("policy_exec", []):
            if e.get("pid") == pid and e.get("result") in ("pass", "review_pass") and 0 <= now - e["ts"] <= AUTH_TTL_S:
                put("fraud_policy_executed", 1, idx, e["ts"])

    return SENResult(tuple(x), prov, evt, inval)
