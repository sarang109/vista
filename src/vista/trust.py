"""Trust registry A (Sec 3.1): issuer/attestor keys, validity windows, rotation, revocation, status.

Status semantics for failure injection (E10): the verifier consults a status snapshot. When the
registry is offline the last snapshot is used; if it is older than max_staleness the verifier
fails closed (StatusUnavailable).

Risk-bounded stale-status grace: a verifier may accept a snapshot up to tau seconds old for low-value tiers.
If revocations of a key arrive as a Poisson process with rate lam_r, a payment verified at staleness s <= tau
uses a key revoked after the snapshot with probability 1 - exp(-lam_r s) <= 1 - exp(-lam_r tau) = eps, so the
expected value accepted on revoked keys is at most eps times the value accepted under grace.
"""
from __future__ import annotations

import math

from dataclasses import dataclass

from .crypto.signed import SigningKey


class TrustError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class KeyRecord:
    institution: str
    key_id: str
    public_bytes: bytes
    role: str  # "issuer" or "attestor"
    not_before: int
    not_after: int
    revoked_at: int | None = None
    tier: int = 1


class TrustRegistry:
    def __init__(self, max_status_staleness_s: int = 300):
        self.keys: dict[str, KeyRecord] = {}
        self.online = True
        self.snapshot_time = 0
        self.max_staleness = max_status_staleness_s
        self.profiles: dict[str, dict] = {}  # institution -> certified assurance profile / tier

    def register(self, institution: str, key: SigningKey, role: str, not_before: int, not_after: int,
                 tier: int = 1) -> KeyRecord:
        rec = KeyRecord(institution, key.key_id, key.public_bytes, role, not_before, not_after, None, tier)
        self.keys[key.key_id] = rec
        return rec

    def revoke(self, key_id: str, at: int):
        self.keys[key_id].revoked_at = at

    def rotate(self, institution: str, role: str, new_key: SigningKey, now: int, validity: int,
               grace: int = 60) -> KeyRecord:
        for r in self.keys.values():
            if r.institution == institution and r.role == role and r.not_after > now + grace:
                r.not_after = now + grace
        return self.register(institution, new_key, role, now, now + validity)

    def refresh(self, now: int):
        if self.online:
            self.snapshot_time = now

    def resolve(self, key_id: str, now: int, role: str | None = None, check_status: bool = True,
                max_staleness: int | None = None) -> KeyRecord:
        """max_staleness overrides the registry default for this lookup (stale-status grace)."""
        rec = self.keys.get(key_id)
        if rec is None:
            raise TrustError("unknown_key")
        if role and rec.role != role:
            raise TrustError("wrong_key_role")
        if not check_status:
            return rec
        if self.online:
            self.snapshot_time = now
        elif now - self.snapshot_time > (self.max_staleness if max_staleness is None else max_staleness):
            raise TrustError("status_unavailable")
        if not (rec.not_before <= now <= rec.not_after):
            raise TrustError("expired_key")
        if rec.revoked_at is not None and rec.revoked_at <= self.snapshot_time:
            raise TrustError("revoked_key")
        return rec


def grace_window(eps: float, revocation_rate_per_s: float) -> float:
    """Largest staleness tau with P(key revoked within tau) <= eps: tau = -ln(1 - eps) / lam_r."""
    return -math.log1p(-eps) / revocation_rate_per_s


def unsafe_probability(staleness_s: float, revocation_rate_per_s: float) -> float:
    return -math.expm1(-revocation_rate_per_s * max(staleness_s, 0.0))
