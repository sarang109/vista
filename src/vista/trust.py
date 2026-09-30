"""Trust registry A (Sec 3.1): issuer/attestor keys, validity windows, rotation, revocation, status.

Status semantics for failure injection (E10): the verifier consults a status snapshot. When the
registry is offline the last snapshot is used; if it is older than max_staleness the verifier
fails closed (StatusUnavailable).
"""
from __future__ import annotations

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

    def resolve(self, key_id: str, now: int, role: str | None = None, check_status: bool = True) -> KeyRecord:
        rec = self.keys.get(key_id)
        if rec is None:
            raise TrustError("unknown_key")
        if role and rec.role != role:
            raise TrustError("wrong_key_role")
        if not check_status:
            return rec
        if self.online:
            self.snapshot_time = now
        elif now - self.snapshot_time > self.max_staleness:
            raise TrustError("status_unavailable")
        if not (rec.not_before <= now <= rec.not_after):
            raise TrustError("expired_key")
        if rec.revoked_at is not None and rec.revoked_at <= self.snapshot_time:
            raise TrustError("revoked_key")
        return rec
