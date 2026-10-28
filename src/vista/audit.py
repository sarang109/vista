"""Foundation N8: commit-and-audit deterrence for dishonest issuers (Sec 27).

* deterrence law  q* = g / (delta (g + F))           (Prop 8.1)
* Merkle commitment over evidence leaves bound to PID (Prop 8.2 enabler)
* append-only hash-chained transparency log L
* audit sampling with unpredictable (keyed-hash / VRF-style) randomness
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading

from .util import canonical_json, sha256


def q_star(g: float, F: float, delta: float) -> float:
    if delta <= 0:
        return 1.0
    return min(1.0, g / (delta * (g + F)))


def lying_profitable(g: float, F: float, q: float, delta: float) -> bool:
    return (1 - q * delta) * g - q * delta * F > 0


# ---------------------------------------------------------------------------- Merkle
def _leaf(b: bytes) -> bytes:
    return sha256(b"\x00" + b)


def _node(a: bytes, b: bytes) -> bytes:
    return sha256(b"\x01" + a + b)


class MerkleTree:
    def __init__(self, leaves: list[bytes]):
        if not leaves:
            leaves = [b""]
        self.levels = [[_leaf(x) for x in leaves]]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = [_node(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i]) for i in range(0, len(cur), 2)]
            self.levels.append(nxt)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, idx: int) -> list[tuple[bytes, bool]]:
        out = []
        for lvl in self.levels[:-1]:
            sib = idx ^ 1
            sib_h = lvl[sib] if sib < len(lvl) else lvl[idx]
            out.append((sib_h, idx % 2 == 0))
            idx //= 2
        return out

    @staticmethod
    def verify(leaf: bytes, proof, root: bytes) -> bool:
        h = _leaf(leaf)
        for sib, left in proof:
            h = _node(h, sib) if left else _node(sib, h)
        return h == root


def evidence_leaves(evidence_items: list[dict], salt: bytes) -> list[bytes]:
    return [salt + canonical_json(e) for e in evidence_items]


# ---------------------------------------------------------------------------- transparency log
class TransparencyLog:
    """Append-only hash chain. Optional JSONL persistence. Thread-safe.
    retain=False keeps only the chain head, entry count and byte count (constant memory for long streams)."""

    def __init__(self, path: str | None = None, retain: bool = True):
        self.entries: list[dict] = []
        self.retain = retain
        self.count = 0
        self.bytes = 0
        self.head = b"\x00" * 32
        self.path = path
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            os.remove(path)

    def append(self, pid: str, sender: str, root: bytes, claim_ids: list[str], ts: int) -> dict:
        with self._lock:
            body = {"idx": self.count, "pid": pid, "sender": sender, "root": root.hex(),
                    "claims": sorted(claim_ids), "ts": ts, "prev": self.head.hex()}
            h = sha256(canonical_json(body))
            self.head = h
            entry = dict(body, entry_hash=h.hex())
            self.count += 1
            self.bytes += len(canonical_json(entry))
            if self.retain:
                self.entries.append(entry)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            return {"idx": entry["idx"], "entry_hash": entry["entry_hash"]}

    def verify_chain(self) -> bool:
        prev = b"\x00" * 32
        for e in self.entries:
            body = {k: e[k] for k in ("idx", "pid", "sender", "root", "claims", "ts", "prev")}
            if body["prev"] != prev.hex() or sha256(canonical_json(body)).hex() != e["entry_hash"]:
                return False
            prev = bytes.fromhex(e["entry_hash"])
        return True

    def lookup(self, idx: int) -> dict:
        return self.entries[idx]


def audit_selected(pid: str, q: float, auditor_key: bytes) -> bool:
    """Unpredictable-to-sender sampling: HMAC(auditor_key, pid) < q * 2^256 (VRF stand-in)."""
    v = int.from_bytes(hmac.new(auditor_key, pid.encode(), hashlib.sha256).digest(), "big")
    return v < int(q * (1 << 256))


def new_auditor_key() -> bytes:
    return secrets.token_bytes(32)
