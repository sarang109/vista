"""Canonical claim schema (Sec 3.2) as a product of finite assurance chains (N1, Sec 20.2).

Attribute i has a chain D_i = {0 < 1 < ... < d_i}; level 0 means "no assurance".
An atomic claim (i, theta), theta >= 1, asserts x_i >= theta. A disclosure profile is a
vector l in D; the sender can support l iff l <= x (product order).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Attribute:
    name: str
    levels: tuple  # labels; index 0 = no assurance
    domain: str = ""
    ttl_s: int | None = None  # freshness TTL for claims on this attribute (None = no TTL)

    @property
    def size(self) -> int:
        return len(self.levels)

    @property
    def d(self) -> int:
        return len(self.levels) - 1


class Schema:
    def __init__(self, version: str, attributes):
        self.version = version
        self.attributes = tuple(attributes)
        self._idx = {a.name: i for i, a in enumerate(self.attributes)}
        if len(self._idx) != len(self.attributes):
            raise ValueError("duplicate attribute names")

    @property
    def n(self) -> int:
        return len(self.attributes)

    @property
    def dims(self) -> tuple:
        return tuple(a.size for a in self.attributes)

    def index(self, name: str) -> int:
        if name not in self._idx:
            raise KeyError(f"unknown attribute {name!r} in schema {self.version}")
        return self._idx[name]

    def level(self, name: str, label: str) -> int:
        a = self.attributes[self.index(name)]
        labels = [str(x).lower() for x in a.levels]
        lab = str(label).lower()
        if lab not in labels:
            raise KeyError(f"unknown level {label!r} for attribute {name}; valid: {a.levels}")
        return labels.index(lab)

    def claim_label(self, i: int, theta: int) -> str:
        a = self.attributes[i]
        return f"{a.name}>={a.levels[theta]}"

    def parse_claim_label(self, s: str) -> tuple[int, int]:
        name, lab = s.split(">=", 1)
        i = self.index(name.strip())
        return i, self.level(name.strip(), lab.strip())

    def profile_claims(self, prof) -> list[tuple[int, int]]:
        return [(i, int(t)) for i, t in enumerate(prof) if t > 0]

    def profile_labels(self, prof) -> list[str]:
        return [self.claim_label(i, t) for i, t in self.profile_claims(prof)]

    def num_states(self) -> int:
        return int(np.prod(self.dims))

    def states(self) -> np.ndarray:
        return np.array(list(itertools.product(*[range(s) for s in self.dims])), dtype=np.int16)

    def all_claims(self) -> list[tuple[int, int]]:
        return [(i, t) for i, a in enumerate(self.attributes) for t in range(1, a.size)]


# The canonical interbank schema used by the prototype (Sec 3.2 table).
DEFAULT_SCHEMA = Schema(
    "vista-claims-2026.1",
    [
        Attribute("identity_assurance", ("L0", "L1", "L2", "L3"), "identity"),
        Attribute("auth_strength", ("A0", "A1", "A2", "A3"), "authentication", ttl_s=600),
        Attribute("phishing_resistant", ("false", "true"), "authentication", ttl_s=600),
        Attribute("credential_age", ("0d", "1d", "7d", "30d", "90d"), "credential"),
        Attribute("device_continuity", ("none", "new", "established"), "device"),
        Attribute("dynamic_binding", ("false", "true"), "authorization", ttl_s=600),
        Attribute("auth_freshness", ("stale", "300s", "60s", "30s"), "authorization", ttl_s=300),
        Attribute("account_age", ("0d", "30d", "90d", "180d", "365d", "730d"), "account"),
        Attribute("fraud_policy_executed", ("false", "true"), "control_execution", ttl_s=600),
    ],
)


def synthetic_schema(n: int, levels: int, version: str | None = None) -> Schema:
    """Synthetic schema for scaling experiments: n attributes with `levels` levels each."""
    return Schema(version or f"synthetic-{n}x{levels}",
                  [Attribute(f"a{i}", tuple(str(k) for k in range(levels))) for i in range(n)])


def join(a, b):
    return tuple(max(x, y) for x, y in zip(a, b))


def leq(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b))
