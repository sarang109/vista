"""Sending-FI engine (Sec 7, modules [1]-[3]): SEN -> MDPS/selection -> proof/VSAC generation.

Selection modes (Sec 21.4 + baselines Sec 9.2):
  DET   deterministic minimum-cost frontier element (v1.0 MDPS, B9)
  RAND  exponential-mechanism randomised selection with parameter lambda (N2)
  BOP   branch-oblivious proof of Phi as a whole (N2, B8)
  ALL   VSAC without MDPS: disclose every available claim (B4)
  FIXED fixed assurance bundle (B3)
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass

import numpy as np

from . import payment as paymod
from . import man as manmod
from . import vsac
from .audit import MerkleTree, TransparencyLog, evidence_leaves
from .claims import Schema
from .crypto.signed import SigningKey
from .policy import Node, frontier, policy_hash
from .sen import DEFAULT_PROFILE, normalize
from .util import sha256


@dataclass
class BuildOutput:
    blob: bytes | None
    x: tuple
    disclosed: tuple | None
    satisfiable: bool
    reason: str = "ok"


class PolicyCache:
    """Cached policy compilation (Sec 5.2): frontier + per-element costs computed once per policy."""

    def __init__(self, schema: Schema, cost_fn, enabled: bool = True):
        self.schema, self.cost_fn, self.enabled = schema, cost_fn, enabled
        self.store: dict = {}
        self.hits = self.misses = 0

    def get(self, node: Node):
        key = id(node)
        if self.enabled and key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        fr = frontier(node, self.schema.n)
        w = np.array([self.cost_fn(u) for u in fr])
        entry = (fr, w, policy_hash(node, self.schema))
        if self.enabled:
            self.store[key] = entry
        return entry


class SenderFI:
    def __init__(self, inst_id: str, schema: Schema, issuer: SigningKey, attestor: SigningKey, cost_fn,
                 profile: dict | None = None, backend: str = "signed", mode: str = "DET", lam: float = 1.0,
                 groups: dict | None = None, log: TransparencyLog | None = None, lifetime_s: int = 120,
                 fixed_bundle: tuple | None = None, rng=None, cache: bool = True, actions=None):
        self.id, self.schema = inst_id, schema
        self.issuer, self.attestor = issuer, attestor
        self.profile = profile or DEFAULT_PROFILE
        self.backend, self.mode, self.lam = backend, mode, lam
        self.groups, self.log, self.lifetime = groups, log, lifetime_s
        self.fixed_bundle = fixed_bundle
        self.rng = rng or np.random.default_rng()
        self.cache = PolicyCache(schema, cost_fn, cache)
        self.cost_fn = cost_fn
        self.actions = actions

    def select(self, x, fr, w):
        avail = (fr <= np.asarray(x)).all(1)
        if not avail.any():
            return None
        idx = np.flatnonzero(avail)
        if self.mode == "RAND":
            logits = -self.lam * (w[idx] - w[idx].min())
            pr = np.exp(logits)
            pr /= pr.sum()
            return tuple(int(v) for v in fr[idx[self.rng.choice(len(idx), p=pr)]])
        return tuple(int(v) for v in fr[idx[int(np.argmin(w[idx]))]])

    def stepup_disclosure(self, x, fr, policy: Node) -> tuple:
        """Negotiation-aware partial disclosure when Phi(x) = 0: for each frontier branch u disclose
        min(u, x) and keep the branch whose residual step-up deficit is smallest (ties: lower
        disclosure cost). Without this the receiver's MAN plans from an empty claim set and
        rejects payments that one step-up would resolve."""
        if self.actions is None:
            return tuple([0] * self.schema.n)
        best, bkey, seen = tuple([0] * self.schema.n), None, set()
        for u in fr:
            part = tuple(int(min(a, b)) for a, b in zip(u, x))
            if part in seen:
                continue
            seen.add(part)
            pl = manmod.plan(policy, part, self.actions, math.inf, 0.0)
            feasible = pl["p_star"] > 0
            key = (0 if feasible else 1, pl["deficit"] if feasible else 0.0, self.cost_fn(part))
            if bkey is None or key < bkey:
                best, bkey = part, key
        return best

    def build(self, p: paymod.Payment, evidence: list[dict], policy: Node, nonce: str, now: int) -> BuildOutput:
        pid = paymod.pid(p)
        sen = normalize(evidence, pid, now, self.profile, self.schema)
        x = sen.profile
        fr, w, ph = self.cache.get(policy)
        chosen = self.select(x, fr, w)
        sat = chosen is not None
        if self.mode == "ALL":
            chosen = x
        elif self.mode == "FIXED":
            chosen = tuple(min(a, b) for a, b in zip(self.fixed_bundle, x))
        elif self.mode == "BOP":
            if not sat:
                chosen = None
        if chosen is None:
            chosen = self.stepup_disclosure(x, fr, policy)
        claims = self.schema.profile_claims(chosen)
        receipt = None
        if self.log is not None:
            used = sorted({sen.provenance[self.schema.attributes[i].name] for i, _ in claims
                           if self.schema.attributes[i].name in sen.provenance})
            salt = sha256(pid.encode() + secrets.token_bytes(16))
            tree = MerkleTree(evidence_leaves([evidence[k] for k in used], salt))
            r = self.log.append(pid, self.id, tree.root, [self.schema.claim_label(i, t) for i, t in claims], now)
            receipt = dict(r, root=tree.root.hex())
        backend = self.backend
        bop = None
        if self.mode == "BOP":
            if sat:
                backend, bop = "bop", {"front": fr, "policy_hash": ph}
                claims = []
            else:
                backend = "signed"
        blob = vsac.build(p=p, schema=self.schema, issuer=self.issuer, attestor=self.attestor,
                          backend=backend, claims=claims, x=x, event_time=sen.event_time, nonce=nonce,
                          now=now, lifetime_s=self.lifetime, mode=self.mode,
                          lam=str(self.lam) if self.mode == "RAND" else "", groups=self.groups, bop=bop,
                          log_receipt=receipt)
        return BuildOutput(blob, x, chosen if backend != "bop" else None, sat)
