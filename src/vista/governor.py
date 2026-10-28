"""Integrated disclosure governor (IDG): one sender-side component through which every disclosure passes.

Offline, once per policy and prior epoch (Algorithm 1, PCGS-Compile):
    frontier M, signatures with their prior mass, governor cost C(v) = base cost + fixed-charge proof cost,
    budget B = (1 + overhead) E[C] under DET, minimum-Lambda kernel table Q[a][v] (pcgs.SignatureSpace.compile)

Online, per payment (Algorithm 2, IDG-Select), in this fixed order:
    1. signature a = A(x)
    2. sticky lookup on key = (sender account, receiver policy, evidence epoch); on a miss, or when the stored
       branch is no longer supported, draw from Q[a] with a keyed PRF (reproducible for audit)
    3. interval ledger: every threshold claim of the branch must be answerable; on a refusal fall back to the
       next-cheapest supported branch, and if none is answerable return a forced failure
    4. fixed-charge cost of the disclosed claims (credential groups touched)
    5. policy-extraction budget: counts disclosures and gates step-up partial disclosures (probes)
    6. audit: hash-chained commitment to (branch, epoch, policy)
The online step is O(m) plus one hash lookup. Each sender institution runs its own governor, so state is
sharded by sender and never shared across institutions.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field

import numpy as np

from . import ledger as led
from .audit import TransparencyLog
from .group_cost import layout_cost
from .pcgs import SignatureSpace
from .policy import Node, frontier, policy_hash
from .policy_extraction import QueryBudget
from .repeated import StickyStore, draw
from .util import canonical_json, sha256


@dataclass
class GovernorConfig:
    name: str = "IDG"
    kernel: str = "pcgs"  # pcgs | exp | det
    overhead: float = 0.0235  # PCGS cost budget relative to DET
    lam: float = 1.0  # exponential-mechanism parameter when kernel == "exp"
    sticky: bool = True
    ledger_rho: dict | None = None  # attribute name -> ledger resolution rho; None disables the ledger
    collusion_aware: bool = False  # one ledger per (account, attribute) across all receivers
    fixed_charge: dict | None = None  # {"weight": w, "F": {group: F_g}, "per_claim": c}; None disables
    budget: bool = False
    probe_limit: int | None = None
    audit: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GovernorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def evidence_epoch(evidence: list[dict]) -> str:
    """Epoch of the sender evidence: changes on reset-type events (credential reset, device change), so a
    sticky choice never outlives the evidence it was drawn for."""
    cred = max((e.get("last_reset_ts", 0) for e in evidence if e.get("type") == "credential"), default=0)
    dev = [(e.get("first_seen_ts", 0), bool(e.get("bound"))) for e in evidence if e.get("type") == "device"]
    return sha256(canonical_json({"cred": cred, "dev": sorted(dev)})).hex()[:16]


class PolicyTable:
    """Compiled per-policy state: frontier, governor cost, kernel table keyed by signature code."""

    def __init__(self, node: Node, schema, prior, base_cost, cfg: GovernorConfig, layout: dict | None):
        t0 = time.perf_counter()
        self.front = frontier(node, schema.n)
        self.hash = policy_hash(node, schema)
        names = [a.name for a in schema.attributes]
        self.base = np.array([base_cost(u) for u in self.front], dtype=float)
        self.fixed = np.zeros(len(self.front))
        if cfg.fixed_charge and layout:
            fc = cfg.fixed_charge
            self.fixed = np.array([layout_cost(u, names, layout, fc.get("F", {}), fc.get("per_claim", 0.0))
                                   for u in self.front])
        w = cfg.fixed_charge.get("weight", 1.0) if cfg.fixed_charge else 0.0
        self.C = self.base + w * self.fixed
        self.order = np.argsort(self.C, kind="stable")
        self.bits = 1 << np.arange(len(self.front), dtype=np.int64)
        avail = (prior.states[:, None, :] >= self.front[None, :, :]).all(2)
        self.ss = SignatureSpace.from_states(avail, prior.p, self.C)
        self.kernel, self.r, self.mu, self.compiled = cfg.kernel, None, 0.0, None
        if cfg.kernel == "pcgs" and len(self.front) > 1:
            self.compiled = self.ss.compile(cfg.overhead)
            Q, self.r, self.mu = self.compiled.Q, self.compiled.r, self.compiled.mu
        elif cfg.kernel == "exp":
            Q = self.ss.kernel_exp(cfg.lam)
        else:
            Q = self.ss.kernel_det()
        self.lam = cfg.lam
        self.rows = {int(c): Q[k] for k, c in enumerate(self.ss.codes)}
        self.Q = Q
        self.compile_s = time.perf_counter() - t0

    def probs(self, avail: np.ndarray) -> np.ndarray:
        code = int(avail.astype(np.int64) @ self.bits)
        row = self.rows.get(code)
        if row is not None:
            return row
        # a signature with no prior mass: the Gibbs form (or the baseline kernel) extends to it directly
        if self.kernel == "pcgs" and self.r is not None:
            up = np.maximum(self.ss.up, 1e-300)
            lg = np.log(np.maximum(self.r, 1e-300)) - self.mu * (self.C - self.C.min()) - np.log(up)
        elif self.kernel == "exp":
            lg = -self.lam * (self.C - self.C.min())
        else:
            p = np.zeros(len(self.C))
            p[self.order[np.flatnonzero(avail[self.order])[0]]] = 1.0
            return p
        lg = np.where(avail, lg, -np.inf)
        z = np.exp(lg - lg.max())
        return z / z.sum()


class Governor:
    def __init__(self, sender_id: str, schema, prior, base_cost, cfg: GovernorConfig, layout: dict | None = None,
                 prf_key: bytes | None = None, rng=None):
        self.id, self.schema, self.prior, self.base_cost = sender_id, schema, prior, base_cost
        self.cfg, self.layout = cfg, layout
        self.prf_key = prf_key or sha256(f"governor|{sender_id}".encode())
        self.rng = rng or np.random.default_rng()
        self.tables: dict = {}
        self.sticky = StickyStore(self.prf_key) if cfg.sticky else None
        self.rho = {schema.index(k): int(v) for k, v in (cfg.ledger_rho or {}).items() if int(v) > 1}
        self.ledgers: dict = {}
        self.budget = QueryBudget(cfg.probe_limit if cfg.budget else None)
        self.log = TransparencyLog(retain=False) if cfg.audit else None
        self.stats = {"selections": 0, "fallbacks": 0, "ledger_refusals": 0, "redraws": 0, "hits": 0, "first": 0}

    def table(self, node: Node) -> PolicyTable:
        t = self.tables.get(id(node))
        if t is None:
            t = self.tables[id(node)] = PolicyTable(node, self.schema, self.prior, self.base_cost, self.cfg,
                                                    self.layout)
        return t

    # ------------------------------------------------------------------ ledger
    def _ledger(self, account, attr: int, audience) -> led.IntervalLedger:
        key = (account, attr, None if self.cfg.collusion_aware else audience)
        L = self.ledgers.get(key)
        if L is None:
            L = self.ledgers[key] = led.IntervalLedger(0, self.schema.attributes[attr].d, self.rho[attr])
        return L

    def _answerable(self, account, audience, prof) -> bool:
        return all(self._ledger(account, i, audience).decide(int(t)) != "refuse"
                   for i, t in enumerate(prof) if t > 0 and i in self.rho)

    def _commit(self, account, audience, prof, x):
        for i, t in enumerate(prof):
            if t > 0 and i in self.rho:
                self._ledger(account, i, audience).answer(int(x[i]), int(t))

    # ------------------------------------------------------------------ Algorithm 2
    def select(self, pid: str, account, audience, node: Node, x, epoch: str, now: int = 0):
        """Returns (profile or None, info). None with info['event'] == 'unsat' when x satisfies no branch and
        'ledger_refused' when every supported branch would breach the ledger (forced failure)."""
        t = self.table(node)
        avail = (t.front <= np.asarray(x)).all(1)
        if not avail.any():
            return None, {"event": "unsat"}
        probs = t.probs(avail)
        if self.sticky is not None:
            v, event = self.sticky.choose((str(account), t.hash, epoch), avail, probs)
            self.stats[{"hit": "hits", "first": "first", "redraw": "redraws"}[event]] += 1
        else:
            v, event = draw(probs, self.rng.random()), "fresh"
        info = {"event": event, "branch": int(v), "fallback": False, "n_branches": len(t.front)}
        if self.rho:
            if not self._answerable(account, audience, t.front[v]):
                alt = [int(u) for u in t.order if avail[u] and u != v and self._answerable(account, audience, t.front[u])]
                if not alt:
                    self.stats["ledger_refusals"] += 1
                    info["event"] = "ledger_refused"
                    return None, info
                v = alt[0]
                info.update(branch=v, fallback=True)
                self.stats["fallbacks"] += 1
            self._commit(account, audience, t.front[v], x)
        info["fixed_charge"] = float(t.fixed[v])
        info["cost"] = float(t.C[v])
        if self.cfg.budget:
            self.budget.count()
        if self.log is not None:
            salt = hmac.new(self.prf_key, pid.encode(), hashlib.sha256).digest()
            root = sha256(canonical_json({"v": int(v), "epoch": epoch, "policy": t.hash}) + salt)
            info["receipt"] = self.log.append(pid, self.id, root, [], now)
        self.stats["selections"] += 1
        return tuple(int(a) for a in t.front[v]), info

    def allow_probe(self, account, audience, node: Node, epoch: str) -> bool:
        """Gate for a step-up partial disclosure (a membership query on the receiver policy)."""
        if not self.cfg.budget:
            return True
        return self.budget.allow_probe((str(account), str(audience), self.table(node).hash, epoch))
