"""Deterministic simulated interbank world: banks, keys, trust registry, policies, payers, engines.

Every parallel worker rebuilds an identical World from (config, seed), so keys and policies agree
across processes without shared state.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np

from .. import payment as paymod
from ..claims import DEFAULT_SCHEMA
from ..crypto.signed import SigningKey
from ..governor import Governor, GovernorConfig
from ..man import Action
from ..policy import frontier, parse, policy_hash
from ..prior import Prior
from ..receiver import ReceiverFI
from ..sen import normalize
from ..sender import SenderFI
from ..trust import TrustRegistry
from ..util import sha256
from ..vsac import NonceStore, Verifier, VerifyConfig
from . import workload as wl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
S = DEFAULT_SCHEMA
GROUPED = {"auth_strength": "auth", "phishing_resistant": "auth", "dynamic_binding": "auth",
           "auth_freshness": "auth", "credential_age": "lifecycle", "device_continuity": "lifecycle",
           "identity_assurance": "kyc_account", "account_age": "kyc_account", "fraud_policy_executed": "policy"}
FRAGMENTED = {a.name: a.name for a in S.attributes}

# Step-up catalog (Sec 25): friction k, planning completion probability p, claims produced,
# simulated user time (s). Payer capability decides whether an attempt can actually succeed.
ACTIONS = [
    Action("passkey_reauth", 2.0, 0.90, ((S.index("auth_strength"), 3), (S.index("phishing_resistant"), 1),
                                          (S.index("auth_freshness"), 3), (S.index("dynamic_binding"), 1))),
    Action("totp_reauth", 2.5, 0.85, ((S.index("auth_strength"), 2), (S.index("auth_freshness"), 3),
                                       (S.index("dynamic_binding"), 1))),
    Action("sms_otp_reauth", 1.5, 0.90, ((S.index("auth_strength"), 1), (S.index("auth_freshness"), 3),
                                          (S.index("dynamic_binding"), 1))),
    Action("selfie_kyc_refresh", 4.0, 0.70, ((S.index("identity_assurance"), 3),)),
    Action("fraud_review", 6.0, 0.80, ((S.index("fraud_policy_executed"), 1),)),
]
ACTION_SECONDS = {"passkey_reauth": 8, "totp_reauth": 15, "sms_otp_reauth": 25, "selfie_kyc_refresh": 60,
                  "fraud_review": 300}


class World:
    def __init__(self, n_banks: int = 50, n_payers: int = 50_000, seed: int = 7, prior_sample: int = 20_000,
                 cost_claim_weight: float = 0.05):
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.arche = wl.load_archetypes(os.path.join(ROOT, "configs", "banks", "archetypes.json"))
        with open(os.path.join(ROOT, "configs", "policies", "policy_library.json")) as f:
            lib = json.load(f)
        self.tiers = lib["tiers_usd"]
        self.policies = {st: {t: parse(txt, S) for t, txt in tiers.items()} for st, tiers in lib["stances"].items()}
        self.policy_text = lib["stances"]
        self.banks = wl.generate_banks(n_banks, self.arche, rng)
        self.pop = wl.Population(n_payers, self.banks, self.arche, rng)
        self.trust = TrustRegistry()
        self.keys = {}
        for b in self.banks:
            iss = SigningKey.from_seed(sha256(f"{seed}|{b.id}|issuer".encode()))
            att = SigningKey.from_seed(sha256(f"{seed}|{b.id}|attestor".encode()))
            self.trust.register(b.id, iss, "issuer", wl.T0 - 30 * wl.DAY, wl.T0 + 365 * wl.DAY)
            self.trust.register(b.id, att, "attestor", wl.T0 - 30 * wl.DAY, wl.T0 + 365 * wl.DAY)
            self.keys[b.id] = (iss, att)
        self.bop_sizes = {policy_hash(n, S): len(frontier(n, S.n))
                          for tiers in self.policies.values() for n in tiers.values()}
        self.prior = self._estimate_prior(rng, prior_sample)
        self.cost_claim_weight = cost_claim_weight
        self._senders, self._receivers = {}, {}
        self.planning_actions = self._calibrated_actions()

    def _calibrated_actions(self) -> list[Action]:
        """Receiver-side planning catalog: completion probability = p(complete | capable) x population
        capability rate, as a receiver would estimate from its own step-up history."""
        share = np.array([b.share for b in self.banks])
        policy_ok = sum(sh for b, sh in zip(self.banks, share)
                        if self.arche["archetypes"][b.archetype]["profile"]["supports_policy_attestation"])
        rate = {"passkey_reauth": float(self.pop.has_passkey.mean()), "totp_reauth": float(self.pop.has_totp.mean()),
                "sms_otp_reauth": 1.0, "selfie_kyc_refresh": float(self.pop.can_selfie.mean()),
                "fraud_review": float(policy_ok / share.sum())}
        return [Action(a.name, a.k, a.p * rate[a.name], a.delta) for a in ACTIONS]

    # ------------------------------------------------------------------ population prior (receiver model)
    def _estimate_prior(self, rng, k: int) -> Prior:
        idx = rng.choice(self.pop.n, size=min(k, self.pop.n), replace=False)
        pr = Prior(S.dims)
        counts = np.full(len(pr.states), 0.01)  # Laplace smoothing
        for i in idx:
            ts = int(wl.T0 + rng.integers(0, 7 * wl.DAY))
            b = self.banks[int(self.pop.bank[i])]
            ev = wl.evidence_for(self.pop, int(i), ts, 0, "pilot", self.arche["archetypes"][b.archetype], rng)
            x = normalize(ev, "pilot", ts + 1, self.arche["archetypes"][b.archetype]["profile"], S).profile
            counts[pr.index_of(x)] += 1
        pr.p = counts / counts.sum()
        return pr

    @lru_cache(maxsize=100_000)
    def surprisal(self, prof: tuple) -> float:
        return self.prior.surprisal(prof)

    def cost(self, prof) -> float:
        prof = tuple(int(v) for v in prof)
        return self.surprisal(prof) + self.cost_claim_weight * sum(1 for v in prof if v > 0)

    # ------------------------------------------------------------------ policies
    def tier(self, amount_minor: int) -> str:
        usd = amount_minor / 100
        if usd < self.tiers["micro"]:
            return "micro"
        if usd < self.tiers["standard"]:
            return "standard"
        if usd < self.tiers["high"]:
            return "high"
        return "very_high"

    def policy_for(self, receiver_idx: int, amount_minor: int):
        return self.policies[self.banks[receiver_idx].stance][self.tier(amount_minor)]

    # ------------------------------------------------------------------ engines
    def sender(self, b: int, mode: str = "DET", backend: str | None = None, lam: float = 1.0, log=None,
               cache: bool = True, gov: GovernorConfig | None = None) -> SenderFI:
        """gov: a GovernorConfig for mode 'IDG' (each sender institution runs its own governor)."""
        key = (b, mode, backend, lam, id(log), cache, gov.name if gov else None)
        if key not in self._senders:
            bank = self.banks[b]
            A = self.arche["archetypes"][bank.archetype]
            iss, att = self.keys[bank.id]
            layout = GROUPED if A["credential_layout"] == "grouped" else FRAGMENTED
            fixed = tuple(min(3, a.d) if a.name in ("auth_strength",) else a.d if a.d == 1 else min(2, a.d)
                          for a in S.attributes)
            governor = None
            if gov is not None:
                governor = Governor(bank.id, S, self.prior, self.cost, gov, layout,
                                    sha256(f"{self.seed}|{bank.id}|governor".encode()),
                                    np.random.default_rng([self.seed, b, 77]))
            self._senders[key] = SenderFI(
                bank.id, S, iss, att, self.cost, A["profile"], backend or A["backend"], mode, lam, layout, log,
                fixed_bundle=fixed, rng=np.random.default_rng(self.seed * 1000 + b), cache=cache,
                actions=self.planning_actions, governor=governor)
        return self._senders[key]

    def receiver(self, b: int, cfg: VerifyConfig | None = None, budget: float = 12.0, p_min: float = 0.3,
                 log=None) -> ReceiverFI:
        key = (b, id(cfg), id(log))
        if key not in self._receivers:
            bank = self.banks[b]
            ver = Verifier(bank.id, {S.version: S}, self.trust, cfg or VerifyConfig(), NonceStore(), log,
                           self.bop_sizes)
            pol = lambda p, _b=b: self.policy_for(_b, p.amount_minor)  # noqa: E731
            self._receivers[key] = ReceiverFI(bank.id, ver, pol, self.planning_actions, budget, p_min)
        return self._receivers[key]

    # ------------------------------------------------------------------ payments
    def payment(self, pay: dict, j: int) -> paymod.Payment:
        return paymod.Payment(
            sender_fi=self.banks[int(pay["sender"][j])].id, receiver_fi=self.banks[int(pay["receiver"][j])].id,
            payer_ref=f"P{int(pay['payer'][j]):07d}", beneficiary_ref=f"B{int(pay['payee_slot'][j]):07d}",
            amount_minor=int(pay["amount_minor"][j]), currency="USD", timestamp_ms=int(pay["ts"][j]) * 1000,
            tx_id=f"TX{self.seed:03d}{int(pay['gidx'][j]) if 'gidx' in pay else j:09d}", rail="RTP", purpose=str(int(pay["purpose"][j])))

    def evidence(self, pay: dict, j: int, pid: str, rng) -> list[dict]:
        b = self.banks[int(pay["sender"][j])]
        return wl.evidence_for(self.pop, int(pay["payer"][j]), int(pay["ts"][j]), int(pay["scenario"][j]), pid,
                               self.arche["archetypes"][b.archetype], rng)

    def can_complete(self, payer: int, sender_b: int, action: Action, scenario: int) -> bool:
        """Ground-truth capability: could this payer (or the attacker holding the session) complete it?"""
        A = self.arche["archetypes"][self.banks[sender_b].archetype]
        if action.name == "passkey_reauth":
            return bool(self.pop.has_passkey[payer]) and scenario != 1  # attacker lacks the passkey
        if action.name == "totp_reauth":
            return bool(self.pop.has_totp[payer]) and scenario != 1
        if action.name == "sms_otp_reauth":
            return True  # SIM-swap-capable attacker can pass SMS too
        if action.name == "selfie_kyc_refresh":
            return bool(self.pop.can_selfie[payer]) and scenario != 1
        if action.name == "fraud_review":
            return A["profile"]["supports_policy_attestation"] and scenario != 1
        return False

    def apply_action(self, evidence: list[dict], action: Action, pid: str, now: int, sender_b: int) -> list[dict]:
        A = self.arche["archetypes"][self.banks[sender_b].archetype]
        ev = list(evidence)
        method = {"passkey_reauth": "fido2_passkey", "totp_reauth": "totp", "sms_otp_reauth": "sms_otp"}.get(action.name)
        if method:
            ev.append({"type": "auth", "method": method, "ts": now, "tx_bound": bool(A["profile"]["supports_dynamic_binding"]),
                       "pid": pid})
        elif action.name == "selfie_kyc_refresh":
            ev.append({"type": "kyc", "level": "enhanced", "ts": now})
        elif action.name == "fraud_review":
            ev.append({"type": "policy_exec", "engine": "manual-review", "result": "review_pass", "ts": now, "pid": pid})
        return ev
