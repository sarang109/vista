"""Receiving-FI engine (Sec 7, modules [4]-[6]): VSAC verifier -> receiver policy -> MAN-S.

Policy sovereignty: the receiver never consumes a sender risk label; it evaluates its own policy
Phi_R on the verified claim set (or accepts a verified branch-oblivious proof of Phi_R itself).
"""
from __future__ import annotations

import time

from . import man as manmod
from . import payment as paymod
from .policy import Node, evaluate
from .vsac import Verifier


class ReceiverFI:
    def __init__(self, inst_id: str, verifier: Verifier, policy_for, actions: list[manmod.Action],
                 budget: float = 12.0, p_min: float = 0.5):
        self.id = inst_id
        self.verifier = verifier
        self.policy_for = policy_for  # payment -> policy node (amount-tiered)
        self.actions = actions
        self.budget, self.p_min = budget, p_min

    def decide(self, p: paymod.Payment, blob: bytes | None, now: int, failed: frozenset = frozenset()) -> dict:
        """failed: step-up actions that already failed for this payment (excluded from re-planning)."""
        t0 = time.perf_counter()
        vr = self.verifier.verify(p, blob, now)
        t1 = time.perf_counter()
        pol: Node = self.policy_for(p)
        out = {"verify_ms": (t1 - t0) * 1e3, "verify_reason": vr.reason}
        if not vr.ok:
            # fail closed: unverifiable evidence never satisfies a requirement
            out.update(decision="REJECT", policy_ms=0.0, man_ms=0.0, profile=None, plan=None)
            return out
        ok = vr.policy_satisfied_bop or evaluate(pol, vr.profile)
        t2 = time.perf_counter()
        out["policy_ms"] = (t2 - t1) * 1e3
        out["profile"] = vr.profile
        if ok:
            out.update(decision="ACCEPT", man_ms=0.0, plan=None)
            return out
        acts = [a if a.name not in failed else manmod.Action(a.name, a.k, 0.0, ()) for a in self.actions]
        plan = manmod.plan(pol, vr.profile, acts, self.budget, self.p_min)
        out["man_ms"] = (time.perf_counter() - t2) * 1e3
        out["decision"] = plan["decision"]
        out["plan"] = plan
        return out
