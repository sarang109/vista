"""Realistic real-time-payment workload generator (Sec 10).

Design goals: transaction-like data at scale (10^5-10^6 payments) with the statistical texture of
retail instant-payment traffic, while never pretending to estimate real fraud prevalence:
  * banks with Zipf market shares and three capability archetypes
  * payer population with heavy-tailed activity, account ages, KYC levels, device binding,
    credential-reset recency and authentication-method preferences
  * payments with diurnal/weekly arrival intensity, purpose-dependent lognormal amounts,
    round-number effects, repeat payer->payee relationships and interbank routing
  * latent scenario tags: normal / ato_like (account-takeover-like evidence) / scam_like
    (deceived customer with strong authentication) - used only to characterise decisions
Optional backbone: PaySim-format CSV (step,type,amount,nameOrig,...,nameDest,...,isFraud).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

DAY = 86_400
T0 = 1_788_220_800  # 2026-09-01T00:00:00Z
METHODS = ("fido2_passkey", "totp", "push_number_match", "push", "sms_otp", "password")
KYC = ("none", "basic", "standard", "enhanced")
PURPOSES = {  # share, median USD, lognormal sigma
    "p2p": (0.55, 45.0, 1.0),
    "bill": (0.25, 140.0, 0.8),
    "merchant": (0.12, 300.0, 1.1),
    "b2b": (0.08, 2800.0, 1.2),
}
HOUR_W = np.array([0.6, 0.4, 0.3, 0.3, 0.3, 0.5, 1.0, 2.0, 3.2, 4.0, 4.4, 4.8,
                   5.2, 5.0, 4.6, 4.4, 4.6, 5.2, 5.6, 5.0, 4.0, 3.0, 2.0, 1.2])


def load_archetypes(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass
class Bank:
    idx: int
    id: str
    archetype: str
    share: float
    stance: str


def generate_banks(n: int, arche_cfg: dict, rng) -> list[Bank]:
    names = list(arche_cfg["mix"].keys())
    probs = np.array([arche_cfg["mix"][k] for k in names])
    st_names = list(arche_cfg["stance_mix"].keys())
    st_p = np.array([arche_cfg["stance_mix"][k] for k in st_names])
    shares = 1.0 / np.arange(1, n + 1) ** 1.1
    shares /= shares.sum()
    banks = []
    for b in range(n):
        banks.append(Bank(b, f"FI{b:03d}", str(rng.choice(names, p=probs)), float(shares[b]),
                          str(rng.choice(st_names, p=st_p))))
    return banks


class Population:
    """Column-oriented payer table (numpy arrays) so 10^5-10^6 payers stay cheap."""

    def __init__(self, n: int, banks: list[Bank], arche_cfg: dict, rng):
        self.n = n
        shares = np.array([b.share for b in banks])
        self.bank = rng.choice(len(banks), size=n, p=shares)
        new = rng.random(n) < 0.12
        age = np.where(new, rng.uniform(1, 90, n), np.minimum(rng.lognormal(7.0, 0.8, n), 7300))
        self.acct_open_ts = (T0 - age * DAY).astype(np.int64)
        kyc_p = np.where(age > 365, 0.20, 0.08)
        u = rng.random(n)
        self.kyc = np.where(u < 0.12, 1, np.where(u < 1 - kyc_p, 2, 3))
        self.primary = np.zeros(n, dtype=np.int8)
        self.secondary = np.zeros(n, dtype=np.int8)
        self.device_bound = np.zeros(n, dtype=bool)
        arch = [banks[b].archetype for b in range(len(banks))]
        for b in range(len(banks)):
            m = self.bank == b
            k = int(m.sum())
            if not k:
                continue
            A = arche_cfg["archetypes"][arch[b]]
            mix = np.array([A["auth_mix"].get(meth, 0.0) for meth in METHODS])
            mix /= mix.sum()
            self.primary[m] = rng.choice(len(METHODS), size=k, p=mix)
            self.secondary[m] = rng.choice(len(METHODS), size=k, p=mix)
            self.device_bound[m] = rng.random(k) < A["device_bound_rate"]
        dev_new = rng.random(n) < 0.08
        dev_age = np.where(dev_new, rng.uniform(0, 30, n), rng.lognormal(5.0, 1.0, n))
        dev_age = np.minimum(dev_age, age)
        self.device_first_seen_ts = (T0 - dev_age * DAY).astype(np.int64)
        recent = rng.random(n) < 0.03
        reset_age = np.where(recent, rng.uniform(0, 7, n), rng.exponential(400, n))
        reset_age = np.minimum(reset_age, age)
        self.last_reset_ts = (T0 - reset_age * DAY).astype(np.int64)
        self.has_passkey = (self.primary == 0) | (self.secondary == 0) | \
            ((np.array([arch[b] for b in self.bank]) == "modern") & (rng.random(n) < 0.25))
        self.has_totp = (self.primary == 1) | (self.secondary == 1) | (rng.random(n) < 0.15)
        self.can_selfie = rng.random(n) < 0.6
        self.activity = rng.lognormal(0.0, 1.0, n)
        self.n_payees = 1 + rng.poisson(4, n)


def generate_payments(pop: Population, banks: list[Bank], n_payments: int, days: int, rng,
                      ato_rate: float = 0.005, scam_rate: float = 0.003) -> dict:
    act = pop.activity / pop.activity.sum()
    payer = rng.choice(pop.n, size=n_payments, p=act)
    # arrivals: day with weekend dip, hour from diurnal profile, uniform within hour
    dw = np.array([1.0 if ((d + 1) % 7) not in (5, 6) else 0.7 for d in range(days)])  # 2026-09-01 is Tue
    day = rng.choice(days, size=n_payments, p=dw / dw.sum())
    hour = rng.choice(24, size=n_payments, p=HOUR_W / HOUR_W.sum())
    ts = T0 + day * DAY + hour * 3600 + rng.integers(0, 3600, n_payments)
    order = np.argsort(ts, kind="stable")
    payer, ts = payer[order], ts[order]
    pnames = list(PURPOSES)
    pp = np.array([PURPOSES[k][0] for k in pnames])
    purpose = rng.choice(len(pnames), size=n_payments, p=pp)
    med = np.array([PURPOSES[k][1] for k in pnames])[purpose]
    sig = np.array([PURPOSES[k][2] for k in pnames])[purpose]
    usd = rng.lognormal(np.log(med), sig)
    roundy = (rng.random(n_payments) < 0.35) & (purpose == 0)
    usd = np.where(roundy, np.maximum(5, np.round(usd / 5) * 5), usd)
    u = rng.random(n_payments)
    scenario = np.where(u < ato_rate, 1, np.where(u < ato_rate + scam_rate, 2, 0))
    # fraud-like scenarios move larger sums to new payees
    usd = np.where(scenario > 0, rng.lognormal(np.log(1800), 0.9, n_payments), usd)
    usd = np.clip(usd, 1.0, 500_000.0)
    amount_minor = np.round(usd * 100).astype(np.int64)
    # payees: repeat relationships (Zipf over the payer's payee list), 10% new payee
    new_payee = (rng.random(n_payments) < 0.10) | (scenario > 0)
    k = pop.n_payees[payer]
    rank = np.minimum(rng.zipf(1.6, n_payments), k) - 1
    payee_slot = np.where(new_payee, 1000 + rng.integers(0, 10**6, n_payments), rank)
    # payee's bank is a stable function of (payer, payee) so repeat payments route identically
    cum = np.cumsum([b.share for b in banks])
    sender = pop.bank[payer]
    h = np.array([int(hashlib.blake2b(f"{a}|{b}".encode(), digest_size=8).hexdigest(), 16)
                  for a, b in zip(payer.tolist(), payee_slot.tolist())], dtype=np.uint64)
    recv = np.minimum(np.searchsorted(cum, (h >> np.uint64(11)).astype(np.float64) / 2.0**53), len(banks) - 1)
    clash = recv == sender  # on-us payments are not interbank; reroute deterministically
    recv = np.where(clash, (recv + 1 + (h % np.uint64(len(banks) - 1)).astype(np.int64)) % len(banks), recv)
    return {"payer": payer, "ts": ts.astype(np.int64), "amount_minor": amount_minor, "purpose": purpose,
            "scenario": scenario, "sender": sender, "receiver": recv, "payee_slot": payee_slot,
            "new_payee": new_payee}


def load_paysim(path: str, pop: Population, banks: list[Bank], limit: int, rng) -> dict:
    """PaySim CSV backbone: amounts, timing (step = hour), counterparties and isFraud labels.
    isFraud rows (PaySim models account takeover + cash-out) are tagged ato_like."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["type"] not in ("TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT"):
                continue
            rows.append(r)
            if len(rows) >= limit:
                break
    n = len(rows)
    hid = lambda s, m: int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) % m  # noqa: E731
    payer = np.array([hid(r["nameOrig"], pop.n) for r in rows])
    ts = np.array([T0 + int(r["step"]) * 3600 + rng.integers(0, 3600) for r in rows], dtype=np.int64)
    amount = np.array([max(1, int(round(float(r["amount"]) * 100))) for r in rows], dtype=np.int64)
    scen = np.array([1 if r.get("isFraud") == "1" else 0 for r in rows])
    sender = pop.bank[payer]
    recv = np.array([hid(r["nameDest"], len(banks)) for r in rows])
    recv = np.where(recv == sender, (recv + 1) % len(banks), recv)
    ptype = {"PAYMENT": 2, "TRANSFER": 0, "CASH_OUT": 0, "DEBIT": 1}
    return {"payer": payer, "ts": ts, "amount_minor": amount,
            "purpose": np.array([ptype[r["type"]] for r in rows]), "scenario": scen, "sender": sender,
            "receiver": recv, "payee_slot": np.array([hid(r["nameDest"], 10**6) for r in rows]),
            "new_payee": scen > 0}


def evidence_for(pop: Population, i: int, ts: int, scenario: int, pid: str, arche: dict, rng) -> list[dict]:
    """Private sender-side evidence E_S for one payment (Sec 3.2 table 'private source evidence')."""
    ev = []
    kyc = KYC[int(pop.kyc[i])]
    ev.append({"type": "kyc", "level": kyc, "ts": int(pop.acct_open_ts[i])})
    ev.append({"type": "account", "opened_ts": int(pop.acct_open_ts[i])})
    last_reset = int(min(pop.last_reset_ts[i], ts))
    dev_first, bound = int(min(pop.device_first_seen_ts[i], ts)), bool(pop.device_bound[i])
    method = METHODS[int(pop.primary[i])] if rng.random() < 0.85 else METHODS[int(pop.secondary[i])]
    tx_bound = rng.random() < arche["dyn_link_rate"] and method in ("fido2_passkey", "totp", "push_number_match", "sms_otp")
    auth_age = rng.uniform(2, 90) if rng.random() < 0.93 else rng.uniform(90, 500)
    if scenario == 1:  # account-takeover-like: fresh reset, new/unbound device, weak auth, no linking
        last_reset = int(ts - rng.uniform(0.05, 2.0) * DAY)
        dev_first = int(ts - rng.uniform(0.0, 0.5) * DAY)
        bound = rng.random() < 0.3
        method = "sms_otp" if rng.random() < 0.7 else "password"
        tx_bound = False
    auth_ts = int(ts - auth_age)
    if scenario == 1 and rng.random() < 0.2:
        last_reset = auth_ts + 5  # reset *after* the session authenticated -> SEN invalidates auth claims
    ev.append({"type": "credential", "last_reset_ts": last_reset})
    ev.append({"type": "device", "first_seen_ts": dev_first, "bound": bound})
    ev.append({"type": "auth", "method": method, "ts": auth_ts, "tx_bound": bool(tx_bound), "pid": pid})
    if rng.random() < arche["policy_exec_rate"]:
        ev.append({"type": "policy_exec", "engine": "fraud-engine-v7", "result": "pass", "ts": int(ts - 1), "pid": pid})
    return ev


def dump_workload_csv(path: str, pay: dict, banks: list[Bank]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scen = ("normal", "ato_like", "scam_like")
    purp = list(PURPOSES)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tx_idx", "timestamp", "sender_fi", "receiver_fi", "payer", "amount_usd", "purpose",
                    "new_payee", "scenario"])
        for j in range(len(pay["ts"])):
            w.writerow([j, int(pay["ts"][j]), banks[pay["sender"][j]].id, banks[pay["receiver"][j]].id,
                        f"P{pay['payer'][j]:07d}", f"{pay['amount_minor'][j] / 100:.2f}",
                        purp[int(pay["purpose"][j]) % len(purp)], int(pay["new_payee"][j]),
                        scen[int(pay["scenario"][j])]])
