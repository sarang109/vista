"""Canonical payment encoding and payment identifier PID = H(Encode(P))  (Sec 1.1, 4.2)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace

from .util import canonical_json, sha256_hex

ENCODING_VERSION = "vista-payment-v1"
_CCY = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class Payment:
    sender_fi: str
    receiver_fi: str
    payer_ref: str
    beneficiary_ref: str
    amount_minor: int  # integer minor units (cents); floats are never used
    currency: str
    timestamp_ms: int
    tx_id: str
    rail: str = "RTP"
    purpose: str = ""

    def with_(self, **kw) -> "Payment":
        return replace(self, **kw)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def encode(p: Payment) -> bytes:
    if not isinstance(p.amount_minor, int) or isinstance(p.amount_minor, bool) or p.amount_minor < 0:
        raise ValueError("amount_minor must be a non-negative integer")
    ccy = p.currency.upper()
    if not _CCY.match(ccy):
        raise ValueError("currency must be an ISO-4217 alphabetic code")
    body = {k: (_norm(v) if isinstance(v, str) else v) for k, v in asdict(p).items()}
    body["currency"] = ccy
    body["amount_minor"] = str(p.amount_minor)  # string avoids cross-language integer width issues
    body["timestamp_ms"] = str(p.timestamp_ms)
    body["_v"] = ENCODING_VERSION
    return canonical_json(body)


def pid(p: Payment) -> str:
    return sha256_hex(encode(p))


MATERIAL_FIELDS = ("sender_fi", "receiver_fi", "payer_ref", "beneficiary_ref", "amount_minor",
                   "currency", "timestamp_ms", "tx_id", "rail", "purpose")
