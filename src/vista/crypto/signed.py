"""Ed25519 signatures (RFC 8032) via the `cryptography` package."""
from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ..util import sha256_hex


class SigningKey:
    def __init__(self, sk: Ed25519PrivateKey | None = None):
        self._sk = sk or Ed25519PrivateKey.generate()
        self.public_bytes = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.key_id = sha256_hex(self.public_bytes)[:16]

    @classmethod
    def from_seed(cls, seed: bytes) -> "SigningKey":
        """Deterministic key from a 32-byte seed (lets parallel workers rebuild the same world)."""
        return cls(Ed25519PrivateKey.from_private_bytes(seed[:32]))

    def sign(self, msg: bytes) -> bytes:
        return self._sk.sign(msg)


_pub_cache: dict = {}


def verify(public_bytes: bytes, sig: bytes, msg: bytes) -> bool:
    pk = _pub_cache.get(public_bytes)
    if pk is None:
        pk = Ed25519PublicKey.from_public_bytes(public_bytes)
        _pub_cache[public_bytes] = pk
    try:
        pk.verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False
