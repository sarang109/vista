"""CDS (Cramer-Damgard-Schoenmakers 1994) witness-indistinguishable OR-proofs of knowledge of one of
n discrete logarithms, made non-interactive with Fiat-Shamir, over the RFC 3526 2048-bit MODP group.

Branch-oblivious proof (BOP, Prop 2.6): per payment the sender's attestor derives a Schnorr key per
frontier branch u of M(Phi) and hands the prover the secret keys only for branches available at x.
The prover shows knowledge of *some* secret key; the receiver learns only Phi(x)=1, not which branch.
Pure Python big-int arithmetic: slow but dependency-free; cost grows linearly in |M| (as predicted).
"""
from __future__ import annotations

import hashlib
import secrets

try:  # GMP-backed modular exponentiation (~14x faster); pure-Python fallback keeps the artifact portable
    import gmpy2

    def _pw(b: int, e: int, m: int) -> int:
        return int(gmpy2.powmod(b, e, m))
    BIGINT_BACKEND = "gmpy2"
except ImportError:  # pragma: no cover
    def _pw(b: int, e: int, m: int) -> int:
        return pow(b, e, m)
    BIGINT_BACKEND = "python"

# RFC 3526 group 14 (2048-bit safe prime). q = (p-1)/2 prime; g = 4 generates the order-q subgroup.
P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
Q = (P - 1) // 2
G = 4
ELEM_BYTES = 256


SK_BITS = 256  # short-exponent Schnorr keys (short-exponent DL assumption in the 2048-bit group)
CH_BITS = 256  # challenge space 2^256; per-branch challenges split additively mod 2^256
Z_BITS = SK_BITS + CH_BITS + 128  # responses over the integers, statistically hiding (distance ~2^-128)
CH_MOD = 1 << CH_BITS


def keygen():
    sk = secrets.randbits(SK_BITS) | 1
    return sk, _pw(G, sk, P)


def _challenge(context: bytes, pks, ts) -> int:
    h = hashlib.sha512()
    h.update(b"vista-cds-v1|")
    h.update(context)
    for v in list(pks) + list(ts):
        h.update(v.to_bytes(ELEM_BYTES, "big"))
    return int.from_bytes(h.digest(), "big") % CH_MOD


def prove(pks: list[int], j: int, sk_j: int, context: bytes) -> tuple[list[int], list[int]]:
    n = len(pks)
    cs, zs, ts = [0] * n, [0] * n, [0] * n
    r = secrets.randbits(Z_BITS)
    for i in range(n):
        if i == j:
            ts[i] = _pw(G, r, P)
        else:  # simulate branch i
            cs[i] = secrets.randbits(CH_BITS)
            zs[i] = secrets.randbits(Z_BITS)
            ts[i] = (_pw(G, zs[i], P) * _pw(pks[i], -cs[i], P)) % P
    c = _challenge(context, pks, ts)
    cs[j] = (c - sum(cs[i] for i in range(n) if i != j)) % CH_MOD
    zs[j] = r + cs[j] * sk_j
    return cs, zs


def verify(pks: list[int], proof: tuple[list[int], list[int]], context: bytes) -> bool:
    """pks are certified by the attestor signature, so no subgroup-membership exponentiation is needed."""
    cs, zs = proof
    if len(cs) != len(pks) or len(zs) != len(pks):
        return False
    if any(not (0 <= c < CH_MOD) for c in cs) or any(not (0 <= z < (1 << (Z_BITS + 2))) for z in zs):
        return False
    if any(not (1 < pk < P - 1) for pk in pks):
        return False
    ts = [(_pw(G, z, P) * _pw(pk, -c, P)) % P for pk, c, z in zip(pks, cs, zs)]
    return _challenge(context, pks, ts) == sum(cs) % CH_MOD


def to_hex(v: int) -> str:
    return v.to_bytes(ELEM_BYTES, "big").hex()


def from_hex(s: str) -> int:
    return int(s, 16)
