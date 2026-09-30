"""Property tests (Sec 8.3 item 16): any material change to a payment breaks its capsule."""
import numpy as np

from vista import payment as pm, vsac
from vista.claims import DEFAULT_SCHEMA as S
from vista.crypto.signed import SigningKey
from vista.trust import TrustRegistry

T0 = 1_788_220_800


def _setup():
    iss, att = SigningKey(), SigningKey()
    tr = TrustRegistry()
    tr.register("FI000", iss, "issuer", T0 - 10, T0 + 10**6)
    tr.register("FI000", att, "attestor", T0 - 10, T0 + 10**6)
    return iss, att, tr


def test_every_material_field_is_bound():
    rng = np.random.default_rng(3)
    iss, att, tr = _setup()
    x = tuple(a.d for a in S.attributes)
    for trial in range(40):
        p = pm.Payment("FI000", "FI001", f"P{trial}", "B1", int(rng.integers(1, 10**7)), "USD", T0 * 1000,
                       f"TX{trial}", "RTP", "0")
        for field in pm.MATERIAL_FIELDS:
            v = vsac.Verifier("FI001", {S.version: S}, tr)
            n = v.nonces.issue("FI000", 1, T0, rng)[0]
            blob = vsac.build(p=p, schema=S, issuer=iss, attestor=att, backend="signed",
                              claims=[(1, 3)], x=x, event_time={"auth_strength": T0}, nonce=n, now=T0,
                              lifetime_s=120, mode="DET")
            val = getattr(p, field)
            mutated = p.with_(**{field: (val + 1) if isinstance(val, int) else (val + "x" if field != "currency" else "EUR")})
            r = v.verify(mutated, blob, T0 + 1)
            assert not r.ok, (field, r.reason)
            assert v.verify(p, blob, T0 + 1).ok  # the untouched payment still verifies


def test_canonicalisation_is_order_independent():
    from vista.util import canonical_json
    a = {"b": 1, "a": [1, {"y": "2", "x": "1"}]}
    b = {"a": [1, {"x": "1", "y": "2"}], "b": 1}
    assert canonical_json(a) == canonical_json(b)
