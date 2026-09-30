from vista.crypto import hashchain, orproof, selective_disclosure as sd
from vista.crypto.signed import SigningKey, verify

B = {"pid": "a" * 64, "aud": "FI2", "nonce": "n" * 32}


def test_ed25519():
    k = SigningKey()
    s = k.sign(b"m")
    assert verify(k.public_bytes, s, b"m") and not verify(k.public_bytes, s, b"m2")
    assert SigningKey.from_seed(b"\x01" * 32).key_id == SigningKey.from_seed(b"\x01" * 32).key_id


def test_selective_disclosure():
    k = SigningKey()
    cred, disc = sd.issue(k, "auth", ["auth>=A1", "auth>=A2"], 3, B)
    assert len(cred["body"]["digests"]) == 3  # decoy padding hides x
    pres = sd.present(cred, disc, ["auth>=A2"])
    assert sd.verify(pres, k.public_bytes, B) == ["auth>=A2"]
    pres["disclosed"][0][1] = "auth>=A3"
    try:
        sd.verify(pres, k.public_bytes, B)
        assert False
    except ValueError:
        pass


def test_hashchain_threshold():
    k = SigningKey()
    com, seed = hashchain.commit(k, "account_age", 4, B)
    w = hashchain.prove(seed, 4, 2, "account_age", B)
    assert hashchain.verify(com, k.public_bytes, 2, w, B)
    assert not hashchain.verify(com, k.public_bytes, 3, w, B)  # cannot stretch a witness upward
    assert not hashchain.verify(com, k.public_bytes, 2, w, dict(B, pid="b" * 64))
    assert hashchain.audit_open(com, seed, 4, B) and not hashchain.audit_open(com, seed, 5, B)


def test_or_proof_sound_and_bound():
    keys = [orproof.keygen() for _ in range(3)]
    pks = [p for _, p in keys]
    for j in range(3):
        pr = orproof.prove(pks, j, keys[j][0], b"ctx")
        assert orproof.verify(pks, pr, b"ctx")
        assert not orproof.verify(pks, pr, b"other-ctx")
    cs, zs = orproof.prove(pks, 1, keys[1][0], b"ctx")
    zs[0] += 1
    assert not orproof.verify(pks, (cs, zs), b"ctx")
    # proving with a wrong secret fails verification
    assert not orproof.verify(pks, orproof.prove(pks, 0, keys[1][0], b"ctx"), b"ctx")
