"""Cryptographic backends (Sec 4.3). All primitives are standard; VISTA claims no primitive novelty.

signed.py              Ed25519 signed predicates (mandatory baseline)
selective_disclosure.py salted-digest selective disclosure credentials (SD-JWT style), decoy padding
hashchain.py           hash-chain threshold proofs (HashWires-style): prove x >= theta, hide x
orproof.py             CDS witness-indistinguishable OR-proofs of Schnorr keys (branch-oblivious proof, BOP)
"""
