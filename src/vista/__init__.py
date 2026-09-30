"""VISTA research artifact.

Every module maps to a section of the VISTA IEEE research blueprint (v1.1):

    payment.py          Sec 1.1   canonical payment encoding and PID
    claims.py           Sec 3.2   canonical claim schema, assurance chains (N1)
    policy.py           Sec 3.3   monotone receiver policy language, FRONTIER (N1)
    cost.py / prior.py  Sec 3.4   cost models, receiver prior (N4)
    sen.py              Sec 5.1   security evidence normalization
    mdps_exact.py       Sec 5.2   exact MDPS (frontier, read-once DP, ILP, brute force)
    mdps_greedy.py      Sec 5.2   greedy MDPS
    group_cost.py       Sec 24    fixed-charge MDPS (N5)
    selection.py        Sec 21    selection-channel leakage, LP kernel, BOP (N2)
    ledger.py           Sec 22    cumulative disclosure control, OCGD (N3)
    leakage.py          Sec 23    correlation-aware leakage (N4)
    man.py              Sec 5.3/25 MAN and stochastic MAN-S (N6)
    policy_extraction.py Sec 26   policy-extraction bounds (N7)
    audit.py            Sec 27    commit-and-audit deterrence, Merkle log (N8)
    crypto/             Sec 4.3   Ed25519, selective disclosure, hash-chain thresholds, CDS OR-proofs
    vsac.py / trust.py  Sec 4     capsule build/verify, trust registry
    sender.py / receiver.py Sec 7 end-to-end protocol engines
    simulator/          Sec 10    banks, realistic transaction workload, attacks
    experiments/        Sec 9/29  E1-E22 + baselines + ablations
"""

__version__ = "0.1.0"
