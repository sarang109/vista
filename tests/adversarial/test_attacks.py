"""Adversarial suite (E5): every attack is rejected by the full verifier, for every backend."""
import csv
import os

import pytest

from vista.experiments import exp_attacks
from vista.simulator.world import World


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("att")
    w = World(n_banks=8, n_payers=2000, seed=3, prior_sample=1000)
    p = exp_attacks.run({"e5_base_payments": 5, "seed": 2}, str(out), w)[0]
    return list(csv.DictReader(open(p)))


def test_full_verifier_rejects_everything(results):
    full = [r for r in results if r["ablation"] == "full"]
    assert full and all(int(r["accepted"]) == 0 for r in full), [r for r in full if int(r["accepted"])]


def test_ablations_expose_each_invariant(results):
    got = {(r["ablation"], r["attack"]) for r in results if r["backend"] == "signed" and float(r["acceptance_rate"]) > 0}
    for need in [("no_transaction_binding", "amount_substitution"), ("no_transaction_binding", "cross_tx_transplant"),
                 ("no_freshness", "expired_capsule"), ("no_nonce_no_replay_cache", "replay_same_payment"),
                 ("no_status_check", "revoked_issuer_key"), ("no_signature_check", "claim_injection")]:
        assert need in got, need
