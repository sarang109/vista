"""Regression: E3 re-sends the same payments in several modes. Receivers persist inside a worker, so
without clearing the replay cache between tasks later modes were rejected as replays."""
import csv
import os
import tempfile

from vista.experiments import exp_e2e


def test_e3_modes_are_not_rejected_as_replays():
    out = tempfile.mkdtemp()
    cfg = {"world": {"source": "synthetic", "n_banks": 12, "n_payers": 4000, "seed": 5, "prior_sample": 800,
                     "n_payments": 1200, "days": 7}, "workers": 2, "chunk": 150, "e3_payments": 300,
           "scaling_payments": 150, "scaling_workers": [1]}
    exp_e2e.run(cfg, out)
    rows = {(r["mode"], r["backend"]): r for r in csv.DictReader(open(os.path.join(out, "e3_disclosure_modes.csv")))}
    full = [k for k in rows if k[0] in ("ALL", "DET", "RAND", "BOP")]
    rates = {k: float(rows[k]["accept_rate_initial"]) for k in full}
    assert min(rates.values()) > 0.5, rates  # a replay-poisoned mode would be near 0
    assert max(rates.values()) - min(rates.values()) < 1e-9, rates  # same payments, same decisions
    assert all(float(rows[k]["matches_receiver_policy"]) == 1.0 for k in full)
    assert float(rows[("FIXED", "signed")]["accept_rate_initial"]) <= min(rates.values())  # fixed bundle may under-disclose
