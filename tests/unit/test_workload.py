import csv

import numpy as np

from vista import payment as pm
from vista.sen import normalize
from vista.simulator import workload as wl
from vista.simulator.world import World


def test_workload_statistics():
    w = World(n_banks=15, n_payers=4000, seed=5, prior_sample=500)
    pay = wl.generate_payments(w.pop, w.banks, 20000, 7, np.random.default_rng(1))
    assert (pay["sender"] != pay["receiver"]).all()  # interbank only
    assert np.all(np.diff(pay["ts"]) >= 0)  # time ordered
    usd = pay["amount_minor"] / 100
    assert 20 < np.median(usd) < 200 and usd.max() > 1000  # heavy-tailed retail amounts
    hours = (pay["ts"] % 86400) // 3600
    assert np.bincount(hours, minlength=24)[12] > 3 * np.bincount(hours, minlength=24)[3]  # diurnal
    assert 0.002 < (pay["scenario"] == 1).mean() < 0.01


def test_paysim_loader(tmp_path):
    p = tmp_path / "paysim.csv"
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
                     "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])
        for i in range(200):
            wr.writerow([1 + i // 20, ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN"][i % 4], f"{100 + i * 13.37:.2f}",
                         f"C{i % 37}", 0, 0, f"M{i % 23}", 0, 0, 1 if i % 50 == 0 else 0, 0])
    w = World(n_banks=10, n_payers=1000, seed=5, prior_sample=300)
    pay = wl.load_paysim(str(p), w.pop, w.banks, 1000, np.random.default_rng(0))
    assert len(pay["ts"]) == 150  # CASH_IN rows are not interbank payments
    assert (pay["sender"] != pay["receiver"]).all()
    assert pay["scenario"].sum() == 4  # rows 0,50,100,150 are TRANSFER/CASH_OUT frauds
    j = 0
    pmt = w.payment(pay, j)
    x = normalize(w.evidence(pay, j, pm.pid(pmt), np.random.default_rng(0)), pm.pid(pmt), int(pay["ts"][j]) + 1).profile
    assert len(x) == 9
