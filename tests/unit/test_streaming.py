import csv
import io
import json
import os

import numpy as np

from vista.experiments.stream_agg import Hist, Mom, StreamAgg, wilson
from vista.progress import Progress, fmt_dur
from vista.simulator import source as src
from vista.simulator.world import World


def test_hist_percentiles_close_to_exact(rng):
    x = rng.lognormal(-1.0, 1.2, 200_000)
    h = Hist(1e-4, 1e4, 2400, True)
    for part in np.array_split(x, 7):  # chunked, like the pipeline
        h.add(part)
    for q in (50, 95, 99):
        assert abs(h.pct(q) / np.percentile(x, q) - 1) < 0.01
    assert h.max == x.max()
    z = Hist(1e-4, 1e4, 2400, True)
    z.add(np.array([0.0] * 60 + [1.0] * 40))
    assert z.pct(50) == 0.0 and z.pct(99) > 0.9


def test_moments_and_wilson(rng):
    x = rng.normal(3, 2, 50_000)
    m = Mom()
    for part in np.array_split(x, 5):
        m.add(part)
    lo, hi = m.ci()
    assert abs(m.mean - x.mean()) < 1e-9 and lo < 3 < hi
    lo, hi = wilson(50, 100)
    assert 0.4 < lo < 0.5 < hi < 0.6


def test_progress_message_and_eta():
    t = [1000.0]
    p = Progress(clock=lambda: t[0], stream=io.StringIO())
    p.start([("a", 10.0), ("b", 100.0), ("c", 50.0)])
    p.begin("b")
    t[0] += 60
    p.update(250, 1000, "payments")
    m = p.message()
    assert "25.0%" in m and "remaining ~3m00s" in m  # 60s for 25% -> 180s left
    assert "total remaining ~3m50s" in m  # + prior of step c (50s)
    p.begin("c")
    t[0] += 10
    assert "remaining ~0m40s" in p.message()  # no counter: prior 50s - 10s
    p.stop()
    assert fmt_dur(3725) == "1h02m05s" and fmt_dur(None) == "unknown"


def _fake_paysim(path, n=300):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
                    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])
        types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        for i in range(n):
            t = types[i % 5]
            w.writerow([1 + i // 10, t, f"{10 + i * 3.17:.2f}", f"C{1000 + i}", 0, 0,
                        ("M" if t == "PAYMENT" else "C") + str(5000 + i), 0, 0,
                        1 if t in ("TRANSFER", "CASH_OUT") and i % 10 == 1 else 0, 0])


def test_paysim_cache_roundtrip_and_source(tmp_path):
    csv_p, cache = str(tmp_path / "p.csv"), str(tmp_path / "cache")
    _fake_paysim(csv_p)
    meta = src.build_cache(csv_p, cache, expected=None)
    assert meta["rows_total"] == 300 and meta["rows_usable"] == 240 and meta["rows_excluded_cash_in"] == 60
    assert src.ensure_cache(csv_p, cache, allow_download=False)["source_sha256"] == meta["source_sha256"]  # reused
    w = World(n_banks=10, n_payers=2000, seed=3, prior_sample=300)
    s = src.PaySimSource(cache)
    assert s.n == 240
    a = s.take(np.arange(0, 240, 7), w)
    b = s.take(np.arange(0, 240, 7), w)
    assert all(np.array_equal(a[k], b[k]) for k in a)  # deterministic
    assert (a["sender"] != a["receiver"]).all() and a["amount_minor"].min() >= 1
    assert a["scenario"].sum() == int(np.asarray(s.arr["fraud"])[np.arange(0, 240, 7)].sum())
    assert src.PaySimSource(cache, limit=50).n == 50


def test_paysim_statistics_mismatch_is_refused(tmp_path):
    csv_p = str(tmp_path / "p.csv")
    _fake_paysim(csv_p)
    try:
        src.build_cache(csv_p, str(tmp_path / "c"), expected=src.EXPECTED)
        assert False, "a file that is not PaySim must be refused"
    except src.PaySimMismatch:
        pass


def test_stream_agg_matches_exact_recomputation():
    """Run a tiny pipeline, then recompute the headline tables exactly from the detailed CSV."""
    from vista.experiments import exp_e2e
    import tempfile
    out = tempfile.mkdtemp()
    cfg = {"world": {"source": "synthetic", "n_banks": 12, "n_payers": 4000, "seed": 5, "prior_sample": 800,
                     "n_payments": 1500, "days": 7}, "workers": 2, "chunk": 250, "e3_payments": 250,
           "scaling_payments": 250, "scaling_workers": [1]}
    exp_e2e.run(cfg, out)
    rows = list(csv.DictReader(open(os.path.join(out, "e2e_transactions.csv"))))
    assert len(rows) == 1500 and [int(r["tx_idx"]) for r in rows] == list(range(1500))  # ordered, complete
    ag = {r["approach"]: r for r in csv.DictReader(open(os.path.join(out, "h1_decision_fidelity.csv")))}
    sat = np.array([r["policy_satisfiable"] == "True" for r in rows])
    vista = np.array([r["decision_initial"] == "ACCEPT" for r in rows])
    assert abs(float(ag["VISTA_initial"]["agreement_with_receiver_policy"]) - (vista == sat).mean()) < 1e-9
    b1 = np.array([r["b1_accept"] == "True" for r in rows])
    assert abs(float(ag["B1_sender_risk_label"]["false_accept_rate"]) - (b1 & ~sat).sum() / (~sat).sum()) < 1e-6
    h2 = {r["strategy"]: r for r in csv.DictReader(open(os.path.join(out, "h2_leakage_by_strategy.csv")))}
    det = np.array([float(r["leak_DET_bits"]) for r in rows if r["policy_satisfiable"] == "True"]) \
        if "leak_DET_bits" in rows[0] else None
    assert abs(float(h2["VISTA_MDPS_DET"]["mean_leakage_bits"]) - det.mean()) < 1e-3  # CSV rounds to 6 sig. digits
    sc = {r["scenario"]: r for r in csv.DictReader(open(os.path.join(out, "e8_decisions_by_scenario.csv")))}
    assert sum(int(r["payments"]) for r in sc.values()) == 1500
    lat = {r["component"]: r for r in csv.DictReader(open(os.path.join(out, "e7_latency.csv")))}
    e2e = np.array([float(r["e2e_ms"]) for r in rows])
    assert abs(float(lat["e2e_ms"]["p99"]) / np.percentile(e2e, 99) - 1) < 0.02
    assert abs(float(lat["e2e_ms"]["mean"]) / e2e.mean() - 1) < 1e-3
