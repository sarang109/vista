"""Constant-memory aggregation of the end-to-end stream (works for 10^3 or 10^8 payments).

Workers send small per-chunk numpy columns; this class folds them into counters, running moments and
fixed-size histograms. Percentiles come from histograms (log-spaced with ~0.8% relative bin width for
latencies, 0.01-bit bins for leakage); means, rates and confidence intervals are exact running moments
(normal-approximation CIs, which are tight at these sample sizes).
"""
from __future__ import annotations

import math
import os

import numpy as np

from ..claims import DEFAULT_SCHEMA as S
from ..util import write_csv

SCEN = ("normal", "ato_like", "scam_like")
TIERS = ("micro", "standard", "high", "very_high")
ARCH = ("modern", "mainstream", "legacy")
STANCE = ("liberal", "moderate", "conservative")
DEC = ("ACCEPT", "STEP_UP", "HOLD", "REJECT")
LAT = ("gen_ms", "net_ms", "verify_ms", "policy_ms", "man_ms", "e2e_ms")
LEAK = (("B2_raw_attributes", "leak_B2_raw_bits", None), ("B4_vsac_disclose_all", "leak_B4_all_bits", "claims_B4_all"),
        ("B3_fixed_bundle", "leak_B3_fixed_bits", "claims_B3_fixed"), ("VISTA_MDPS_DET", "leak_DET_bits", "claims_DET"),
        ("VISTA_BOP_floor", "leak_BOP_floor_bits", None))


class Hist:
    def __init__(self, lo: float, hi: float, bins: int, log: bool):
        self.log = log
        self.edges = np.geomspace(lo, hi, bins + 1) if log else np.linspace(lo, hi, bins + 1)
        self.counts = np.zeros(bins, dtype=np.int64)
        self.zeros = 0
        self.n = 0
        self.max = -math.inf

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        x = x[~np.isnan(x)]
        if not x.size:
            return
        self.n += x.size
        self.max = max(self.max, float(x.max()))
        if self.log:
            z = x <= 0
            self.zeros += int(z.sum())
            x = x[~z]
        i = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.counts) - 1)
        self.counts += np.bincount(i, minlength=len(self.counts))

    def pct(self, q: float) -> float:
        if self.n == 0:
            return math.nan
        rank = q / 100.0 * self.n
        if rank <= self.zeros:
            return 0.0
        i = int(np.searchsorted(np.cumsum(self.counts) + self.zeros, rank))
        i = min(i, len(self.counts) - 1)
        a, b = self.edges[i], self.edges[i + 1]
        return math.sqrt(a * b) if self.log else (a + b) / 2


class Mom:
    """Running n, sum, sum of squares (NaN-aware)."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x[~np.isnan(x)]
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float((x * x).sum())

    @property
    def mean(self):
        return self.s / self.n if self.n else math.nan

    def ci(self, z=1.96):
        if self.n < 2:
            return math.nan, math.nan
        var = max(self.ss / self.n - self.mean ** 2, 0.0) * self.n / (self.n - 1)
        h = z * math.sqrt(var / self.n)
        return self.mean - h, self.mean + h


def wilson(k: float, n: float, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


class StreamAgg:
    G = {"scenario": 3, "tier": 4, "heterogeneity": 9}

    def __init__(self):
        self.n = 0
        self.lat = {k: Hist(1e-4, 1e4, 2400, True) for k in LAT}
        self.lat_mom = {k: Mom() for k in LAT}
        self.h1 = {a: np.zeros(6) for a in ("B0_metadata_only", "B1_sender_risk_label", "VISTA_initial")}
        self.ideal_pos = 0
        self.leak = {n: (Mom(), Hist(0, 64, 6400, False)) for n, _, _ in LEAK}
        self.claims = {n: Mom() for n, _, c in LEAK if c}
        self.raw_bytes = Mom()
        self.grp = {g: {} for g in self.G}
        self.h3 = {k: Mom() for k in ("man_fr", "bl_fr", "diff_fr", "man_rounds", "bl_rounds", "man_sec", "bl_sec")}
        self.h3_n = self.h3_man_ok = self.h3_bl_ok = 0

    # ------------------------------------------------------------------ fold one chunk
    def add(self, c: dict):
        m = len(c["sat"])
        self.n += m
        for k in LAT:
            self.lat[k].add(c[k])
            self.lat_mom[k].add(c[k])
        sat = c["sat"].astype(bool)
        self.ideal_pos += int(sat.sum())
        for name, dec in (("B0_metadata_only", c["b0"].astype(bool)), ("B1_sender_risk_label", c["b1"].astype(bool)),
                          ("VISTA_initial", c["decision_initial"] == 0)):
            a = self.h1[name]
            a += [m, (dec == sat).sum(), (dec & ~sat).sum(), (~dec & sat).sum(), dec.sum(), 0]
        for name, col, ccol in LEAK:
            mom, h = self.leak[name]
            v = c[col][sat]
            mom.add(v)
            h.add(v)
            if ccol:
                self.claims[name].add(c[ccol][sat])
        self.raw_bytes.add(c["raw_evidence_bytes"])
        di, df = c["decision_initial"], c["decision_final"]
        su = (di == 1) | (di == 2)
        metrics = {"payments": np.ones(m), "sat": sat, "i0": di == 0, "i1": di == 1, "i2": di == 2, "i3": di == 3,
                   "f0": df == 0, "f2": df == 2, "f3": df == 3, "su": su, "su_ok": su & (df == 0),
                   "su_rounds": np.where(su, c["stepup_rounds"], 0), "su_fr": np.where(su, c["stepup_friction"], 0),
                   "leak": c["leakage_bits"], "bytes": c["capsule_bytes"], "vfail": ~c["verify_ok"].astype(bool)}
        het = c["archetype"].astype(int) * 3 + c["stance"].astype(int)
        for g, code in (("scenario", c["scenario"]), ("tier", c["tier"]), ("heterogeneity", het)):
            code = code.astype(int)
            d = self.grp[g]
            for k, w in metrics.items():
                d[k] = d.get(k, 0) + np.bincount(code, weights=np.asarray(w, dtype=np.float64), minlength=self.G[g])
        if su.any():
            self.h3_n += int(su.sum())
            man_ok = (df == 0)[su]
            bl_ok = (c["blanket_accept"] == 1)[su]
            self.h3_man_ok += int(man_ok.sum())
            self.h3_bl_ok += int(bl_ok.sum())
            man_fr, bl_fr = c["stepup_friction"][su], c["blanket_friction"][su]
            self.h3["man_fr"].add(man_fr)
            self.h3["bl_fr"].add(bl_fr)
            self.h3["diff_fr"].add(man_fr.astype(np.float64) - bl_fr)
            self.h3["man_rounds"].add(c["stepup_rounds"][su])
            self.h3["bl_rounds"].add(c["blanket_rounds"][su])
            self.h3["man_sec"].add(c["stepup_user_seconds"][su])
            self.h3["bl_sec"].add(c["blanket_user_seconds"][su])

    # ------------------------------------------------------------------ output tables
    def write(self, out: str, wall: float, workers: int) -> list[str]:
        files = []
        rows = []
        for k in LAT:
            h, mo = self.lat[k], self.lat_mom[k]
            rows.append({"component": k, "n": h.n, "mean": mo.mean, "p50": h.pct(50), "p95": h.pct(95),
                         "p99": h.pct(99), "max": h.max})
        rows.append({"component": "throughput", "n": self.n, "mean": self.n / wall, "p50": "", "p95": "", "p99": "",
                     "max": f"{workers} workers, wall {wall:.1f}s"})
        files.append(write_csv(os.path.join(out, "e7_latency.csv"), rows))

        neg = self.n - self.ideal_pos
        rows = []
        for name, a in self.h1.items():
            n, agree, fa, fr, acc, _ = a
            lo, hi = wilson(agree, n)
            rows.append({"approach": name, "payments": int(n), "agreement_with_receiver_policy": agree / n,
                         "ci95_lo": lo, "ci95_hi": hi, "false_accept_rate": fa / max(neg, 1),
                         "false_reject_rate": fr / max(self.ideal_pos, 1), "accept_rate": acc / n})
        files.append(write_csv(os.path.join(out, "h1_decision_fidelity.csv"), rows))

        rows = []
        for name, col, ccol in LEAK:
            mom, h = self.leak[name]
            lo, hi = mom.ci()
            rows.append({"strategy": name, "payments": mom.n, "mean_leakage_bits": mom.mean, "ci95_lo": lo,
                         "ci95_hi": hi, "p95_leakage_bits": h.pct(95),
                         "mean_claims": self.claims[name].mean if ccol else (S.n if "raw" in name else 0.0)})
        files.append(write_csv(os.path.join(out, "h2_leakage_by_strategy.csv"), rows))

        if self.h3_n:
            h = self.h3
            n = self.h3_n
            lo, hi = h["diff_fr"].ci()
            mk = lambda name, ok, fr, rd, sec: {  # noqa: E731
                "strategy": name, "stepup_payments": n, "resolution_rate": ok / n, "mean_friction": fr.mean,
                "mean_rounds": rd.mean, "mean_user_seconds": sec.mean,
                "friction_per_resolved": fr.s / max(ok, 1)}
            rows = [mk("MAN_S", self.h3_man_ok, h["man_fr"], h["man_rounds"], h["man_sec"]),
                    mk("B6_blanket", self.h3_bl_ok, h["bl_fr"], h["bl_rounds"], h["bl_sec"]),
                    {"strategy": "paired_diff_MAN_minus_blanket", "stepup_payments": n,
                     "mean_friction": h["diff_fr"].mean, "resolution_rate": (self.h3_man_ok - self.h3_bl_ok) / n,
                     "friction_ci95": f"[{lo:.3f}, {hi:.3f}]"}]
            files.append(write_csv(os.path.join(out, "h3_man_vs_blanket.csv"), rows))

        def labels(g):
            if g == "scenario":
                return list(SCEN)
            if g == "tier":
                return list(TIERS)
            return [f"{a}->{s}" for a in ARCH for s in STANCE]

        for g in self.G:
            d = self.grp[g]
            rows = []
            for code, lab in enumerate(labels(g)):
                n = d["payments"][code]
                if n == 0:
                    continue
                su = d["su"][code]
                nan = math.nan
                rows.append({g: lab, "payments": int(n), "satisfiable_rate": d["sat"][code] / n,
                             "initial_accept": d["i0"][code] / n, "initial_stepup": d["i1"][code] / n,
                             "initial_hold": d["i2"][code] / n, "initial_reject": d["i3"][code] / n,
                             "final_accept": d["f0"][code] / n, "final_hold": d["f2"][code] / n,
                             "final_reject": d["f3"][code] / n,
                             "stepup_resolved_rate": d["su_ok"][code] / su if su else nan,
                             "mean_stepup_rounds": d["su_rounds"][code] / su if su else nan,
                             "mean_stepup_friction": d["su_fr"][code] / su if su else nan,
                             "mean_leakage_bits": d["leak"][code] / n, "mean_capsule_bytes": d["bytes"][code] / n,
                             "verification_failures": int(d["vfail"][code])})
            files.append(write_csv(os.path.join(out, f"e8_decisions_by_{g}.csv"), rows))
        return files
