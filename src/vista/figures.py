"""Regenerate paper figures from the result CSVs (Sec 11.2: every figure is regenerable)."""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# validated categorical order (dataviz reference palette, light mode) + chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p)
    plt.close(fig)
    return p


def _hbar(ax, labels, vals, color=SERIES[0], fmt="{:.2f}"):
    y = np.arange(len(labels))
    ax.barh(y, vals, color=color, height=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK2, fontsize=8)


def make_all(res: str, fig_dir: str) -> list[str]:
    os.makedirs(fig_dir, exist_ok=True)
    out = []

    rows = _read(os.path.join(res, "h2_leakage_by_strategy.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        _hbar(ax, [r["strategy"] for r in rows], [_f(r["mean_leakage_bits"]) for r in rows])
        ax.set_xlabel("mean surprisal leakage per accepted payment (bits)")
        ax.set_title("H2: disclosure leakage by strategy (realistic workload)")
        out.append(_save(fig, fig_dir, "h2_leakage_by_strategy.png"))

    rows = _read(os.path.join(res, "h1_decision_fidelity.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(6, 2.0))
        _hbar(ax, [r["approach"] for r in rows], [100 * _f(r["agreement_with_receiver_policy"]) for r in rows],
              fmt="{:.1f}%")
        ax.set_xlim(0, 110)
        ax.set_xlabel("agreement with the receiver's own policy on true evidence (%)")
        ax.set_title("H1: decision fidelity")
        out.append(_save(fig, fig_dir, "h1_decision_fidelity.png"))

    rows = _read(os.path.join(res, "e12_leakage_cost_pareto.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        rows = [r for r in rows if r["prior"] == "uniform"]
        for k, mode in enumerate(("DET", "EXP", "LP", "BOP")):
            rr = [r for r in rows if r["mode"] == mode]
            xs = [100 * _f(r["cost_overhead"]) for r in rr]
            ys = [_f(r["Lambda_bits"]) for r in rr]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", color=SERIES[k],
                    lw=2 if len(rr) > 1 else 0, label=mode)
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xlabel("expected disclosure cost overhead vs DET (%)")
        ax.set_ylabel("selection leakage Λ (bits)")
        ax.set_title("E12: selection leakage vs cost (toy, uniform prior)")
        ax.legend()
        out.append(_save(fig, fig_dir, "e12_pareto.png"))

    rows = _read(os.path.join(res, "e14_probing_attack.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        strategies = ("binary", "random", "scan")
        rhos = [r["rho"] for r in rows if r["strategy"] == "binary"]
        x = np.arange(len(rhos))
        w = 0.26
        for k, st in enumerate(strategies):
            vals = [_f(r["mean_bits_learned"]) for r in rows if r["strategy"] == st]
            ax.bar(x + (k - 1) * w, vals, w - 0.03, color=SERIES[k], label=st)
        ax.plot(x, [_f(r["bound_bits"]) for r in rows if r["strategy"] == "binary"], color=INK2, lw=1.2,
                marker="_", ms=14, ls="none", label="bound log2(|D|/ρ)")
        ax.set_xticks(x, rhos)
        ax.set_xlabel("ledger resolution ρ")
        ax.set_ylabel("bits learned about a 128-value attribute")
        ax.set_title("E14: threshold probing vs interval ledger")
        ax.legend(ncol=2)
        out.append(_save(fig, fig_dir, "e14_probing.png"))

    rows = _read(os.path.join(res, "e16_cumulative_leakage.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for k, st in enumerate(("disclose_all", "history_blind", "sticky")):
            rr = [r for r in rows if r["strategy"] == st]
            ax.plot([_f(r["payment_index"]) for r in rr], [_f(r["mean_cumulative_bits"]) for r in rr],
                    color=SERIES[k], label=st)
        ax.set_xlabel("payments to the same receiver")
        ax.set_ylabel("cumulative surprisal (bits)")
        ax.set_title("E16: cumulative leakage over repeated payments")
        ax.legend()
        out.append(_save(fig, fig_dir, "e16_cumulative.png"))

    rows = _read(os.path.join(res, "e1_mdps_summary.csv"))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for k, solver in enumerate(("ilp", "frontier_cached", "readonce_dp", "greedy")):
            for fam_i, fam in enumerate(("cnf", "dnf")):
                rr = [r for r in rows if r["solver"] == solver and r["family"] == fam]
                if not rr:
                    continue
                axes[fam_i].plot([_f(r["n_claims"]) for r in rr], [_f(r["p50_ms"]) for r in rr], marker="o",
                                 color=SERIES[k], label=solver)
        for fam_i, fam in enumerate(("cnf", "dnf")):
            axes[fam_i].set_yscale("log")
            axes[fam_i].set_xlabel("claims in universe")
            axes[fam_i].set_ylabel("median solve time (ms)")
            axes[fam_i].set_title(f"E1: MDPS latency, {fam.upper()} policies")
            axes[fam_i].legend()
        out.append(_save(fig, fig_dir, "e1_mdps_latency.png"))

    rows = _read(os.path.join(res, "e18_fixed_charge_greedy.csv"))
    rr = [r for r in rows if r.get("kind") == "adversarial_family"]
    if rr:
        fig, ax = plt.subplots(figsize=(5, 3))
        ks = [_f(r["k"]) for r in rr]
        ax.plot(ks, [_f(r["myopic_ratio"]) for r in rr], marker="o", color=SERIES[1], label="myopic greedy")
        ax.plot(ks, [_f(r["density_ratio"]) for r in rr], marker="o", color=SERIES[0], label="density greedy (DG)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (clauses)")
        ax.set_ylabel("cost / optimum")
        ax.set_title("E18: fixed-charge proofs, adversarial family")
        ax.legend()
        out.append(_save(fig, fig_dir, "e18_greedy.png"))

    rows = _read(os.path.join(res, "e19_stochastic_man.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        names = ("optimal_dp", "ratio_rule", "fixed_order", "blanket")
        vals = [np.nanmean([_f(r["friction_per_success"]) for r in rows if r["strategy"] == n
                            and np.isfinite(_f(r["friction_per_success"]))]) for n in names]
        _hbar(ax, list(names), vals)
        ax.set_xlabel("expected friction per successful step-up")
        ax.set_title("E19: stochastic MAN strategies (random read-once catalogs)")
        out.append(_save(fig, fig_dir, "e19_man.png"))

    rows = _read(os.path.join(res, "e13_bop_cost.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for k, b in enumerate(("bop", "signed_cheapest_branch")):
            rr = [r for r in rows if r["backend"] == b]
            ax.plot([_f(r["branches"]) for r in rr], [_f(r["gen_p50_ms"]) + _f(r["verify_p50_ms"]) for r in rr],
                    marker="o", color=SERIES[k], label=b)
        ax.set_xlabel("frontier branches |M(Φ)|")
        ax.set_ylabel("generate + verify, median (ms)")
        ax.set_title("E13: branch-oblivious proof cost")
        ax.legend()
        out.append(_save(fig, fig_dir, "e13_bop.png"))

    rows = _read(os.path.join(res, "e7_throughput_scaling.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([_f(r["workers"]) for r in rows], [_f(r["throughput_tps"]) for r in rows], marker="o", color=SERIES[0])
        ax.set_xlabel("worker processes")
        ax.set_ylabel("payments / second")
        ax.set_title("E7: end-to-end throughput scaling")
        out.append(_save(fig, fig_dir, "e7_throughput.png"))

    rows = _read(os.path.join(res, "e10_failure_injection.csv"))
    rows = [r for r in rows if r["minute"] != "summary"]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.plot([_f(r["minute"]) for r in rows], [100 * _f(r["availability_when_satisfiable"]) for r in rows],
                color=SERIES[0])
        for ph, lab in (("registry_outage", "registry outage"), ("key_revoked", "key revoked"),
                        ("schema_skew", "schema skew")):
            ms = [_f(r["minute"]) for r in rows if r["phase"] == ph]
            if ms:
                ax.axvspan(min(ms), max(ms) + 1, color=GRID, alpha=0.8, lw=0)
                ax.text(min(ms) + 0.3, 3, lab, color=INK2, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xlabel("minute")
        ax.set_ylabel("accepted | satisfiable (%)")
        ax.set_title("E10: fail-closed behaviour under injected faults")
        out.append(_save(fig, fig_dir, "e10_failures.png"))
    return out
