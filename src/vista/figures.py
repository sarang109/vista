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
    out += _revision_figures(res, fig_dir)
    return out


def _revision_figures(res: str, fig_dir: str) -> list[str]:
    """E23-E28: leakage-cost frontier, repeated disclosure, real-signal priors, governor ablation, grace."""
    out = []
    rows = _read(os.path.join(res, "e24_frontier_curve.csv"))
    if rows:
        priors = [p for p in ("uniform", "skewed") if any(r["prior"] == p for r in rows)]
        fig, axes = plt.subplots(1, len(priors), figsize=(4.6 * len(priors), 3.4), sharey=True, squeeze=False)
        for ax, pr in zip(axes[0], priors):
            rr = [r for r in rows if r["prior"] == pr]
            for k, mode in enumerate(("PCGS", "RAND")):
                pts = sorted((100 * _f(r["cost_overhead"]), _f(r["Lambda_bits"])) for r in rr if r["mode"] == mode)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[k], label=mode, marker="o", ms=3)
            lp = [(100 * _f(r["cost_overhead"]), _f(r["Lambda_bits"])) for r in rr if r["mode"] == "LP"]
            if lp:
                ax.plot(*zip(*lp), ls="none", marker="s", ms=6, color=SERIES[2], label="LP ε-private")
            det = [r for r in rr if r["mode"] == "DET"][0]
            ax.plot([0.0], [_f(det["Lambda_bits"])], ls="none", marker="D", ms=7, color=SERIES[3], label="DET",
                    clip_on=False)
            fl = [r for r in rr if r["mode"] == "FLOOR"][0]
            ax.axhline(_f(fl["Lambda_bits"]), color=MUTED, lw=1, ls="--")
            xmax = max(100 * _f(r["cost_overhead"]) for r in rr if r["mode"] in ("PCGS", "LP", "FLOOR"))
            ax.text(xmax * 0.35, _f(fl["Lambda_bits"]) - 0.004, "floor: best kernel at any cost (> 0, Theorem 1)",
                    color=INK2, fontsize=7, va="top")
            ax.set_xlim(0, xmax * 1.05)
            ax.set_xlabel("expected cost overhead vs DET (%)")
            ax.set_title(f"E24: leakage-cost frontier ({pr} prior)")
        axes[0][0].set_ylabel("selection leakage Λ (bits)")
        axes[0][0].legend()
        out.append(_save(fig, fig_dir, "e24_frontier.png"))

    rows = _read(os.path.join(res, "e25_theory.csv"))
    if rows:
        fams = [f for f in ("toy_64_state", "policy_library") if any(r["family"] == f for r in rows)]
        fig, axes = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 3.3), squeeze=False)
        names = {"DET": "DET", "RAND_fresh": "RAND, fresh draws", "PCGS_fresh": "PCGS, fresh draws",
                 "PCGS_SBS": "PCGS + sticky (SBS)"}
        for ax, fam in zip(axes[0], fams):
            for k, st in enumerate(("RAND_fresh", "PCGS_fresh", "PCGS_SBS", "DET")):
                rr = [r for r in rows if r["family"] == fam and r["strategy"] == st]
                ax.plot([_f(r["n"]) for r in rr], [_f(r["mean_Lambda_bits"]) for r in rr], color=SERIES[k],
                        marker="o", ms=3, label=names[st], ls="--" if st == "DET" else "-")
            ax.set_xlabel("payments to the same receiver, evidence unchanged")
            ax.set_title(f"E25: Λ over repeated payments ({fam.replace('_', ' ')})")
        axes[0][0].set_ylabel("selection leakage Λ_n (bits)")
        axes[0][0].legend()
        out.append(_save(fig, fig_dir, "e25_repeated.png"))

    rows = [r for r in _read(os.path.join(res, "e25_workload.csv")) if r["n"] in ("1", "2", "3", "4", "5")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for k, st in enumerate(("RAND_fresh", "PCGS_fresh", "PCGS_SBS", "DET")):
            rr = [r for r in rows if r["strategy"] == st]
            ax.plot([_f(r["n"]) for r in rr], [100 * _f(r["attacker_lift"]) for r in rr], color=SERIES[k],
                    marker="o", label=st, ls="--" if st == "DET" else "-")
        ax.set_xlabel("position in (payer, receiver, policy) sequence")
        ax.set_ylabel("attacker accuracy above baseline (pp)")
        ax.set_title("E25: branch-inference attacker on the payment workload")
        ax.legend()
        out.append(_save(fig, fig_dir, "e25_workload_attack.png"))

    rows = _read(os.path.join(res, "e26_association.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(6, 2.6))
        short = {"device_continuity": "device", "account_age": "account age", "auth_freshness": "freshness",
                 "network_continuity": "network"}
        labels = [f"{short.get(r['chain_i'], r['chain_i'])} ~ {short.get(r['chain_j'], r['chain_j'])}" for r in rows]
        vals = [_f(r["tp2_significant_violations"]) / max(_f(r["tp2_minors"]), 1) * 100 for r in rows]
        _hbar(ax, labels, vals, fmt="{:.0f}%")
        ax.set_xlim(0, 110)
        ax.set_xlabel("adjacent 2x2 minors with a significant FKG (TP2) violation (%)")
        ax.set_title("E26: positive association on real login signals")
        out.append(_save(fig, fig_dir, "e26_fkg.png"))

    rows = _read(os.path.join(res, "e27_governor_ablation.csv"))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"wspace": 0.6})
        _hbar(axes[0], [r["config"] for r in rows], [_f(r["mean_exposure_weighted_Lambda_bits"]) for r in rows],
              fmt="{:.3f}")
        axes[0].set_xlim(0, 1.25 * max(_f(r["mean_exposure_weighted_Lambda_bits"]) for r in rows))
        axes[0].set_xlabel("exposure-weighted Λ per payment (bits)")
        axes[0].set_title("E27: leakage by governor configuration")
        _hbar(axes[1], [r["config"] for r in rows], [_f(r["e2e_p99_ms"]) for r in rows], color=SERIES[1], fmt="{:.1f}")
        axes[1].set_xlim(0, 1.2 * max(_f(r["e2e_p99_ms"]) for r in rows))
        axes[1].set_xlabel("end-to-end p99 latency (ms)")
        axes[1].set_title("E27: latency by configuration")
        out.append(_save(fig, fig_dir, "e27_governor.png"))

    rows = _read(os.path.join(res, "e28_grace_timeline.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        for k, cfg in enumerate(sorted({r["config"] for r in rows}, reverse=True)):
            rr = [r for r in rows if r["config"] == cfg]
            ax.plot([_f(r["minute"]) for r in rr], [100 * _f(r["availability_low_tiers"]) for r in rr],
                    color=SERIES[k], label=cfg.replace("_", " "))
        ms = [_f(r["minute"]) for r in rows if r["phase"] == "registry_outage"]
        if ms:
            ax.axvspan(min(ms), max(ms) + 1, color=GRID, alpha=0.8, lw=0)
            ax.text(min(ms) + 0.3, 88, "registry outage", color=INK2, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xlabel("minute")
        ax.set_ylabel("micro+standard accepted | satisfiable (%)")
        ax.set_title("E28: stale-status grace for low-value tiers")
        ax.legend(loc="center left")
        out.append(_save(fig, fig_dir, "e28_grace.png"))
    return out
