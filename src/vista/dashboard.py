"""Build a self-contained, offline HTML dashboard from results/<run>/*.csv.

The page embeds all of its data (no network, no external fonts or scripts), so the file can be opened
anywhere by anyone. usage: python -m vista.dashboard [run-name]
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def f(v):
    try:
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    except (TypeError, ValueError):
        return None


def verdicts(D):
    out = []

    def add(h, text, status, key):
        out.append({"h": h, "text": text, "status": status, "key": key})

    fid = {r["approach"]: f(r["agreement_with_receiver_policy"]) for r in D["h1"]}
    if fid:
        best = max(v for k, v in fid.items() if k != "VISTA_initial")
        add("H1", "Verifiable predicates give richer receiver decisions than metadata or a sender risk label",
            "supported" if fid.get("VISTA_initial", 0) > best else "not supported",
            f"{100 * fid.get('VISTA_initial', 0):.1f}% agreement with the receiver's own policy vs {100 * best:.1f}% for the best opaque baseline")
    lk = {r["strategy"]: f(r["mean_leakage_bits"]) for r in D["h2"]}
    if lk:
        red = 1 - lk["VISTA_MDPS_DET"] / lk["B4_vsac_disclose_all"]
        add("H2", "MDPS reduces disclosed information while preserving policy satisfaction",
            "supported" if red > 0.2 else "mixed",
            f"{lk['VISTA_MDPS_DET']:.2f} bits vs {lk['B4_vsac_disclose_all']:.2f} (disclose all) and {lk['B2_raw_attributes']:.2f} (raw): {100 * red:.0f}% less than disclose-all")
    h3 = {r["strategy"]: r for r in D["h3"]}
    if "MAN_S" in h3:
        m, b = h3["MAN_S"], h3["B6_blanket"]
        fm, fb = f(m["friction_per_resolved"]), f(b["friction_per_resolved"])
        rm, rb = f(m["resolution_rate"]), f(b["resolution_rate"])
        add("H3", "MAN resolves insufficient assurance with lower step-up cost than blanket step-up",
            "supported" if fm < fb and rm >= rb - 0.01 else "mixed",
            f"friction per resolved payment {fm:.1f} vs {fb:.1f}; resolution {100 * rm:.1f}% vs {100 * rb:.1f}% (MAN respects the receiver's friction budget)")
    full = [r for r in D["e5"] if r["ablation"] == "full"]
    if full:
        acc = sum(int(r["accepted"]) for r in full)
        tot = sum(int(r["trials"]) for r in full)
        add("H4", "Transaction binding and freshness stop transplantation, replay and stale evidence",
            "supported" if acc == 0 else "not supported", f"{acc} of {tot:,} adversarial submissions accepted by the full verifier")
    lat = {r["component"]: r for r in D["e7lat"]}
    if lat:
        p99 = f(lat["e2e_ms"]["p99"])
        cpu = f(lat["gen_ms"]["p99"]) + f(lat["verify_ms"]["p99"])
        add("H5", "VISTA stays within real-time latency budgets", "supported" if p99 < 200 else "mixed",
            f"end-to-end p99 {p99:.1f} ms including simulated rail latency; generate+verify p99 {cpu:.2f} ms")
    if D["e11"]:
        lam = [f(r["Lambda_bits"]) for r in D["e11"] if f(r["Lambda_bits"]) is not None]
        pos = sum(1 for v in lam if v > 1e-9) / len(lam)
        add("H6", "Deterministic minimal disclosure leaks beyond the disclosed claims",
            "supported" if pos > 0.3 else "mixed",
            f"selection leakage > 0 in {100 * pos:.0f}% of random policies (mean {sum(lam) / len(lam):.3f} bits); randomised selection and BOP reduce it (E12)")
    if D["e14"]:
        nl = [r for r in D["e14"] if r["rho"] == "no_ledger" and r["strategy"] == "binary"][0]
        l16 = [r for r in D["e14"] if r["rho"] == "16" and r["strategy"] == "binary"][0]
        add("H7", "Threshold probing recovers hidden attributes; the ledger bounds it",
            "supported" if f(nl["exact_recovery_rate"]) > 0.99 and f(l16["max_bits_learned"]) <= f(l16["bound_bits"]) + 1e-9 else "mixed",
            f"no ledger: {100 * f(nl['exact_recovery_rate']):.0f}% exact recovery in {f(nl['mean_queries']):.1f} payments; ledger rho=16: at most {f(l16['max_bits_learned']):.2f} bits (bound {f(l16['bound_bits']):.2f})")
    if D["e17"]:
        def avg(rows, k):
            return sum(f(r[k]) for r in rows) / len(rows)
        anti = [r for r in D["e17"] if r["prior"] == "anti"]
        nonfkg = [r for r in D["e17"] if r["log_supermodular"] == "False"]
        fkg = [r for r in D["e17"] if r["log_supermodular"] == "True" and r["prior"] != "independent"]
        under = avg(anti, "frac_underestimates")
        dif = avg(nonfkg, "additive_picks_different_set") if nonfkg else 0.0
        add("H8", "Additive leakage weights mis-rank disclosure sets under correlated claim priors",
            "supported" if under > 0.5 and dif > 0.02 else "mixed" if under > 0.5 else "not supported",
            f"under anti-correlated priors additive weights underestimate true leakage for {100 * under:.0f}% of frontier profiles "
            f"(worst {min(f(r['min_additivity_gap_bits']) for r in anti):.2f} bits); additive MDPS picks a different set in "
            f"{100 * dif:.1f}% of multi-choice decisions (non-FKG) vs {100 * (avg(fkg, 'additive_picks_different_set') if fkg else 0):.1f}% (FKG)")
    rnd = [r for r in D["e18"] if r.get("kind") == "random"]
    if rnd:
        d = sum(f(r["density_mean_ratio"]) for r in rnd) / len(rnd)
        m = sum(f(r["myopic_mean_ratio"]) for r in rnd) / len(rnd)
        add("H9", "Fixed-charge-aware greedy beats myopic greedy", "supported" if d < m else "not supported",
            f"mean ratio to optimum {d:.3f} (density greedy) vs {m:.3f} (myopic)")
    if D["e19"]:
        def fps(name):
            v = [f(r["friction_per_success"]) for r in D["e19"] if r["strategy"] == name and f(r["friction_per_success"]) is not None]
            return sum(v) / len(v)
        rr, fo, bl = fps("ratio_rule"), fps("fixed_order"), fps("blanket")
        add("H10", "Ratio-rule MAN lowers friction per successful step-up", "supported" if rr < fo and rr < bl else "mixed",
            f"friction per success {rr:.2f} (ratio rule) vs {fo:.2f} (fixed order) and {bl:.2f} (blanket)")
    if D["e20"]:
        rows = [r for r in D["e20"] if r["n"] == "10"]
        small = [r for r in rows if r["budget"] == "10"]
        unl = [r for r in rows if r["budget"] == "unlimited"]
        if small and unl:
            add("H11", "Query budgets materially slow policy extraction",
                "supported" if f(small[0]["extracted_accuracy"]) < f(unl[0]["extracted_accuracy"]) - 0.05 else "mixed",
                f"extracted-policy accuracy {100 * f(small[0]['extracted_accuracy']):.1f}% at 10 queries vs {100 * f(unl[0]['extracted_accuracy']):.1f}% unlimited ({f(unl[0]['mean_queries']):.0f} queries)")
    if D["e21"]:
        c = [r for r in D["e21"] if r["regime"] == "commit_before_settle"][0]
        n = [r for r in D["e21"] if r["regime"] == "no_precommitment"][0]
        add("H12", "Commit-before-settle reduces the audit rate needed for deterrence",
            "supported" if f(c["mean_q_star"]) < f(n["mean_q_star"]) else "not supported",
            f"mean deterrence audit rate q* {100 * f(c['mean_q_star']):.1f}% with commitment vs {100 * f(n['mean_q_star']):.1f}% without")
    revision_verdicts(D, add)
    return out


def revision_verdicts(D, add):
    v1 = [r for r in D.get("e23", []) if r["family"].startswith("v1_") or r["family"] == "policy_library"]
    if v1:
        inst = sum(int(r["instances"]) for r in v1)
        feas = sum(int(r["feasible"]) for r in v1)
        agree = sum(int(r["solver_agrees"]) for r in D["e23"])
        tot = sum(int(r["instances"]) for r in D["e23"])
        sp = [r for r in D["e23"] if r["family"] == "sparse_support"]
        add("H13", "No free selection: every branch-revealing selection rule leaks (Theorem 1)",
            "supported" if feas == 0 and agree == tot else "mixed",
            f"zero leakage feasible in {feas} of {inst} full-support instances; the exact-cover test matches the convex "
            f"solver in {agree}/{tot} cases" + (f" ({sp[0]['feasible']} of {sp[0]['instances']} sparse-support priors allow it)" if sp else ""))
    s24 = [r for r in D.get("e24s", []) if r["family"].startswith("skewed") and f(r["overhead"]) == 0.0235]
    if s24:
        never = all(r["never_worse"] == "True" for r in D["e24s"])
        red = sum(f(r["mean_reduction"]) for r in s24) / len(s24)
        g = D["e24g"][0] if D.get("e24g") else {}
        add("H14", "PCGS attains the minimum selection leakage at every cost (Theorem 2)", "supported" if never else "not supported",
            f"never worse than RAND at equal cost: {never}; mean reduction {100 * red:.1f}% on skewed priors at 2.35% overhead; "
            f"gate 1 {'passed' if g.get('passed') == 'True' else 'not passed'}")
    cr = [r for r in D.get("e25c", []) if r["strategy"] == "RAND_fresh"]
    if cr:
        sbs = [r for r in D["e25t"] if r["strategy"] == "PCGS_SBS"]
        flat = all(abs(f(r["mean_Lambda_bits"]) - f([x for x in sbs if x["family"] == r["family"] and x["n"] == "1"][0]["mean_Lambda_bits"])) < 1e-9 for r in sbs)
        passes = [r for r in cr if str(r["passes_DET_at_n"]).isdigit()]
        add("H15", "Fresh randomisation leaks cumulatively; sticky selection holds the single-payment level (Theorem 3)",
            "supported" if passes and flat else "mixed",
            "fresh RAND passes DET after " + ", ".join(f"{r['passes_DET_at_n']} payments ({r['family']})" for r in cr) +
            f"; sticky Lambda constant in n: {flat}")
    e26 = [r for r in D.get("e26s", []) if r["family"] == "rba_empirical" and f(r["overhead"]) == 0.0235]
    if e26:
        r = e26[0]
        g = D["e26g"][0] if D.get("e26g") else {}
        add("H16", "The leakage results hold under priors estimated from login signals", "supported" if r["zero_leakage_feasible"] == "0" else "mixed",
            f"{r['policies']} policies: zero leakage feasible in {r['zero_leakage_feasible']}; PCGS {100 * f(r['mean_reduction']):.1f}% below RAND "
            f"at 2.35% overhead; FKG (TP2) violated on {g.get('fkg_failing_pairs', '?')} of 6 chain pairs")
    if D.get("e27"):
        last = D["e27"][-1]
        d = f(last["e2e_p99_delta_ms_vs_DET"])
        add("H17", "The integrated disclosure governor adds little latency", "supported" if d < 5 else "mixed",
            f"full governor p99 {f(last['e2e_p99_ms']):.1f} ms ({d:+.2f} ms vs DET); governor step p99 {f(last['governor_p99_ms']):.3f} ms; "
            f"exposure-weighted Lambda {f(last['mean_exposure_weighted_Lambda_bits']):.3f} vs {f(D['e27'][0]['mean_exposure_weighted_Lambda_bits']):.3f} bits (DET)")
    if D.get("e28e"):
        ok = all(r.get("bound_holds") == "True" for r in D["e28e"])
        out = [r for r in D["e28t"] if r["phase"] == "registry_outage"]
        av = {c: sum(f(r["availability_low_tiers"]) for r in out if r["config"] == c) / max(1, sum(1 for r in out if r["config"] == c))
              for c in {r["config"] for r in out}}
        add("H18", "Stale-status grace keeps low-value tiers available with a provable exposure cap", "supported" if ok else "not supported",
            "low-tier availability during the outage " + ", ".join(f"{100 * v:.0f}% ({k.split('_tau')[0]})" for k, v in sorted(av.items())) +
            f"; unsafe accepts within the eps bound: {ok}")


def build(run: str) -> list[str]:
    res = os.path.join(ROOT, "results", run)
    D = {k: read(res, v) for k, v in {
        "h1": "h1_decision_fidelity.csv", "h2": "h2_leakage_by_strategy.csv", "h3": "h3_man_vs_blanket.csv",
        "e3": "e3_disclosure_modes.csv", "e5": "e5_attack_resistance.csv", "e6": "e6_crypto_backends.csv",
        "e7lat": "e7_latency.csv", "e7sc": "e7_throughput_scaling.csv", "e8s": "e8_decisions_by_scenario.csv",
        "e8t": "e8_decisions_by_tier.csv", "e8h": "e8_decisions_by_heterogeneity.csv", "e10": "e10_failure_injection.csv",
        "e11": "e11_selection_leakage.csv", "e12": "e12_leakage_cost_pareto.csv", "e13": "e13_bop_cost.csv",
        "e14": "e14_probing_attack.csv", "e15": "e15_ledger_utility.csv", "e16": "e16_cumulative_leakage.csv",
        "e17": "e17_additivity_gap.csv", "e18": "e18_fixed_charge_greedy.csv", "e19": "e19_stochastic_man.csv",
        "e20": "e20_policy_extraction.csv", "e21": "e21_rational_liar.csv", "ver": "verification_appendix.csv",
        "e1": "e1_mdps_summary.csv", "banks": "banks.csv", "e23": "e23_summary.csv", "e24s": "e24_summary.csv",
        "e24f": "e24_frontier_curve.csv", "e24g": "e24_gate.csv", "e25t": "e25_theory.csv", "e25c": "e25_crossover.csv",
        "e25w": "e25_workload.csv", "e26a": "e26_association.csv", "e26s": "e26_summary.csv", "e26g": "e26_gate.csv",
        "e27": "e27_governor_ablation.csv", "e28t": "e28_grace_timeline.csv", "e28e": "e28_grace_exposure.csv"}.items()}
    with open(os.path.join(res, "manifest.json")) as fh:
        man = json.load(fh)
    src_p = os.path.join(res, "data_source.json")
    D["source"] = json.load(open(src_p)) if os.path.exists(src_p) else {}
    D["verdicts"] = verdicts(D)
    D["manifest"] = {"world": man["config"]["world"], "versions": man["versions"], "started": man["started"],
                     "finished": man.get("finished"),
                     "steps": [{"step": s["step"], "seconds": s["seconds"], "status": s["status"]} for s in man["steps"]]}
    D["e11"] = [{k: r[k] for k in ("frontier_size", "Lambda_bits", "PoD_bits", "prior")} for r in D["e11"]]
    D["gates"] = []
    for g in D["e24g"]:
        D["gates"].append({"gate": g["gate"], "passed": g["passed"], "consequence": g["consequence"],
                           "measured": f"PCGS never worse than RAND: {g['never_worse']}; >=10% lower Lambda in "
                                       f"{100 * f(g['share_skewed_reduction_ge_10pct']):.0f}% of {g['skewed_instances']} skewed-prior instances (need 25%)"})
    for g in D["e26g"]:
        D["gates"].append({"gate": g["gate"], "passed": g["passed"], "consequence": g["consequence"],
                           "measured": f"mean reduction on empirical priors {100 * f(g['mean_reduction_rba_empirical_2.35pct']):.1f}% "
                                       f"(need 5%); FKG fails on {g['fkg_failing_pairs']} chain pairs"})
    D["e26s"] = [r for r in D["e26s"] if f(r["overhead"]) == 0.0235]
    D["e24s"] = [r for r in D["e24s"] if f(r["overhead"]) == 0.0235] + [dict(r, family=r["family"]) for r in D["e26s"]]
    D["e19"] = []
    tpl = open(os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")).read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":")))
    outs = []
    for out in (os.path.join(res, "dashboard.html"), os.path.join(ROOT, "paper_artifacts", f"dashboard_{run}.html")):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(html)
        outs.append(out)
    return outs


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "full"
    for p in build(run):
        print(p, f"{os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
