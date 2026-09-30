"""One-command reproduction: python -m vista.run_all --config configs/experiments/smoke.json

Runs every experiment, writes CSVs under results/<run-name>/, regenerates figures and tables under
paper_artifacts/, and records a manifest (config, versions, hardware, timings, file hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _versions():
    import cryptography
    import numpy
    import scipy
    v = {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
         "cryptography": cryptography.__version__, "platform": platform.platform(), "machine": platform.machine(),
         "cpus": os.cpu_count()}
    try:
        import gmpy2
        v["gmpy2"] = gmpy2.version()
    except ImportError:
        v["gmpy2"] = "absent (pure-Python fallback)"
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "experiments", "smoke.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="", help="comma-separated step names")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out = args.out or os.path.join(ROOT, "results", name)
    os.makedirs(out, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    from .experiments import exp_attacks, exp_crypto, exp_e2e, exp_ledger, exp_mdps, exp_misc, exp_selection
    from .experiments import exp_verification
    from .simulator.world import World

    world_holder = {}

    def world():
        if "w" not in world_holder:
            wc = cfg["world"]
            world_holder["w"] = World(min(wc["n_banks"], 30), min(wc["n_payers"], 30_000), wc["seed"],
                                      min(wc.get("prior_sample", 20_000), 20_000))
        return world_holder["w"]

    steps = [
        ("verification", lambda: exp_verification.run(cfg, out)),
        ("e1", lambda: exp_mdps.run_e1(cfg, out)),
        ("e2", lambda: exp_mdps.run_e2(cfg, out, world())),
        ("e4", lambda: exp_misc.run_e4(cfg, out, world())),
        ("e5", lambda: exp_attacks.run(cfg, out, world())),
        ("e6", lambda: exp_crypto.run_e6(cfg, out)),
        ("e11", lambda: exp_selection.run_e11(cfg, out)),
        ("e12", lambda: exp_selection.run_e12(cfg, out)),
        ("e13", lambda: exp_crypto.run_e13(cfg, out)),
        ("e14", lambda: exp_ledger.run_e14(cfg, out)),
        ("e15", lambda: exp_ledger.run_e15(cfg, out)),
        ("e16", lambda: exp_ledger.run_e16(cfg, out)),
        ("e17", lambda: exp_misc.run_e17(cfg, out)),
        ("e18", lambda: exp_misc.run_e18(cfg, out)),
        ("e19", lambda: exp_misc.run_e19(cfg, out)),
        ("e20", lambda: exp_misc.run_e20(cfg, out)),
        ("e21", lambda: exp_misc.run_e21(cfg, out)),
        ("e22", lambda: exp_misc.run_e22(cfg, out)),
        ("e2e", lambda: exp_e2e.run(cfg, out)),
        ("e10", lambda: exp_e2e.run_e10(cfg, out)),
    ]
    manifest = {"name": name, "config": cfg, "versions": _versions(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "steps": []}
    prev_path = os.path.join(out, "manifest.json")
    prev_steps = []
    if only and os.path.exists(prev_path):  # partial re-run: keep the record of steps not re-run
        with open(prev_path) as fh:
            prev = json.load(fh)
        manifest["started"] = prev.get("started", manifest["started"])
        prev_steps = [st for st in prev.get("steps", []) if st["step"] not in only]
    failed = False
    for step, fn in steps:
        if only and step not in only:
            continue
        t0 = time.time()
        print(f"[vista] {step:>12} ...", flush=True, end=" ")
        try:
            files = fn()
            dt = time.time() - t0
            print(f"done in {dt:6.1f}s -> {', '.join(os.path.basename(f) for f in files)}", flush=True)
            manifest["steps"].append({"step": step, "seconds": round(dt, 2), "status": "ok",
                                      "files": [os.path.relpath(f, ROOT) for f in files]})
        except Exception as e:  # keep going; record the failure honestly
            failed = True
            print(f"FAILED: {e}", flush=True)
            traceback.print_exc()
            manifest["steps"].append({"step": step, "seconds": round(time.time() - t0, 2), "status": "failed",
                                      "error": repr(e)})

    order = [st for st, _ in steps]
    manifest["steps"] = sorted(prev_steps + manifest["steps"], key=lambda st: order.index(st["step"]))
    from .figures import make_all
    figs = make_all(out, os.path.join(ROOT, "paper_artifacts", "figures", name))
    tab_dir = os.path.join(ROOT, "paper_artifacts", "tables", name)
    os.makedirs(tab_dir, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".csv") and "transactions" not in f:
            shutil.copy(os.path.join(out, f), tab_dir)
    manifest["figures"] = [os.path.relpath(f, ROOT) for f in figs]
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["sha256"] = {}
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p) and f.endswith(".csv"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            manifest["sha256"][f] = h.hexdigest()
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[vista] results: {out}\n[vista] figures: {len(figs)} in paper_artifacts/figures/{name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
