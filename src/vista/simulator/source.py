"""Transaction sources for the end-to-end run: synthetic arrays or the public PaySim log.

PaySim (E. Lopez-Rojas et al.) is a public mobile-money transaction simulator calibrated on real
logs: 6,362,620 rows, 8,213 labelled frauds. This module downloads it once, verifies it against the
published statistics, and converts it to memory-mapped .npy columns so that every worker process can
read any slice without loading the file (RAM stays flat regardless of dataset size).

What PaySim provides: amount, hourly timing, counterparties, isFraud. What stays synthetic: which bank
an account belongs to and all security evidence (no public dataset contains interbank authentication
evidence). PaySim fraud rows map to the `ato_like` scenario.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import urllib.request
from collections import Counter

import numpy as np

from . import workload as wl

URL = ("https://huggingface.co/datasets/vitaliy-sharandin/synthetic-fraud-detection/resolve/main/"
       "PS_20174392719_1491204439457_log.csv")
# Published PaySim statistics (Kaggle "Synthetic Financial Datasets For Fraud Detection").
EXPECTED = {"rows": 6_362_620, "fraud": 8_213, "flagged": 16, "steps": 743,
            "types": {"CASH_OUT": 2_237_500, "PAYMENT": 2_151_495, "CASH_IN": 1_399_284,
                      "TRANSFER": 532_909, "DEBIT": 41_432}}
PINNED_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"  # Hugging Face copy
CACHE_VERSION = 1
TYPE_CODE = {"TRANSFER": 0, "CASH_OUT": 1, "PAYMENT": 2, "DEBIT": 3}  # CASH_IN is not a payment
PURPOSE_OF = np.array([0, 0, 2, 1], dtype=np.int8)  # TRANSFER/CASH_OUT -> p2p, PAYMENT -> merchant, DEBIT -> bill
COLS = ("step", "type", "amount_minor", "orig", "dest", "dest_m", "fraud")
_DT = {"step": np.int16, "type": np.int8, "amount_minor": np.int64, "orig": np.int64, "dest": np.int64,
       "dest_m": np.int8, "fraud": np.int8}


class PaySimMismatch(ValueError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download(path: str, url: str = URL, report=None) -> str:
    """Stream the CSV to `path` (atomic rename). No account or key is needed for this mirror."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "vista-artifact/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if report:
                report(done, total, "MB-downloaded", 1e-6)
    os.replace(tmp, path)
    return path


def build_cache(csv_path: str, cache_dir: str, expected: dict | None = EXPECTED, report=None) -> dict:
    """One streaming pass over the CSV. Verifies the published statistics (refuses on mismatch),
    drops CASH_IN, and writes compact column files. Returns the metadata."""
    size = os.path.getsize(csv_path)
    sha = sha256_file(csv_path)
    parts: dict[str, list] = {k: [] for k in COLS}
    buf: dict[str, list] = {k: [] for k in COLS}
    stats = {"rows": 0, "fraud": 0, "flagged": 0, "types": Counter(), "max_step": 0}
    t0 = time.time()

    def flush():
        for k in COLS:
            parts[k].append(np.asarray(buf[k], dtype=_DT[k]))
            buf[k].clear()

    with open(csv_path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {h: i for i, h in enumerate(head)}
        need = ("step", "type", "amount", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud")
        if any(n not in ix for n in need):
            raise PaySimMismatch(f"not a PaySim file; header = {head}")
        for row in rd:
            stats["rows"] += 1
            t = row[ix["type"]]
            stats["types"][t] += 1
            step = int(row[ix["step"]])
            stats["max_step"] = max(stats["max_step"], step)
            fr = row[ix["isFraud"]] == "1"
            stats["fraud"] += fr
            stats["flagged"] += row[ix["isFlaggedFraud"]] == "1"
            if t in TYPE_CODE:
                nd = row[ix["nameDest"]]
                buf["step"].append(step)
                buf["type"].append(TYPE_CODE[t])
                buf["amount_minor"].append(max(1, int(round(float(row[ix["amount"]]) * 100))))
                buf["orig"].append(int(row[ix["nameOrig"]][1:]))
                buf["dest"].append(int(nd[1:]))
                buf["dest_m"].append(1 if nd[0] == "M" else 0)
                buf["fraud"].append(1 if fr else 0)
            if stats["rows"] % 500_000 == 0:
                flush()
                if report:
                    report(stats["rows"], expected["rows"] if expected else 0, "rows-parsed", 1.0)
    flush()
    if expected:
        got = {"rows": stats["rows"], "fraud": stats["fraud"], "flagged": stats["flagged"],
               "steps": stats["max_step"], "types": dict(stats["types"])}
        if got != expected:
            raise PaySimMismatch(f"file does not match published PaySim statistics.\n expected {expected}\n got      {got}")
    os.makedirs(cache_dir, exist_ok=True)
    n_use = 0
    for k in COLS:
        arr = np.concatenate(parts[k])
        np.save(os.path.join(cache_dir, k + ".npy"), arr)
        n_use = len(arr)
    meta = {"version": CACHE_VERSION, "source_sha256": sha, "source_bytes": size,
            "sha256_matches_pinned_hf_copy": sha == PINNED_SHA256, "rows_total": stats["rows"],
            "rows_usable": n_use, "rows_excluded_cash_in": stats["rows"] - n_use,
            "fraud_total": stats["fraud"], "fraud_usable": int(np.concatenate(parts["fraud"]).sum()),
            "flagged_total": stats["flagged"], "types": dict(stats["types"]), "max_step": stats["max_step"],
            "verified_against_published_statistics": bool(expected), "parse_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def ensure_cache(csv_path: str, cache_dir: str, url: str = URL, allow_download: bool = True, report=None) -> dict:
    """Download (if needed), verify and cache. Idempotent: a valid cache is reused."""
    meta_p = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(csv_path):
        if not allow_download:
            raise FileNotFoundError(csv_path)
        download(csv_path, url, report)
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            meta = json.load(fh)
        ok = (meta.get("version") == CACHE_VERSION and meta.get("source_bytes") == os.path.getsize(csv_path)
              and all(os.path.exists(os.path.join(cache_dir, k + ".npy")) for k in COLS))
        if ok and meta.get("source_sha256") == sha256_file(csv_path):
            return meta
    return build_cache(csv_path, cache_dir, EXPECTED, report)


# ------------------------------------------------------------------------------------------ sources
_K = np.uint64(0x9E3779B97F4A7C15)
_M = np.uint64(0xBF58476D1CE4E5B9)


def _mix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) * _K
    x ^= x >> np.uint64(29)
    x *= _M
    x ^= x >> np.uint64(32)
    return x


class PaySimSource:
    kind = "paysim"

    def __init__(self, cache_dir: str, limit: int | None = None, amount_scale: float = 1.0):
        self.arr = {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r") for k in COLS}
        with open(os.path.join(cache_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        n = len(self.arr["step"])
        self.n = min(n, limit) if limit else n
        self.amount_scale = amount_scale

    def take(self, idx: np.ndarray, world) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        a = {k: np.asarray(v[idx]) for k, v in self.arr.items()}
        pop = world.pop
        payer = (_mix(a["orig"]) % np.uint64(pop.n)).astype(np.int64)
        offset = (_mix(idx + 7) % np.uint64(3600)).astype(np.int64)  # deterministic within-hour position
        ts = wl.T0 + (a["step"].astype(np.int64) - 1) * 3600 + offset
        amount = a["amount_minor"]
        if self.amount_scale != 1.0:
            amount = np.maximum(1, np.round(amount * self.amount_scale)).astype(np.int64)
        sender = pop.bank[payer]
        cum = np.cumsum([b.share for b in world.banks])
        h = _mix(a["dest"] * 2 + a["dest_m"])
        recv = np.minimum(np.searchsorted(cum, (h >> np.uint64(11)).astype(np.float64) / 2.0 ** 53), len(cum) - 1)
        nb = len(world.banks)
        recv = np.where(recv == sender, (recv + 1 + (h % np.uint64(nb - 1)).astype(np.int64)) % nb, recv)
        fraud = a["fraud"].astype(np.int64)
        return {"payer": payer, "ts": ts, "amount_minor": amount, "purpose": PURPOSE_OF[a["type"]].astype(np.int64),
                "scenario": fraud, "sender": sender, "receiver": recv, "payee_slot": a["dest"] % 10_000_000,
                "new_payee": fraud > 0, "gidx": idx}

    def describe(self) -> dict:
        return dict(self.meta, kind="paysim", rows_used=int(self.n), amount_scale=self.amount_scale)


class ArraySource:
    kind = "synthetic"

    def __init__(self, pay: dict):
        self.pay = pay
        self.n = len(pay["ts"])

    def take(self, idx: np.ndarray, world=None) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        d = {k: v[idx] for k, v in self.pay.items()}
        d["gidx"] = idx
        return d

    def describe(self) -> dict:
        return {"kind": "synthetic", "rows_used": int(self.n)}


def resolve_path(p: str) -> str:
    from .world import ROOT
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def make_source(world, wc: dict):
    """PaySim when wc['source'] == 'paysim' (cache must exist; see ensure_cache), else synthetic."""
    if wc.get("source") == "paysim":
        cache = resolve_path(wc.get("paysim_cache", "data/paysim_cache"))
        return PaySimSource(cache, wc.get("n_payments"), float(wc.get("paysim_amount_scale", 1.0)))
    rng = np.random.default_rng(wc["seed"] + 100)
    return ArraySource(wl.generate_payments(world.pop, world.banks, wc["n_payments"], wc.get("days", 7), rng))
