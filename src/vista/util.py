"""Shared helpers: canonical encoding, hashing, timing statistics, CSV output."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np


def _check_canonical(o):
    if isinstance(o, float):
        raise TypeError("floats are not allowed in canonical encodings; use integers or strings")
    if isinstance(o, dict):
        for k, v in o.items():
            if not isinstance(k, str):
                raise TypeError("canonical dict keys must be strings")
            _check_canonical(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _check_canonical(v)


def canonical_json(obj) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, no floats."""
    _check_canonical(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def percentiles(xs: Iterable[float]) -> dict:
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.apply_along_axis(stat, 1, a[idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


@contextmanager
def timer():
    t = {"ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t["ms"] = (time.perf_counter() - t0) * 1000.0


def write_csv(path: str, rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})
    return path


def _fmt(v):
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


def log2_safe(p: float) -> float:
    return math.log2(p) if p > 0 else -math.inf
