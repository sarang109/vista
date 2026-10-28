"""Sender-sharded scans of the payment stream for stateful sender components (sticky store, ledger, budget).

Each sender institution runs its own governor, so the stream is partitioned by sender bank: a shard owns a
set of banks and sees all of their payments in stream order, which keeps every per-account state consistent
without sharing anything between processes. Shards are balanced greedily by market share.
"""
from __future__ import annotations

import numpy as np

from ..simulator.source import _mix


def bank_shards(banks, k: int) -> list[list[int]]:
    load = [0.0] * k
    out: list[list[int]] = [[] for _ in range(k)]
    for b in sorted(range(len(banks)), key=lambda i: -banks[i].share):
        j = int(np.argmin(load))
        out[j].append(b)
        load[j] += banks[b].share
    return [s for s in out if s]


def payer_selected(payer: np.ndarray, fraction: float, salt: int = 1_000_003) -> np.ndarray:
    """Stable payer subsample: keeps all payments of a selected payer, so repeat sequences stay intact.
    Different salts give independent subsamples (e.g. the sample and its train/test split)."""
    if fraction >= 1.0:
        return np.ones(len(payer), dtype=bool)
    h = _mix(payer.astype(np.int64) + salt)
    return (h >> np.uint64(11)).astype(np.float64) / 2.0 ** 53 < fraction


def iter_shard(src, world, shard: list[int], n_rows: int, payer_fraction: float, block: int = 100_000):
    """Yield (pay, local index j) for every payment of the shard's banks among the first n_rows rows."""
    banks = np.asarray(shard)
    for a in range(0, n_rows, block):
        pay = src.take(np.arange(a, min(a + block, n_rows)), world)
        keep = np.flatnonzero(np.isin(pay["sender"], banks) & payer_selected(pay["payer"], payer_fraction))
        for j in keep:
            yield pay, int(j)
