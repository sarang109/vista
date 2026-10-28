"""Receiver prior pi over evidence vectors (Sec 19.3, 23).

The prior is a modelling assumption (the receiver's population model). We provide
independent, uniform and pairwise (Ising-style) priors; J >= 0 gives a log-supermodular
(MTP2 / FKG) prior, J <= 0 an anti-ferromagnetic one.
"""
from __future__ import annotations

import math

import numpy as np


class Prior:
    def __init__(self, dims, probs: np.ndarray | None = None, states: np.ndarray | None = None):
        self.dims = tuple(int(d) for d in dims)
        if states is None:
            grids = np.meshgrid(*[np.arange(d) for d in self.dims], indexing="ij")
            states = np.stack([g.ravel() for g in grids], axis=1).astype(np.int16)
        self.states = states
        n = len(states)
        p = np.full(n, 1.0 / n) if probs is None else np.asarray(probs, dtype=float)
        self.p = p / p.sum()
        self._radix = np.array([int(np.prod(self.dims[i + 1:])) for i in range(len(self.dims))])

    # -- constructors
    @classmethod
    def uniform(cls, dims):
        return cls(dims)

    @classmethod
    def independent(cls, marginals):
        dims = [len(m) for m in marginals]
        pr = cls(dims)
        logp = np.zeros(len(pr.states))
        for i, m in enumerate(marginals):
            logp += np.log(np.asarray(m, dtype=float)[pr.states[:, i]])
        pr.p = np.exp(logp - logp.max())
        pr.p /= pr.p.sum()
        return pr

    @classmethod
    def ising(cls, dims, J: np.ndarray, field: list | None = None):
        """pi(x) ~ exp( sum_i a_i(x_i) + sum_{i<j} J_ij u_i u_j ), u_i = x_i / d_i in [0,1]."""
        pr = cls(dims)
        u = pr.states / np.maximum(np.array(pr.dims) - 1, 1)
        e = np.zeros(len(pr.states))
        n = len(dims)
        for i in range(n):
            if field is not None:
                e += np.asarray(field[i], dtype=float)[pr.states[:, i]]
            for j in range(i + 1, n):
                e += J[i, j] * u[:, i] * u[:, j]
        pr.p = np.exp(e - e.max())
        pr.p /= pr.p.sum()
        return pr

    @classmethod
    def random_ising(cls, dims, rng, kind: str = "ferro", strength: float = 2.0, field_sd: float = 0.7):
        """field_sd sets how skewed the single-attribute marginals are (larger = more skewed)."""
        n = len(dims)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mag = rng.uniform(0, strength)
                if kind == "ferro":
                    J[i, j] = mag
                elif kind == "anti":
                    J[i, j] = -mag
                elif kind == "mixed":
                    J[i, j] = mag * rng.choice([-1, 1])
        field = [rng.normal(0, field_sd, size=d) for d in dims]
        return cls.ising(dims, J, field)

    # -- queries
    def index_of(self, state) -> int:
        return int(np.dot(np.asarray(state), self._radix))

    def mask_ge(self, prof) -> np.ndarray:
        return (self.states >= np.asarray(prof)).all(1)

    def mass_ge(self, prof) -> float:
        return float(self.p[self.mask_ge(prof)].sum())

    def surprisal(self, prof) -> float:
        m = self.mass_ge(prof)
        return -math.log2(m) if m > 0 else math.inf

    def sample(self, rng, k: int) -> np.ndarray:
        idx = rng.choice(len(self.p), size=k, p=self.p)
        return self.states[idx]

    def is_log_supermodular(self, tol: float = 1e-12, max_pairs: int = 200_000, rng=None) -> tuple[bool, int]:
        """Check pi(x v y) pi(x ^ y) >= pi(x) pi(y); returns (holds, violations)."""
        N = len(self.states)
        rng = rng or np.random.default_rng(0)
        if N * N <= max_pairs:
            I, Jx = np.triu_indices(N, 1)
        else:
            I = rng.integers(0, N, max_pairs)
            Jx = rng.integers(0, N, max_pairs)
        X, Y = self.states[I], self.states[Jx]
        up = np.maximum(X, Y) @ self._radix
        lo = np.minimum(X, Y) @ self._radix
        lhs = np.log(self.p[up] + 1e-300) + np.log(self.p[lo] + 1e-300)
        rhs = np.log(self.p[I] + 1e-300) + np.log(self.p[Jx] + 1e-300)
        v = int((lhs < rhs - 1e-9).sum())
        return v == 0, v
