"""Cost models (Sec 3.4, R1/R8).

Cost(S) = alpha*Leakage + beta*ProofBytes + gamma*VerifyCost + delta*AcquisitionCost

Two leakage models are implemented, as the blueprint requires:
  * AdditiveCost  - per-claim sensitivity weights (controlled experiments)
  * SurprisalCost - information-theoretic surprisal -log2 pi(X >= l) (N4)
Profiles disclose the strongest threshold per attribute (subsumption closure, N1).
"""
from __future__ import annotations

import numpy as np

from .prior import Prior


class AdditiveCost:
    def __init__(self, table):
        # table[i][theta], theta = 0..d_i, table[i][0] = 0
        self.table = [np.asarray(t, dtype=float) for t in table]

    def claim(self, i: int, theta: int) -> float:
        return float(self.table[i][theta])

    def __call__(self, prof) -> float:
        return float(sum(self.table[i][int(t)] for i, t in enumerate(prof)))

    def many(self, profs: np.ndarray) -> np.ndarray:
        return np.array([self(p) for p in profs])

    def is_isotone(self) -> bool:
        return all(np.all(np.diff(t) >= 0) for t in self.table)

    @classmethod
    def random_isotone(cls, dims, rng, lo=0.2, hi=3.0):
        return cls([np.concatenate([[0.0], np.cumsum(rng.uniform(lo, hi, size=d - 1))]) for d in dims])

    @classmethod
    def random_nonisotone(cls, dims, rng, lo=0.2, hi=5.0):
        return cls([np.concatenate([[0.0], rng.uniform(lo, hi, size=d - 1)]) for d in dims])

    @classmethod
    def from_sensitivity(cls, dims, sens):
        """Weight grows linearly with the threshold level: w(i,t) = sens_i * t."""
        return cls([np.arange(d, dtype=float) * s for d, s in zip(dims, sens)])


class SurprisalCost:
    def __init__(self, prior: Prior):
        self.prior = prior

    def __call__(self, prof) -> float:
        return self.prior.surprisal(prof)

    def many(self, profs: np.ndarray) -> np.ndarray:
        return np.array([self(p) for p in profs])


class WeightedCost:
    """Scalarised combination alpha*Leak + beta*Bytes + gamma*Verify (R8: report sensitivity)."""

    def __init__(self, parts: list[tuple[float, object]]):
        self.parts = parts

    def __call__(self, prof) -> float:
        return float(sum(w * c(prof) for w, c in self.parts))

    def many(self, profs):
        return np.array([self(p) for p in profs])


class ProofSizeCost:
    """Proof bytes as a function of the disclosed profile under a backend's measured sizes.
    fixed = envelope/signature bytes paid once, per_claim = bytes per disclosed claim."""

    def __init__(self, fixed: float, per_claim: float):
        self.fixed, self.per_claim = fixed, per_claim

    def __call__(self, prof) -> float:
        k = sum(1 for t in prof if t > 0)
        return self.fixed + self.per_claim * k
