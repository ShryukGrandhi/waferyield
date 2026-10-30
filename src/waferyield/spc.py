"""Statistical process control for per-wafer yield (or any per-wafer metric).

* Shewhart individuals chart with the four Western Electric run rules
* EWMA chart with time-varying limits
* Tabular CUSUM (one-sided, detects downward shifts in yield)

Control limits are estimated from a Phase-I baseline, then applied to new
data in Phase II.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Baseline:
    mean: float
    sigma: float

    @classmethod
    def from_data(cls, x: np.ndarray, method: str = "mr") -> "Baseline":
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            raise ValueError("baseline needs at least 2 points")
        if method == "mr":
            # moving-range estimate, robust to slow drift: sigma = MRbar / d2(n=2)
            sigma = np.mean(np.abs(np.diff(x))) / 1.128
        elif method == "std":
            sigma = x.std(ddof=1)
        else:
            raise ValueError(method)
        if sigma <= 0:
            raise ValueError("baseline has zero variation")
        return cls(float(x.mean()), float(sigma))


@dataclass
class Signal:
    index: int
    rule: str


@dataclass
class ChartResult:
    statistic: np.ndarray
    signals: list[Signal] = field(default_factory=list)

    def first_signal(self) -> int | None:
        return min((s.index for s in self.signals), default=None)


def western_electric(x: np.ndarray, base: Baseline,
                     rules: tuple[int, ...] = (1, 2, 3, 4)) -> ChartResult:
    """Western Electric rules on standardised individuals.

    1: one point beyond 3 sigma
    2: two of three consecutive points beyond 2 sigma, same side
    3: four of five consecutive points beyond 1 sigma, same side
    4: eight consecutive points on the same side of the centre line
    """
    z = (np.asarray(x, dtype=float) - base.mean) / base.sigma
    sig: list[Signal] = []
    n = len(z)
    for i in range(n):
        if 1 in rules and abs(z[i]) > 3:
            sig.append(Signal(i, "WE1"))
            continue
        for rule, win, need, thr in ((2, 3, 2, 2.0), (3, 5, 4, 1.0)):
            if rule in rules and i + 1 >= win:
                w = z[i + 1 - win: i + 1]
                # the current point must itself be part of the run
                if (z[i] > thr and (w > thr).sum() >= need) or \
                   (z[i] < -thr and (w < -thr).sum() >= need):
                    sig.append(Signal(i, f"WE{rule}"))
                    break
        else:
            if 4 in rules and i + 1 >= 8:
                w = z[i - 7: i + 1]
                if (w > 0).all() or (w < 0).all():
                    sig.append(Signal(i, "WE4"))
    return ChartResult(z, sig)


def ewma(x: np.ndarray, base: Baseline, lam: float = 0.2, L: float = 2.86) -> ChartResult:
    """EWMA chart. lam=0.2, L=2.86 gives an in-control ARL near 500 for normal data."""
    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    z = np.empty_like(x)
    prev = base.mean
    sig = []
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev
        var = base.sigma ** 2 * lam / (2 - lam) * (1 - (1 - lam) ** (2 * (i + 1)))
        if abs(prev - base.mean) > L * np.sqrt(var):
            sig.append(Signal(i, "EWMA"))
    return ChartResult(z, sig)


def cusum_low(x: np.ndarray, base: Baseline, k: float = 0.5, h: float = 4.77) -> ChartResult:
    """Lower one-sided tabular CUSUM in sigma units (detects yield drops).
    k=0.5, h=4.77 gives a one-sided in-control ARL near 740 for normal data."""
    z = (np.asarray(x, dtype=float) - base.mean) / base.sigma
    s = np.empty_like(z)
    c = 0.0
    sig = []
    for i, zi in enumerate(z):
        c = max(0.0, c - zi - k)
        s[i] = c
        if c > h:
            sig.append(Signal(i, "CUSUM"))
            c = 0.0  # restart after a signal
    return ChartResult(s, sig)


def run_length(result: ChartResult, start: int = 0) -> int | None:
    """Points from `start` until the first signal at or after it (1-based)."""
    idx = [s.index for s in result.signals if s.index >= start]
    return (min(idx) - start + 1) if idx else None
