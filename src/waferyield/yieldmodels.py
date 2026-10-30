"""Die-yield models and estimation of defect density from wafer maps.

Models (A = die area in cm^2, D0 = defect density per cm^2):

    Poisson            Y = exp(-A D0)
    Murphy             Y = ((1 - exp(-A D0)) / (A D0))^2
    Seeds              Y = 1 / (1 + A D0)
    Negative binomial  Y = (1 + A D0 / alpha)^-alpha        (Stapper)

The negative binomial model has two parameters. They are estimated with the
window method: the die grid is tiled into k-die windows, and a window "passes"
only if every die in it passes. Treating a window as a die of area k*A gives
Y(k) = (1 + k A D0 / alpha)^-alpha; fitting this curve over several window
sizes separates D0 from the clustering parameter alpha. With no clustering
(alpha -> inf) the curve collapses to Y(k) = Y(1)^k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import FAIL, OFF, WaferGeometry

DEFAULT_WINDOWS = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4))


def poisson_yield(ad):
    return np.exp(-np.asarray(ad, dtype=float))


def murphy_yield(ad):
    ad = np.asarray(ad, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = ((1 - np.exp(-ad)) / ad) ** 2
    return np.where(ad == 0, 1.0, y)


def seeds_yield(ad):
    return 1.0 / (1.0 + np.asarray(ad, dtype=float))


def negbin_yield(ad, alpha):
    return (1.0 + np.asarray(ad, dtype=float) / alpha) ** (-alpha)


def invert_poisson(y: float) -> float:
    """A*D0 implied by an observed yield under the Poisson model."""
    return -np.log(y)


def wafer_yield(wmap: np.ndarray) -> float:
    valid = wmap != OFF
    return float((wmap[valid] != FAIL).mean())


def window_yields(
    maps: np.ndarray,
    windows: tuple[tuple[int, int], ...] = DEFAULT_WINDOWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled window yields over a stack of wafer maps.

    Returns (dies_per_window, yield, n_windows) arrays, one entry per window
    shape. Only windows fully on-wafer are counted.
    """
    maps = np.asarray(maps)
    if maps.ndim == 2:
        maps = maps[None]
    n, ny, nx = maps.shape
    valid = maps[0] != OFF
    fail = maps == FAIL
    ks, ys, counts = [], [], []
    for h, w in windows:
        th, tw = ny // h, nx // w
        v = valid[: th * h, : tw * w].reshape(th, h, tw, w).all(axis=(1, 3))
        f = fail[:, : th * h, : tw * w].reshape(n, th, h, tw, w).any(axis=(2, 4))
        n_win = int(v.sum()) * n
        if n_win == 0:
            continue
        good = int((~f[:, v]).sum())
        ks.append(h * w)
        ys.append(good / n_win)
        counts.append(n_win)
    return np.array(ks, float), np.array(ys), np.array(counts)


@dataclass
class NegBinFit:
    d0_per_cm2: float
    alpha: float
    die_area_cm2: float
    rmse: float

    def predict(self, die_area_cm2: float) -> float:
        return float(negbin_yield(die_area_cm2 * self.d0_per_cm2, self.alpha))


def fit_negbin_window(
    maps: np.ndarray,
    geom: WaferGeometry,
    windows: tuple[tuple[int, int], ...] = DEFAULT_WINDOWS,
) -> NegBinFit:
    """Estimate (D0, alpha) from pooled window yields by weighted least squares
    on log-yield. Weights are the binomial counts of each window size."""
    k, y, n = window_yields(maps, windows)
    if len(k) < 2:
        raise ValueError("need at least two usable window sizes")
    y = np.clip(y, 1e-6, 1 - 1e-9)
    A = geom.die_area_cm2
    # binomial standard error of log(y) ~ sqrt((1-y)/(n*y))
    sigma = np.sqrt((1 - y) / (n * y)) + 1e-6

    def resid(p):
        log_d0, log_alpha = p
        pred = negbin_yield(k * A * np.exp(log_d0), np.exp(log_alpha))
        return (np.log(pred) - np.log(y)) / sigma

    d0_init = max(-np.log(y[0]) / A, 1e-4)
    best = None
    for a0 in (0.5, 2.0, 10.0):
        sol = least_squares(resid, x0=[np.log(d0_init), np.log(a0)],
                            bounds=([np.log(1e-6), np.log(0.05)], [np.log(1e3), np.log(1e4)]))
        if best is None or sol.cost < best.cost:
            best = sol
    d0, alpha = np.exp(best.x)
    pred = negbin_yield(k * A * d0, alpha)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return NegBinFit(float(d0), float(alpha), A, rmse)


def compare_models(maps: np.ndarray, geom: WaferGeometry,
                   windows: tuple[tuple[int, int], ...] = DEFAULT_WINDOWS) -> dict[str, float]:
    """RMSE of each single-parameter model (calibrated on 1x1 yield) and the
    fitted negative binomial against observed window yields."""
    k, y, _ = window_yields(maps, windows)
    A = geom.die_area_cm2
    y1 = y[0]
    # calibrate each one-parameter model's A*D0 on the single-die yield
    from scipy.optimize import brentq

    def calib(f):
        return brentq(lambda ad: f(ad) - y1, 1e-9, 1e3)

    out = {}
    for name, f in (("poisson", poisson_yield), ("murphy", murphy_yield), ("seeds", seeds_yield)):
        ad = calib(f)
        out[name] = float(np.sqrt(np.mean((f(k * ad) - y) ** 2)))
    nb = fit_negbin_window(maps, geom, windows)
    out["negbin"] = float(np.sqrt(np.mean((negbin_yield(k * A * nb.d0_per_cm2, nb.alpha) - y) ** 2)))
    return out
