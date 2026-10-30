"""Fixed-length spatial features for wafer maps.

The features are hand-designed so that each systematic signature lands in a
distinct region of feature space:

  * radial profile (equal-area annuli)        -> center, donut, edge-ring
  * angular profile of the outer band          -> edge-ring vs edge-loc
  * Moran's I spatial autocorrelation          -> random vs clustered
  * shape of the largest connected fail blob   -> scratch (elongated), loc
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import FAIL, WaferGeometry

N_ANNULI = 6
N_SECTORS = 8
OUTER_R = 0.78

_EIGHT = np.ones((3, 3), dtype=bool)
_ROOK = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)


def feature_names() -> list[str]:
    names = ["fail_frac"]
    names += [f"annulus_{i}" for i in range(N_ANNULI)]
    names += ["radial_corr"]
    names += ["outer_sector_max", "outer_sector_cv", "outer_sector_cover",
              "inner_sector_cv"]
    names += ["morans_i"]
    names += ["blob_frac_of_fails", "blob_frac_of_wafer", "blob_elongation",
              "blob_fill", "blob_r_mean", "blob_r_std", "blob_count_norm"]
    return names


class FeatureExtractor:
    """Precomputes geometry-dependent index arrays once, then featurises maps fast."""

    def __init__(self, geom: WaferGeometry):
        self.geom = geom
        mask = geom.mask
        r, theta = geom.polar
        self.mask = mask
        self.r = r
        self.n_valid = int(mask.sum())

        # equal-area annuli: bin on r^2
        r2 = np.clip(r ** 2, 0, 1 - 1e-12)
        self.annulus = np.where(mask, (r2 * N_ANNULI).astype(int), -1)
        self.annulus_size = np.bincount(self.annulus[mask], minlength=N_ANNULI)

        sector = (theta / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
        outer = mask & (r >= OUTER_R)
        inner = mask & (r < OUTER_R)
        self.outer_sector = np.where(outer, sector, -1)
        self.inner_sector = np.where(inner, sector, -1)
        self.outer_sector_size = np.bincount(sector[outer], minlength=N_SECTORS)
        self.inner_sector_size = np.bincount(sector[inner], minlength=N_SECTORS)

        # rook-neighbour weights restricted to on-wafer dies
        mf = mask.astype(float)
        self.neigh_count = ndimage.convolve(mf, _ROOK, mode="constant") * mf
        self.w_total = float(self.neigh_count.sum())
        self.r_valid = r[mask]
        self.r_valid_c = self.r_valid - self.r_valid.mean()
        self.r_valid_ss = float((self.r_valid_c ** 2).sum())

    def __call__(self, wmap: np.ndarray) -> np.ndarray:
        return self.transform_one(wmap)

    def transform_one(self, wmap: np.ndarray) -> np.ndarray:
        if wmap.shape != self.mask.shape:
            raise ValueError(f"map shape {wmap.shape} != geometry shape {self.mask.shape}")
        fail = (wmap == FAIL) & self.mask
        fv = fail[self.mask].astype(float)
        n_fail = fv.sum()
        frac = n_fail / self.n_valid
        out = [frac]

        ann = np.bincount(self.annulus[fail], minlength=N_ANNULI) / self.annulus_size
        out += list(ann)

        # point-biserial correlation of fail indicator with radius
        fc = fv - fv.mean()
        denom = np.sqrt(self.r_valid_ss * (fc ** 2).sum())
        out.append(float((fc * self.r_valid_c).sum() / denom) if denom > 0 else 0.0)

        outer = np.bincount(self.outer_sector[fail & (self.outer_sector >= 0)],
                            minlength=N_SECTORS) / self.outer_sector_size
        inner = np.bincount(self.inner_sector[fail & (self.inner_sector >= 0)],
                            minlength=N_SECTORS) / self.inner_sector_size
        o_mean = outer.mean()
        out.append(outer.max())
        out.append(outer.std() / o_mean if o_mean > 0 else 0.0)
        out.append(float((outer >= 0.5 * outer.max()).mean()) if outer.max() > 0 else 0.0)
        i_mean = inner.mean()
        out.append(inner.std() / i_mean if i_mean > 0 else 0.0)

        out.append(self._morans_i(fail, fv))
        out += self._blob_features(fail, n_fail)
        return np.asarray(out, dtype=float)

    def transform(self, maps: np.ndarray) -> np.ndarray:
        return np.stack([self.transform_one(m) for m in maps])

    def _morans_i(self, fail: np.ndarray, fv: np.ndarray) -> float:
        """Moran's I of the fail indicator with rook adjacency over on-wafer dies."""
        z = np.zeros(self.mask.shape)
        mean = fv.mean()
        var = ((fv - mean) ** 2).sum()
        if var == 0:
            return 0.0
        z[self.mask] = fv - mean
        lag = ndimage.convolve(z, _ROOK, mode="constant")
        num = float((z * lag)[self.mask].sum())
        return (self.n_valid / self.w_total) * num / var

    def _blob_features(self, fail: np.ndarray, n_fail: float) -> list[float]:
        labels, n = ndimage.label(fail, structure=_EIGHT)
        if n == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sizes = np.bincount(labels.ravel())[1:]
        k = int(sizes.argmax()) + 1
        size = float(sizes[k - 1])
        ys, xs = np.nonzero(labels == k)
        if size >= 3:
            cov = np.cov(np.vstack([xs, ys]).astype(float))
            ev = np.linalg.eigvalsh(cov)
            elong = 1.0 - (ev[0] + 1e-9) / (ev[1] + 1e-9)
        else:
            elong = 0.0
        bbox = (np.ptp(xs) + 1) * (np.ptp(ys) + 1)
        rb = self.r[ys, xs]
        return [
            size / n_fail,
            size / self.n_valid,
            float(elong),
            size / bbox,
            float(rb.mean()),
            float(rb.std()),
            n / self.n_valid,
        ]
