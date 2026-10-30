"""Synthetic wafer-map generator with known ground truth.

Each wafer gets a background of random (point) defects plus, optionally, one
systematic spatial signature. The signature classes follow the ones used in
public fab datasets such as WM-811K: center, donut, edge-ring, edge-loc,
loc (local cluster), scratch, random, and none.

Random defects follow the compound-Poisson (negative binomial) model used in
yield engineering: each wafer draws a defect-density multiplier
G ~ Gamma(alpha, 1/alpha), and each die then receives Poisson(A * D0 * G)
defects. A die with one or more defects fails. Pooled over many wafers this
gives the Stapper yield Y = (1 + A*D0/alpha)^-alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import FAIL, OFF, PASS, WaferGeometry

PATTERNS = (
    "none",
    "random",
    "center",
    "donut",
    "edge_ring",
    "edge_loc",
    "loc",
    "scratch",
)


@dataclass(frozen=True)
class RandomDefectModel:
    d0_per_cm2: float = 0.1
    alpha: float = 2.0  # cluster parameter; large alpha -> Poisson

    def die_fail_mask(self, geom: WaferGeometry, rng: np.random.Generator) -> np.ndarray:
        g = rng.gamma(self.alpha, 1.0 / self.alpha) if np.isfinite(self.alpha) else 1.0
        lam = geom.die_area_cm2 * self.d0_per_cm2 * g
        counts = rng.poisson(lam, size=geom.shape)
        return (counts > 0) & geom.mask


def _bernoulli(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.random(p.shape) < p


def _seg_distance(px, py, ax, ay, bx, by):
    """Distance from points (px, py) to segment a-b, vectorised over points."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _angle_diff(a, b):
    d = np.mod(a - b, 2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def pattern_fail_mask(pattern: str, geom: WaferGeometry, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of dies killed by a systematic signature."""
    r, theta = geom.polar
    x, y = geom.die_centers_mm
    R = geom.usable_radius_mm
    shape = geom.shape

    if pattern in ("none", "random"):
        p = np.zeros(shape)
    elif pattern == "center":
        sigma = rng.uniform(0.18, 0.35)
        p = rng.uniform(0.75, 0.95) * np.exp(-0.5 * (r / sigma) ** 2)
    elif pattern == "donut":
        r0 = rng.uniform(0.4, 0.6)
        w = rng.uniform(0.06, 0.1)
        p = rng.uniform(0.7, 0.95) * np.exp(-0.5 * ((r - r0) / w) ** 2)
    elif pattern == "edge_ring":
        w = rng.uniform(0.08, 0.16)
        p = np.where(r > 1.0 - w, rng.uniform(0.6, 0.95), 0.0)
    elif pattern == "edge_loc":
        w = rng.uniform(0.12, 0.25)
        center = rng.uniform(0, 2 * np.pi)
        half_span = np.deg2rad(rng.uniform(20, 50))
        in_arc = _angle_diff(theta, center) < half_span
        p = np.where((r > 1.0 - w) & in_arc, rng.uniform(0.6, 0.9), 0.0)
    elif pattern == "loc":
        rad = rng.uniform(0.15, 0.7)
        ang = rng.uniform(0, 2 * np.pi)
        cx, cy = rad * R * np.cos(ang), rad * R * np.sin(ang)
        sigma = rng.uniform(0.06, 0.11) * R
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        p = rng.uniform(0.75, 0.95) * np.exp(-0.5 * d2 / sigma ** 2)
    elif pattern == "scratch":
        # a slightly bent scratch: two joined segments
        p0 = rng.uniform(-0.6, 0.6, size=2) * R
        heading = rng.uniform(0, np.pi)
        length = rng.uniform(0.6, 1.2) * R
        bend = rng.normal(0, 0.15)
        p1 = p0 + 0.5 * length * np.array([np.cos(heading), np.sin(heading)])
        p2 = p1 + 0.5 * length * np.array([np.cos(heading + bend), np.sin(heading + bend)])
        width = 0.55 * min(geom.die_w_mm, geom.die_h_mm)
        d = np.minimum(
            _seg_distance(x, y, *p0, *p1),
            _seg_distance(x, y, *p1, *p2),
        )
        p = np.where(d < width, rng.uniform(0.8, 1.0), 0.0)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    return _bernoulli(p, rng) & geom.mask


def background_model(pattern: str, rng: np.random.Generator) -> RandomDefectModel:
    if pattern == "random":
        return RandomDefectModel(d0_per_cm2=rng.uniform(0.6, 1.4), alpha=50.0)
    return RandomDefectModel(d0_per_cm2=rng.uniform(0.02, 0.12), alpha=2.0)


def make_wafer(
    pattern: str,
    geom: WaferGeometry,
    rng: np.random.Generator,
    background: RandomDefectModel | None = None,
) -> np.ndarray:
    """Return an int8 wafer map (OFF / PASS / FAIL) with the given signature."""
    bg = background if background is not None else background_model(pattern, rng)
    fail = bg.die_fail_mask(geom, rng) | pattern_fail_mask(pattern, geom, rng)
    m = np.full(geom.shape, OFF, dtype=np.int8)
    m[geom.mask] = PASS
    m[fail] = FAIL
    return m


def make_dataset(
    n_per_class: int,
    geom: WaferGeometry,
    seed: int = 0,
    patterns: tuple[str, ...] = PATTERNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced labelled set: maps of shape (N, ny, nx) and string labels (N,)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array(patterns), n_per_class)
    rng.shuffle(labels)
    maps = np.stack([make_wafer(str(lab), geom, rng) for lab in labels])
    return maps, labels


def make_lot(
    n_wafers: int,
    geom: WaferGeometry,
    model: RandomDefectModel,
    seed: int = 0,
) -> np.ndarray:
    """Wafers carrying only random defects from one density model."""
    rng = np.random.default_rng(seed)
    return np.stack([make_wafer("none", geom, rng, background=model) for _ in range(n_wafers)])


def simulate_wafer_yields(
    n_wafers: int,
    geom: WaferGeometry,
    model: RandomDefectModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-wafer yields drawn from the same distribution as `make_lot`, without
    building the maps: given the wafer multiplier G, die fails are independent
    with p = 1 - exp(-A D0 G), so the fail count is Binomial(gross_die, p)."""
    if np.isfinite(model.alpha):
        g = rng.gamma(model.alpha, 1.0 / model.alpha, size=n_wafers)
    else:
        g = np.ones(n_wafers)
    p = 1.0 - np.exp(-geom.die_area_cm2 * model.d0_per_cm2 * g)
    n = geom.gross_die
    return 1.0 - rng.binomial(n, p) / n
