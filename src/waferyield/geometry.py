"""Die grid layout on a round wafer.

A wafer map is a 2D integer array indexed [row, col] over a rectangular die
grid that covers the wafer. Cells use these codes:

    OFF  (-1)  die site not fully inside the usable wafer area
    PASS ( 0)  die passed electrical test
    FAIL ( 1)  die failed electrical test
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

OFF = -1
PASS = 0
FAIL = 1


@dataclass(frozen=True)
class WaferGeometry:
    diameter_mm: float = 300.0
    die_w_mm: float = 10.0
    die_h_mm: float = 10.0
    edge_exclusion_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0 or self.die_w_mm <= 0 or self.die_h_mm <= 0:
            raise ValueError("wafer diameter and die size must be positive")
        if self.edge_exclusion_mm < 0 or 2 * self.edge_exclusion_mm >= self.diameter_mm:
            raise ValueError("edge exclusion must be in [0, radius)")

    @property
    def usable_radius_mm(self) -> float:
        return self.diameter_mm / 2.0 - self.edge_exclusion_mm

    @property
    def die_area_cm2(self) -> float:
        return self.die_w_mm * self.die_h_mm / 100.0

    @cached_property
    def shape(self) -> tuple[int, int]:
        ny = int(math.ceil(self.diameter_mm / self.die_h_mm))
        nx = int(math.ceil(self.diameter_mm / self.die_w_mm))
        # keep the grid symmetric about the wafer centre
        return ny + (ny % 2), nx + (nx % 2)

    @cached_property
    def die_centers_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of every die centre in mm, wafer centre at the origin."""
        ny, nx = self.shape
        xs = (np.arange(nx) - nx / 2 + 0.5) * self.die_w_mm
        ys = (np.arange(ny) - ny / 2 + 0.5) * self.die_h_mm
        x, y = np.meshgrid(xs, ys)
        return x, y

    @cached_property
    def mask(self) -> np.ndarray:
        """True where all four die corners lie inside the usable radius."""
        x, y = self.die_centers_mm
        hw, hh = self.die_w_mm / 2, self.die_h_mm / 2
        r2 = self.usable_radius_mm ** 2
        far_x = np.abs(x) + hw
        far_y = np.abs(y) + hh
        return far_x ** 2 + far_y ** 2 <= r2

    @cached_property
    def polar(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalised radius (0 at centre, 1 at usable edge) and angle in [0, 2pi)."""
        x, y = self.die_centers_mm
        r = np.hypot(x, y) / self.usable_radius_mm
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)
        return r, theta

    @property
    def gross_die(self) -> int:
        return int(self.mask.sum())

    def gross_die_estimate(self) -> float:
        """Textbook gross-die-per-wafer approximation (usable diameter d, die area S):
        pi*d^2/(4S) - pi*d/sqrt(2S)."""
        d = 2 * self.usable_radius_mm
        s = self.die_w_mm * self.die_h_mm
        return math.pi * d * d / (4 * s) - math.pi * d / math.sqrt(2 * s)

    def empty_map(self) -> np.ndarray:
        m = np.full(self.shape, OFF, dtype=np.int8)
        m[self.mask] = PASS
        return m
