"""Plain-text rendering of wafer maps."""

from __future__ import annotations

import numpy as np

from .geometry import FAIL, OFF

CHARS = {OFF: " ", 0: ".", FAIL: "X"}


def ascii_map(wmap: np.ndarray) -> str:
    rows = []
    for row in np.asarray(wmap):
        rows.append(" ".join(CHARS.get(int(v), "?") for v in row).rstrip())
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    return "\n".join(rows)
