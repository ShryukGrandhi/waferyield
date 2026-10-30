import numpy as np
import pytest

from waferyield.geometry import FAIL, OFF, PASS, WaferGeometry


def test_mask_dies_fully_inside_usable_radius():
    g = WaferGeometry(300, 10, 10, 3)
    x, y = g.die_centers_mm
    corners = np.hypot(np.abs(x[g.mask]) + 5, np.abs(y[g.mask]) + 5)
    assert corners.max() <= g.usable_radius_mm + 1e-9
    # every excluded site has at least one corner outside
    out = np.hypot(np.abs(x[~g.mask]) + 5, np.abs(y[~g.mask]) + 5)
    assert out.min() > g.usable_radius_mm


def test_mask_is_fourfold_symmetric():
    g = WaferGeometry(200, 7, 7, 2)
    m = g.mask
    assert np.array_equal(m, m[::-1, :])
    assert np.array_equal(m, m[:, ::-1])


@pytest.mark.parametrize("die", [5.0, 8.0, 10.0, 15.0])
def test_gross_die_close_to_textbook_formula(die):
    g = WaferGeometry(300, die, die, 3)
    est = g.gross_die_estimate()
    assert abs(g.gross_die - est) / est < 0.05


def test_polar_normalised():
    g = WaferGeometry()
    r, theta = g.polar
    assert r[g.mask].max() < 1.0
    assert theta.min() >= 0 and theta.max() < 2 * np.pi


def test_empty_map_codes():
    g = WaferGeometry()
    m = g.empty_map()
    assert set(np.unique(m)) == {OFF, PASS}
    assert (m == PASS).sum() == g.gross_die
    assert FAIL == 1


@pytest.mark.parametrize("kw", [dict(diameter_mm=0), dict(die_w_mm=-1), dict(edge_exclusion_mm=200)])
def test_invalid_geometry(kw):
    with pytest.raises(ValueError):
        WaferGeometry(**kw)
