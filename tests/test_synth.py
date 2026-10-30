import numpy as np

from waferyield.geometry import FAIL, OFF, WaferGeometry
from waferyield.synth import PATTERNS, RandomDefectModel, make_dataset, make_lot, make_wafer

G = WaferGeometry()


def test_off_wafer_cells_never_fail():
    rng = np.random.default_rng(0)
    for p in PATTERNS:
        m = make_wafer(p, G, rng)
        assert (m[~G.mask] == OFF).all()
        assert (m[G.mask] != OFF).all()


def test_dataset_balanced_and_reproducible():
    a, la = make_dataset(5, G, seed=7)
    b, lb = make_dataset(5, G, seed=7)
    assert np.array_equal(a, b) and np.array_equal(la, lb)
    vals, counts = np.unique(la, return_counts=True)
    assert set(vals) == set(PATTERNS) and (counts == 5).all()


def _mean_fail_by_radius(pattern, n=40):
    rng = np.random.default_rng(1)
    r, _ = G.polar
    inner = G.mask & (r < 0.3)
    outer = G.mask & (r > 0.9)
    fi = fo = 0.0
    for _ in range(n):
        m = make_wafer(pattern, G, rng) == FAIL
        fi += m[inner].mean()
        fo += m[outer].mean()
    return fi / n, fo / n


def test_center_and_edge_ring_have_opposite_radial_signatures():
    ci, co = _mean_fail_by_radius("center")
    ei, eo = _mean_fail_by_radius("edge_ring")
    assert ci > 3 * co
    assert eo > 3 * ei


def test_random_defects_match_poisson_yield_when_unclustered():
    model = RandomDefectModel(d0_per_cm2=0.4, alpha=np.inf)
    maps = make_lot(200, G, model, seed=2)
    y = (maps[:, G.mask] != FAIL).mean()
    assert abs(y - np.exp(-0.4 * G.die_area_cm2)) < 0.01


def test_fast_yield_stream_matches_map_based_lot():
    from waferyield.synth import simulate_wafer_yields
    from waferyield.yieldmodels import wafer_yield

    model = RandomDefectModel(0.3, 3.0)
    slow = np.array([wafer_yield(m) for m in make_lot(800, G, model, seed=5)])
    fast = simulate_wafer_yields(20000, G, model, np.random.default_rng(6))
    assert abs(slow.mean() - fast.mean()) < 0.01
    assert abs(slow.std() - fast.std()) / fast.std() < 0.12
