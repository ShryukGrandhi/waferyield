import numpy as np
import pytest

from waferyield.features import FeatureExtractor, feature_names
from waferyield.geometry import FAIL, WaferGeometry
from waferyield.synth import make_wafer

G = WaferGeometry()
FX = FeatureExtractor(G)
IDX = {n: i for i, n in enumerate(feature_names())}


def test_vector_length_matches_names():
    v = FX(G.empty_map())
    assert v.shape == (len(feature_names()),)
    assert np.isfinite(v).all()
    assert v[IDX["fail_frac"]] == 0


def test_morans_i_checkerboard_is_negative_and_block_is_positive():
    m = G.empty_map()
    yy, xx = np.indices(G.shape)
    checker = m.copy()
    checker[G.mask & ((yy + xx) % 2 == 0)] = FAIL
    assert FX(checker)[IDX["morans_i"]] < -0.8

    block = m.copy()
    block[G.mask & (xx < G.shape[1] // 2)] = FAIL
    assert FX(block)[IDX["morans_i"]] > 0.8


def test_line_of_fails_is_elongated():
    m = G.empty_map()
    row = G.shape[0] // 2
    m[row, G.mask[row]] = FAIL
    v = FX(m)
    assert v[IDX["blob_elongation"]] > 0.99
    assert v[IDX["blob_frac_of_fails"]] == 1.0


def test_edge_ring_vs_edge_loc_sector_coverage():
    rng = np.random.default_rng(4)
    ring = np.mean([FX(make_wafer("edge_ring", G, rng))[IDX["outer_sector_cover"]] for _ in range(20)])
    loc = np.mean([FX(make_wafer("edge_loc", G, rng))[IDX["outer_sector_cover"]] for _ in range(20)])
    assert ring > 0.9 and loc < 0.5


def test_radial_corr_sign():
    rng = np.random.default_rng(5)
    assert FX(make_wafer("center", G, rng))[IDX["radial_corr"]] < 0
    assert FX(make_wafer("edge_ring", G, rng))[IDX["radial_corr"]] > 0


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        FX(np.zeros((3, 3), dtype=np.int8))
