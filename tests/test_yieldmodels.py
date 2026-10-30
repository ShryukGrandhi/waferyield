import numpy as np
import pytest

from waferyield.geometry import WaferGeometry
from waferyield.synth import RandomDefectModel, make_lot
from waferyield.yieldmodels import (
    compare_models,
    fit_negbin_window,
    murphy_yield,
    negbin_yield,
    poisson_yield,
    seeds_yield,
    wafer_yield,
    window_yields,
)

G = WaferGeometry()


def test_model_ordering_and_limits():
    ad = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    p, m, s = poisson_yield(ad), murphy_yield(ad), seeds_yield(ad)
    assert p[0] == m[0] == s[0] == 1.0
    # for AD > 0: Poisson is most pessimistic, Seeds most optimistic
    assert (p[1:] < m[1:]).all() and (m[1:] < s[1:]).all()
    # negative binomial: alpha=1 is Seeds, alpha->inf is Poisson
    assert np.allclose(negbin_yield(ad, 1.0), s)
    assert np.allclose(negbin_yield(ad, 1e7), p, atol=1e-6)


def test_window_yields_monotone_decreasing():
    maps = make_lot(50, G, RandomDefectModel(0.3, 2.0), seed=0)
    k, y, n = window_yields(maps)
    assert k[0] == 1
    assert (np.diff(y) < 0).all()
    assert n[0] == 50 * G.gross_die
    assert abs(y[0] - np.mean([wafer_yield(m) for m in maps])) < 1e-12


@pytest.mark.parametrize("d0,alpha", [(0.3, 0.7), (0.3, 3.0), (0.15, 1.5)])
def test_window_method_recovers_parameters(d0, alpha):
    maps = make_lot(600, G, RandomDefectModel(d0, alpha), seed=42)
    fit = fit_negbin_window(maps, G)
    assert abs(fit.d0_per_cm2 - d0) / d0 < 0.15
    assert abs(fit.alpha - alpha) / alpha < 0.35
    assert fit.predict(G.die_area_cm2) == pytest.approx(negbin_yield(fit.d0_per_cm2, fit.alpha))


def test_negbin_fits_clustered_data_better_than_poisson():
    maps = make_lot(300, G, RandomDefectModel(0.4, 0.8), seed=1)
    errs = compare_models(maps, G)
    assert errs["negbin"] < errs["poisson"] / 5
    assert errs["negbin"] < errs["murphy"]
