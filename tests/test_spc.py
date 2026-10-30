import numpy as np
import pytest

from waferyield.spc import Baseline, cusum_low, ewma, run_length, western_electric


def test_baseline_mr_estimate_close_to_true_sigma():
    x = np.random.default_rng(0).normal(10, 2, 5000)
    b = Baseline.from_data(x)
    assert abs(b.mean - 10) < 0.1 and abs(b.sigma - 2) < 0.1


def test_baseline_rejects_constant():
    with pytest.raises(ValueError):
        Baseline.from_data(np.ones(10))


B = Baseline(0.0, 1.0)


def _rules(x):
    return [(s.index, s.rule) for s in western_electric(np.array(x, float), B).signals]


def test_we_rule1():
    assert _rules([0, 0, 3.5, 0]) == [(2, "WE1")]


def test_we_rule2_two_of_three():
    assert _rules([2.5, 0.0, 2.5]) == [(2, "WE2")]
    assert _rules([2.5, 0.0, -2.5]) == []


def test_we_rule3_four_of_five():
    assert _rules([1.5, 1.5, 0.0, 1.5, 1.5]) == [(4, "WE3")]


def test_we_rule4_eight_same_side():
    r = _rules([0.2] * 8)
    assert r == [(7, "WE4")]
    assert _rules([0.2] * 7 + [-0.2]) == []


def test_ewma_and_cusum_detect_shift_quickly():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(-1.5, 1, 100)])
    e = run_length(ewma(x, B), start=200)
    c = run_length(cusum_low(x, B), start=200)
    assert e is not None and e < 20
    assert c is not None and c < 20


def test_cusum_ignores_upward_shift():
    x = np.full(100, 2.0)
    assert cusum_low(x, B).signals == []


def test_ewma_in_control_false_alarms_are_rare():
    x = np.random.default_rng(2).normal(0, 1, 20000)
    n_sig = len(ewma(x, B).signals)
    # nominal ARL0 ~ 500 => ~40 signals; allow wide band for autocorrelated runs
    assert n_sig < 200


def test_ewma_lambda_validation():
    with pytest.raises(ValueError):
        ewma(np.zeros(3), B, lam=0)
