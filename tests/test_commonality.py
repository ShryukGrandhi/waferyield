import numpy as np
import pandas as pd

from waferyield.commonality import analyze, simulate_route

STEPS = {"litho": 4, "etch": 6, "cmp": 3, "implant": 2}


def test_finds_injected_bad_chamber():
    rng = np.random.default_rng(0)
    df = simulate_route(400, STEPS, rng, bad_step="etch", bad_tool=2, yield_hit=0.04)
    res = analyze(df, "yield", list(STEPS))
    top = res.iloc[0]
    assert top["step"] == "etch"
    assert top["worst_tool"] == "etch_T2"
    assert top["p_bonferroni"] < 1e-6
    assert 0.02 < top["median_gap"] < 0.06


def test_no_signal_when_nothing_injected():
    rng = np.random.default_rng(3)
    df = simulate_route(400, STEPS, rng)
    res = analyze(df, "yield", list(STEPS))
    assert (res["p_bonferroni"] > 0.01).all()


def test_single_tool_step_handled():
    df = pd.DataFrame({"yield": [0.9, 0.8, 0.85], "s": ["a", "a", "a"]})
    res = analyze(df, "yield", ["s"])
    assert res.loc[0, "p_value"] == 1.0
