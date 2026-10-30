import numpy as np

from waferyield.cli import main
from waferyield.commonality import simulate_route


def test_show(capsys):
    assert main(["show", "donut", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "pattern=donut" in out and "X" in out


def test_fit_yield(capsys):
    assert main(["fit-yield", "--wafers", "100", "--d0", "0.2", "--alpha", "2"]) == 0
    assert "fitted" in capsys.readouterr().out


def test_commonality(tmp_path, capsys):
    df = simulate_route(300, {"a": 3, "b": 3}, np.random.default_rng(0), bad_step="b", yield_hit=0.05)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    assert main(["commonality", str(p), "--steps", "a", "b"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split()[0] == "b"
