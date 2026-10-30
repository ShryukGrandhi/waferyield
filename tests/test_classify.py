import numpy as np

from waferyield.classify import PatternClassifier, evaluate_predictions
from waferyield.geometry import WaferGeometry
from waferyield.synth import make_dataset


def test_classifier_beats_chance_by_wide_margin():
    g = WaferGeometry()
    Xtr, ytr = make_dataset(60, g, seed=10)
    Xte, yte = make_dataset(30, g, seed=11)
    clf = PatternClassifier(g, n_estimators=100).fit(Xtr, ytr)
    rep = clf.evaluate(Xte, yte)
    assert rep.accuracy > 0.8  # chance is 1/8
    for easy in ("center", "donut", "edge_ring", "random"):
        assert rep.recall[easy] > 0.9
    assert rep.confusion.sum() == len(yte)


def test_evaluate_predictions_perfect():
    y = np.array(["a", "b", "a", "c"])
    rep = evaluate_predictions(y, y)
    assert rep.accuracy == 1.0 and rep.macro_f1 == 1.0
    assert np.array_equal(np.diag(rep.confusion), [2, 1, 1])
    assert "a" in rep.confusion_text()
