"""Defect-pattern classifier: spatial features + random forest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .features import FeatureExtractor, feature_names
from .geometry import WaferGeometry


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    classes: list[str]
    recall: dict[str, float]
    confusion: np.ndarray  # rows = true, cols = predicted, ordered as `classes`

    def confusion_text(self) -> str:
        w = max(len(c) for c in self.classes)
        head = " " * (w + 2) + " ".join(f"{c[:6]:>6}" for c in self.classes)
        rows = [head]
        for c, row in zip(self.classes, self.confusion):
            rows.append(f"{c:>{w}}  " + " ".join(f"{v:6d}" for v in row))
        return "\n".join(rows)


class PatternClassifier:
    def __init__(self, geom: WaferGeometry, n_estimators: int = 300, seed: int = 0):
        self.extractor = FeatureExtractor(geom)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, maps: np.ndarray, labels: np.ndarray) -> "PatternClassifier":
        self.model.fit(self.extractor.transform(maps), labels)
        return self

    def predict(self, maps: np.ndarray) -> np.ndarray:
        return self.model.predict(self.extractor.transform(maps))

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, maps: np.ndarray, labels: np.ndarray) -> EvalReport:
        return evaluate_predictions(labels, self.predict(maps))

    def feature_importance(self) -> list[tuple[str, float]]:
        pairs = zip(feature_names(), self.model.feature_importances_)
        return sorted(pairs, key=lambda p: -p[1])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalReport:
    classes = sorted(set(y_true) | set(y_pred))
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return EvalReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        classes=classes,
        recall={c: float(v) for c, v in zip(classes, rec)},
        confusion=confusion_matrix(y_true, y_pred, labels=classes),
    )
