from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass
class FittedClassifier:
    name: str
    estimator: object
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError(f"Estimator {self.name} does not provide predict_proba.")
        return self.estimator.predict_proba(X)


def build_classifier(name: str, seed: int) -> object:
    if name == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "linear_svm":
        base = LinearSVC(class_weight="balanced", random_state=seed, dual="auto")
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=1)
    raise ValueError(f"Unknown classifier: {name}")


def fit_classifier(name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int) -> FittedClassifier:
    estimator = build_classifier(name, seed)
    estimator.fit(X_train, y_train)
    classes = np.asarray(estimator.classes_)
    return FittedClassifier(name=name, estimator=estimator, classes_=classes)
