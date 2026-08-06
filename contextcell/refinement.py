from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class RefinementResult:
    labels: np.ndarray
    probabilities: np.ndarray
    neighbor_indices: np.ndarray


def refine_predictions(
    X: np.ndarray,
    base_probabilities: np.ndarray,
    classes: np.ndarray,
    k: int,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
    confidence_weighted: bool = True,
) -> RefinementResult:
    n = X.shape[0]
    if n < 2:
        labels = classes[np.argmax(base_probabilities, axis=1)]
        return RefinementResult(labels, base_probabilities.copy(), np.empty((n, 0), dtype=int))
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    neighbor_indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    neighbor_probs = base_probabilities[neighbor_indices]
    if confidence_weighted:
        conf = neighbor_probs.max(axis=2, keepdims=True)
        denom = np.maximum(conf.sum(axis=1), 1e-12)
        neighbor_mean = (neighbor_probs * conf).sum(axis=1) / denom
    else:
        neighbor_mean = neighbor_probs.mean(axis=1)

    mixed = self_weight * base_probabilities + neighbor_weight * neighbor_mean
    mixed = mixed / np.maximum(mixed.sum(axis=1, keepdims=True), 1e-12)
    labels = classes[np.argmax(mixed, axis=1)]
    return RefinementResult(labels, mixed, neighbor_indices)


def target_margin(probabilities: np.ndarray, target_pos: int, original_class_pos: int) -> float:
    row = probabilities[target_pos]
    original = float(row[original_class_pos])
    alternatives = np.delete(row, original_class_pos)
    return original - float(alternatives.max(initial=0.0))
