"""Evaluation metrics without hiding episode-level details."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def mean_confidence_interval(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot calculate an interval for an empty list.")
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    return mean, float(1.96 * standard_error)

