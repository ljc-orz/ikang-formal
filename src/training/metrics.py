"""Binary classification metric helpers."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def select_youden_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        targets, probabilities
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = true_positive_rate - false_positive_rate
    scores[~finite] = -np.inf
    return float(thresholds[int(np.argmax(scores))])


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.size == 0:
        raise ValueError("cannot compute metrics for an empty target array")
    threshold = (
        select_youden_threshold(targets, probabilities)
        if threshold is None
        else float(threshold)
    )
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    if len(np.unique(targets)) < 2:
        auroc = math.nan
        auprc = math.nan
    else:
        auroc = float(roc_auc_score(targets, probabilities))
        auprc = float(average_precision_score(targets, probabilities))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": threshold,
    }

